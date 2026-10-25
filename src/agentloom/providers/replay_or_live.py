"""Replay a run up to a cut point and run the rest live (#165).

``ReplayCut`` decides, call by call, whether the recording or the live provider
serves a step, and logs where each call came from. ``ReplayOrLiveProvider`` wraps
every live provider of the gateway, as ``RecordingProvider`` does; the tool
replayer asks the same cut, so a step that runs live runs its tools live too.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from pydantic import BaseModel

from agentloom.core.attempts import running_steps
from agentloom.exceptions import RecordingMismatchError
from agentloom.providers.base import (
    BaseProvider,
    EmbeddingResponse,
    ProviderResponse,
    StreamResponse,
)
from agentloom.providers.mock import MockProvider

if TYPE_CHECKING:
    from agentloom.core.models import WorkflowDefinition

logger = logging.getLogger("agentloom.providers")

T = TypeVar("T")


class CallSource(BaseModel):
    """Where one call of a step came from: the recording (``replay``) or the provider (``live``)."""

    step_id: str
    call: int  # from 1, per step
    source: Literal["replay", "live"]
    reason: Literal["hit", "forced", "miss"]


class ReplayCut:
    """Which steps run live, and where every call so far came from.

    A ``forced`` step runs live, with any subworkflow it runs. Another step is
    served from the recording until a call misses it (its request changed, or it
    makes more calls than were recorded); then it runs live, and its dependents,
    whose prompts change, miss on their own.
    """

    def __init__(self, recording: MockProvider, forced: Iterable[str] = ()) -> None:
        self.recording = recording
        self.forced = frozenset(forced)
        self.calls: list[CallSource] = []
        self._missed: set[str] = set()
        self._numbers: dict[str, int] = {}

    def _reason(self, step_id: str) -> Literal["forced", "miss"] | None:
        # A step's own extra calls (``<id>::summarize``) follow the step.
        own = step_id.split("::", 1)[0]
        if self.forced.intersection((*running_steps(), own)):
            return "forced"
        return "miss" if own in self._missed else None

    def is_live(self, step_id: str) -> bool:
        """Whether ``step_id`` runs live from now on."""
        return self._reason(step_id) is not None

    async def route(
        self,
        step_id: str | None,
        replay: Callable[[], Awaitable[T]],
        live: Callable[[], Awaitable[T]],
    ) -> T:
        """Serve one call from the recording, or from the provider when the cut says so."""
        key = step_id or ""
        number = self._numbers[key] = self._numbers.get(key, 0) + 1
        reason = self._reason(key)
        if reason is None:
            try:
                served = await replay()
            except RecordingMismatchError as exc:
                if step_id:
                    self._missed.add(step_id.split("::", 1)[0])
                logger.info("Step %r leaves the recording at call %d: %s", key, number, exc)
                reason = "miss"
            else:
                self.calls.append(
                    CallSource(step_id=key, call=number, source="replay", reason="hit")
                )
                return served
        self.calls.append(CallSource(step_id=key, call=number, source="live", reason=reason))
        return await live()

    def unused_turns(self) -> dict[str, tuple[int, int]]:
        """Steps served only from the recording that left turns unused: ``(served, recorded)``."""
        served = Counter(call.step_id for call in self.calls if call.source == "replay")
        turns = {step: (n, self.recording.recorded_turns(step)) for step, n in served.items()}
        return {step: t for step, t in turns.items() if t[0] < t[1] and not self.is_live(step)}

    def summary(self) -> list[str]:
        """The call log and any unused turns, one line each, for the CLI."""
        lines = [f"  {c.step_id} #{c.call}: {c.source} ({c.reason})" for c in self.calls]
        lines += [
            f"  {step}: served {count} of {recorded} recorded turns"
            for step, (count, recorded) in self.unused_turns().items()
        ]
        return lines


def cut_steps(
    workflow: WorkflowDefinition, live_from: str | None, force_live: Iterable[str]
) -> set[str]:
    """The top-level steps forced live: ``live_from``, the steps after it, and ``force_live``."""
    from agentloom.core.parser import WorkflowParser

    forced = set(force_live)
    named = forced | ({live_from} if live_from else set())
    if unknown := sorted(named - {step.id for step in workflow.steps}):
        raise ValueError(f"No step {', '.join(map(repr, unknown))} in workflow {workflow.name!r}")
    if live_from:
        forced |= WorkflowParser.build_dag(workflow).transitive_successors({live_from})
    return forced


class ReplayOrLiveProvider(BaseProvider):
    """A live provider whose calls the recording serves while the cut allows."""

    # The cut and the recording key calls by step id; the live adapter never sees it.
    accepts_step_id = True

    def __init__(self, wrapped: BaseProvider, cut: ReplayCut) -> None:
        # Every call goes to ``wrapped`` or to the recording, so the wrapper keeps no credentials.
        super().__init__(base_url=wrapped.base_url)
        self.name = wrapped.name
        self._wrapped = wrapped
        self._cut = cut

    def _live(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if getattr(self._wrapped, "accepts_step_id", False):
            return kwargs
        return {k: v for k, v in kwargs.items() if k != "step_id"}

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        call: dict[str, Any] = {"messages": messages, "model": model, "temperature": temperature}
        call["max_tokens"] = max_tokens
        return await self._cut.route(
            kwargs.get("step_id"),
            lambda: self._cut.recording.complete(**call, **kwargs),
            lambda: self._wrapped.complete(**call, **self._live(kwargs)),
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> StreamResponse:
        call: dict[str, Any] = {"messages": messages, "model": model, "temperature": temperature}
        call["max_tokens"] = max_tokens
        return await self._cut.route(
            kwargs.get("step_id"),
            lambda: self._cut.recording.stream(**call, **kwargs),
            lambda: self._wrapped.stream(**call, **self._live(kwargs)),
        )

    async def embed(
        self,
        inputs: list[str],
        model: str,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        call: dict[str, Any] = {"inputs": inputs, "model": model, "dimensions": dimensions}
        return await self._cut.route(
            kwargs.get("step_id"),
            lambda: self._cut.recording.embed(**call, **kwargs),
            lambda: self._wrapped.embed(**call, **self._live(kwargs)),
        )

    def supports_model(self, model: str) -> bool:
        return self._wrapped.supports_model(model)

    async def close(self) -> None:
        await self._wrapped.close()
