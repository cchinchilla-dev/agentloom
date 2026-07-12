"""Conversation helpers for the ``llm_call`` step.

Kept out of ``llm_call.py`` so trimming policies stay independently
testable and re-usable by the (upcoming) Agent primitive without pulling
in the full step-execution surface.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from agentloom.core.models import Conversation, Message

logger = logging.getLogger("agentloom.steps.conversation")

# A summariser is either a plain sync callable (used mainly by tests) or an
# awaitable one — the ``llm_call`` step passes an async wrapper that hits the
# gateway. Returning an empty string / raising both trigger the deterministic
# concat fallback so the budget contract still holds.
Summarizer = Callable[[list[Message]], str | Awaitable[str]]


@dataclass(slots=True)
class TrimResult:
    """Outcome of a single trim pass.

    ``trimmed_count`` counts messages *removed* from the visible list
    (either dropped or folded into ``summary``). ``summarized_count``
    counts messages that specifically ended up rolled into
    ``Conversation.summary``. ``policy`` is echoed back for the observer
    hook so the metric label matches what actually ran (a no-op returns
    ``policy=""`` — see :meth:`Conversation.token_count` for the
    heuristic notes).
    """

    trimmed_count: int = 0
    summarized_count: int = 0
    policy: str = ""


def load_conversation(raw: Any) -> Conversation:
    """Normalize a state value into a :class:`Conversation`.

    Accepts:

    - ``None`` / missing → fresh empty conversation.
    - A :class:`Conversation` instance → returned as-is.
    - A ``dict`` with (at least) a ``messages`` key → validated through
      Pydantic. Extra keys raise via the model's ``extra="forbid"``, so
      typos in the YAML (``token_budgt:``) fail loudly instead of being
      silently dropped.
    - A ``list`` → treated as the messages payload with default budget /
      policy. Lets a workflow author write ``chat: [{role: user, ...}]``
      without the surrounding envelope for the common case.

    Any other shape raises ``TypeError`` — surfacing a workflow bug on
    the offending step rather than the provider call.
    """
    if raw is None:
        return Conversation()
    if isinstance(raw, Conversation):
        return raw
    if isinstance(raw, dict):
        return Conversation.model_validate(raw)
    if isinstance(raw, list):
        return Conversation.model_validate({"messages": raw})
    raise TypeError(
        f"Conversation state must be a dict / list / Conversation, got {type(raw).__name__}"
    )


def _split_pinned(messages: list[Message]) -> tuple[list[Message], list[Message]]:
    """Split the ``system`` prefix from the trimmable tail.

    Every ``role="system"`` message from the head is pinned; the moment a
    non-system message appears, the rest is considered fair game for
    trimming even if it contains a later system message (rare — usually
    a workflow bug we'd rather surface than silently retain).
    """
    pinned: list[Message] = []
    for i, msg in enumerate(messages):
        if msg.role != "system":
            return pinned, messages[i:]
        pinned.append(msg)
    return pinned, []


def _drop_oldest(convo: Conversation, model: str) -> TrimResult:
    """Drop from the head until the budget is met.

    Pins the leading system prefix so the model always sees its
    instruction turn. If the tail is empty, only the pinned messages
    remain — nothing more we can shed without breaking the contract,
    so the pass exits with a warning and leaves the budget over-limit.
    """
    if convo.token_budget is None:
        return TrimResult(policy="drop_oldest")
    pinned, tail = _split_pinned(convo.messages)
    pinned_tokens = sum(m.token_count(model) for m in pinned)
    if pinned_tokens >= convo.token_budget:
        logger.warning(
            "Conversation trim: pinned system prefix (%d tokens) already exceeds "
            "budget (%d); nothing to drop. Raise token_budget or shorten the system prompt.",
            pinned_tokens,
            convo.token_budget,
        )
        convo.messages = [*pinned, *tail]
        return TrimResult(policy="drop_oldest")
    trimmed = 0
    while tail and (pinned_tokens + sum(m.token_count(model) for m in tail)) > convo.token_budget:
        tail.pop(0)
        trimmed += 1
    convo.messages = [*pinned, *tail]
    return TrimResult(trimmed_count=trimmed, policy="drop_oldest")


def _drop_pairs(convo: Conversation, model: str) -> TrimResult:
    """Drop oldest ``user``+``assistant`` pairs together.

    Preserves the conversation's turn cadence: dropping only the user
    half or only the assistant half of a pair leaves the model with a
    dangling reply / bare question. When the leading trimmable message
    is *not* a user turn (rare — e.g. tool-only continuations), fall
    through to the single-message drop the ``drop_oldest`` policy uses
    so the pass still makes forward progress.
    """
    if convo.token_budget is None:
        return TrimResult(policy="drop_pairs")
    pinned, tail = _split_pinned(convo.messages)
    trimmed = 0
    pinned_tokens = sum(m.token_count(model) for m in pinned)
    while tail and (pinned_tokens + sum(m.token_count(model) for m in tail)) > convo.token_budget:
        head = tail[0]
        if head.role == "user" and len(tail) > 1 and tail[1].role == "assistant":
            tail.pop(0)
            tail.pop(0)
            trimmed += 2
        else:
            tail.pop(0)
            trimmed += 1
    convo.messages = [*pinned, *tail]
    return TrimResult(trimmed_count=trimmed, policy="drop_pairs")


def _plan_summarize_oldest(convo: Conversation, model: str) -> list[Message]:
    """Compute the messages the ``summarize_oldest`` policy would fold.

    Split out from :func:`_summarize_oldest` so the caller can run the
    summariser LLM call *before* mutating the conversation — the pass is
    idempotent when no summarisation is needed and stays synchronous
    inside ``apply_trim_policy``.
    """
    if convo.token_budget is None:
        return []
    pinned, tail = _split_pinned(convo.messages)
    pinned_tokens = sum(m.token_count(model) for m in pinned)
    if pinned_tokens >= convo.token_budget or not tail:
        return []
    to_summarize: list[Message] = []
    working = list(tail)
    while (
        working
        and (pinned_tokens + sum(m.token_count(model) for m in working)) > convo.token_budget
    ):
        to_summarize.append(working.pop(0))
    return to_summarize


def _apply_summary(convo: Conversation, folded: list[Message], summary_text: str) -> TrimResult:
    """Attach ``summary_text`` to the conversation and drop ``folded`` turns.

    When ``summary_text`` is empty (no summariser, or the summariser
    failed) the fallback is a deterministic ``"<speaker>: <body>"``
    concatenation of the folded turns — lossy but budget-respecting.
    """
    if not folded:
        return TrimResult(policy="summarize_oldest")
    if not summary_text:
        parts = []
        for m in folded:
            speaker = m.name or m.role
            body = (m.content or "").strip()
            if body:
                parts.append(f"{speaker}: {body}")
        summary_text = "\n".join(parts)
    if convo.summary:
        convo.summary = f"{convo.summary}\n{summary_text}".strip()
    else:
        convo.summary = summary_text.strip()
    # Remove the folded turns POSITIONALLY, not by value-equality. The
    # folded set is always the leading ``len(folded)`` messages of the
    # trimmable tail (``_plan_summarize_oldest`` pops them from the front,
    # and nothing mutates ``convo.messages`` between planning and applying).
    # A ``[m for m in messages if m not in folded]`` filter would also drop
    # a value-equal duplicate that survived in the retained tail — e.g. a
    # second ``user: "yes"`` turn — silently losing it. Slicing by count
    # keeps identity-independent and duplicate-safe.
    pinned, tail = _split_pinned(convo.messages)
    convo.messages = [*pinned, *tail[len(folded) :]]
    return TrimResult(
        trimmed_count=len(folded),
        summarized_count=len(folded),
        policy="summarize_oldest",
    )


def _summarize_oldest(convo: Conversation, model: str, summarizer: Summarizer | None) -> TrimResult:
    """Sync variant of ``summarize_oldest`` — kept for tests + non-LLM callers.

    When *summarizer* is ``None`` (or returns an empty string) the policy
    still runs the folded-turn concat fallback so the budget contract
    holds. Async summarisers must be run through :func:`apply_trim_policy_async`
    — this helper never awaits.
    """
    folded = _plan_summarize_oldest(convo, model)
    if not folded:
        return TrimResult(policy="summarize_oldest")
    summary_text = ""
    if summarizer is not None:
        try:
            result = summarizer(folded)
        except Exception as exc:  # pragma: no cover — summariser is workflow-provided
            logger.warning(
                "Conversation summarize_oldest policy: summarizer raised %s; "
                "falling back to a plain concatenation.",
                type(exc).__name__,
            )
            result = ""
        if inspect.isawaitable(result):
            logger.warning(
                "Conversation summarize_oldest policy: an async summarizer was "
                "passed to the sync apply_trim_policy; ignoring and falling back "
                "to concat. Use apply_trim_policy_async instead."
            )
            # Close the pending coroutine so we don't leak a warning about it.
            close = getattr(result, "close", None)
            if callable(close):
                close()
            result = ""
        summary_text = result or ""
    return _apply_summary(convo, folded, summary_text)


def apply_trim_policy(
    convo: Conversation,
    model: str,
    *,
    summarizer: Summarizer | None = None,
) -> TrimResult:
    """Run the configured trim policy in-place on ``convo``.

    Sync-only entry point. Async summarisers must be routed through
    :func:`apply_trim_policy_async`. Returns a :class:`TrimResult` even
    when the policy is a no-op so the observer can decide whether to
    emit the ``conversation_trims_total`` counter — a no-op call carries
    ``policy=""``.
    """
    if convo.token_budget is None:
        return TrimResult()
    if convo.token_count(model) <= convo.token_budget:
        return TrimResult()
    if convo.trim_policy == "drop_oldest":
        return _drop_oldest(convo, model)
    if convo.trim_policy == "drop_pairs":
        return _drop_pairs(convo, model)
    if convo.trim_policy == "summarize_oldest":
        return _summarize_oldest(convo, model, summarizer)
    # Pydantic already rejects any other value at parse time, but be
    # explicit here so a future extra policy that isn't wired lands as a
    # loud failure rather than a silent no-op.
    raise ValueError(f"Unknown trim_policy: {convo.trim_policy!r}")


async def apply_trim_policy_async(
    convo: Conversation,
    model: str,
    *,
    summarizer: Summarizer | None = None,
) -> TrimResult:
    """Async trim entry-point — awaits the summariser for ``summarize_oldest``.

    ``drop_oldest`` / ``drop_pairs`` delegate to the sync path (they don't
    need I/O). ``summarize_oldest`` computes which turns would be folded,
    invokes the summariser (awaiting when it returns an awaitable), then
    attaches the result. When the summariser raises the trim still runs
    with the deterministic concat fallback so the budget contract holds.
    """
    if convo.token_budget is None:
        return TrimResult()
    if convo.token_count(model) <= convo.token_budget:
        return TrimResult()
    if convo.trim_policy != "summarize_oldest":
        return apply_trim_policy(convo, model, summarizer=None)
    folded = _plan_summarize_oldest(convo, model)
    if not folded:
        return TrimResult(policy="summarize_oldest")
    summary_text = ""
    if summarizer is not None:
        try:
            result = summarizer(folded)
            if inspect.isawaitable(result):
                result = await result
            summary_text = result or ""
        except Exception as exc:
            logger.warning(
                "Conversation summarize_oldest policy: summarizer raised %s; "
                "falling back to a plain concatenation.",
                type(exc).__name__,
            )
    return _apply_summary(convo, folded, summary_text)


def to_provider_messages(convo: Conversation) -> list[dict[str, Any]]:
    """Serialize the conversation into the dict shape providers expect.

    Drops ``metadata`` and other AgentLoom-only fields — they'd be
    rejected by ``validate_extra_kwargs`` on providers with strict
    request bodies.

    When ``conversation.summary`` is present it is injected as a ``system``
    turn *after* the conversation's own leading system prefix, not before
    it. The workflow's real system prompt must stay the highest-priority
    instruction — placing a model-generated summary ahead of it would let
    summary text (which can contain instruction-like phrasing) override
    the operative system prompt. The summary is a clearly-labelled context
    block that sits between the system prefix and the first non-system turn.
    """
    out: list[dict[str, Any]] = []
    summary_entry: dict[str, Any] | None = (
        {"role": "system", "content": f"Summary of earlier turns:\n{convo.summary}"}
        if convo.summary
        else None
    )
    summary_inserted = summary_entry is None
    for msg in convo.messages:
        if not summary_inserted and msg.role != "system":
            # First non-system turn — drop the summary in ahead of it so it
            # trails the pinned system prefix.
            out.append(summary_entry)  # type: ignore[arg-type]
            summary_inserted = True
        entry: dict[str, Any] = {"role": msg.role, "content": msg.content}
        if msg.name:
            entry["name"] = msg.name
        if msg.tool_call_id:
            entry["tool_call_id"] = msg.tool_call_id
        if msg.tool_calls:
            entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        out.append(entry)
    if not summary_inserted:
        # Conversation was empty or all-system — append the summary at the end.
        out.append(summary_entry)  # type: ignore[arg-type]
    return out
