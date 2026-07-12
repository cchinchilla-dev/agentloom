"""Regression tests for the Conversation primitive (#119)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agentloom.core.models import (
    Attachment,
    Conversation,
    Message,
    StepDefinition,
    StepType,
    ToolCallSpec,
    WorkflowConfig,
    WorkflowDefinition,
)
from agentloom.core.parser import WorkflowParser
from agentloom.core.results import StepStatus, TokenUsage
from agentloom.core.state import StateManager, _json_default
from agentloom.observability.metrics import MetricsManager
from agentloom.observability.observer import WorkflowObserver
from agentloom.observability.schema import SpanAttr
from agentloom.providers.base import ProviderResponse, ToolCall
from agentloom.providers.gateway import ProviderGateway
from agentloom.steps._conversation import (
    _apply_summary,
    _drop_oldest,
    _drop_pairs,
    _summarize_oldest,
    apply_trim_policy,
    apply_trim_policy_async,
    load_conversation,
    to_provider_messages,
)
from agentloom.steps.base import StepContext
from agentloom.steps.llm_call import LLMCallStep
from tests.conftest import MockProvider

# ---- Model layer ------------------------------------------------------------


class TestMessageModel:
    def test_defaults_are_sane(self) -> None:
        m = Message(role="user", content="hello")
        assert m.name is None
        assert m.tool_call_id is None
        assert m.tool_calls == []
        assert m.metadata == {}

    def test_rejects_unknown_role(self) -> None:
        with pytest.raises(ValueError, match="role"):
            Message(role="assistent", content="typo")  # type: ignore[arg-type]

    def test_extra_field_forbidden(self) -> None:
        with pytest.raises(ValueError):
            Message(role="user", content="hi", nickname="alice")  # type: ignore[call-arg]

    def test_token_count_grows_with_content(self) -> None:
        short = Message(role="user", content="hi").token_count()
        long = Message(role="user", content="hi " * 400).token_count()
        assert long > short

    def test_token_count_accounts_for_tool_calls(self) -> None:
        plain = Message(role="assistant", content="ok").token_count()
        with_call = Message(
            role="assistant",
            content="ok",
            tool_calls=[
                ToolCallSpec(
                    id="c1",
                    name="lookup_user",
                    arguments={"id": 123, "verbose": True},
                )
            ],
        ).token_count()
        assert with_call > plain

    def test_token_count_never_zero(self) -> None:
        assert Message(role="user", content="").token_count() >= 1


class TestConversationModel:
    def test_defaults(self) -> None:
        c = Conversation()
        assert c.messages == []
        assert c.token_budget is None
        assert c.trim_policy == "drop_oldest"

    def test_rejects_zero_budget(self) -> None:
        with pytest.raises(ValueError):
            Conversation(token_budget=0)

    def test_rejects_unknown_policy(self) -> None:
        with pytest.raises(ValueError):
            Conversation(trim_policy="chop-in-half")  # type: ignore[arg-type]

    def test_coerces_dict_messages_from_yaml(self) -> None:
        c = Conversation.model_validate(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            }
        )
        assert len(c.messages) == 2
        assert isinstance(c.messages[0], Message)

    def test_roundtrip_to_dict(self) -> None:
        c = Conversation(
            messages=[Message(role="user", content="hi", name="alice")],
            token_budget=100,
            trim_policy="drop_pairs",
        )
        data = c.model_dump()
        restored = Conversation.model_validate(data)
        assert restored == c


# ---- Trimming policies ------------------------------------------------------


class TestTrimmingPolicies:
    @staticmethod
    def _make(msgs: list[Message], **overrides: Any) -> Conversation:
        # Big enough words to bump the token budget cheaply.
        return Conversation(messages=msgs, **overrides)

    def test_drop_oldest_preserves_system_prefix(self) -> None:
        convo = self._make(
            [
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="q1"),
                Message(role="assistant", content="a1"),
                Message(role="user", content="q2"),
                Message(role="assistant", content="a2"),
            ],
            token_budget=15,
            trim_policy="drop_oldest",
        )
        result = apply_trim_policy(convo, model="mock-model")
        assert convo.messages[0].role == "system"
        assert result.trimmed_count > 0

    def test_drop_pairs_removes_user_assistant_together(self) -> None:
        convo = self._make(
            [
                Message(role="user", content="q1 " * 30),
                Message(role="assistant", content="a1 " * 30),
                Message(role="user", content="q2"),
                Message(role="assistant", content="a2"),
            ],
            token_budget=10,
            trim_policy="drop_pairs",
        )
        apply_trim_policy(convo, model="mock-model")
        # Both q1 and a1 are dropped together
        assert not any(m.content.startswith("q1") for m in convo.messages)
        assert not any(m.content.startswith("a1") for m in convo.messages)

    def test_drop_pairs_falls_back_when_head_not_user(self) -> None:
        convo = self._make(
            [
                Message(role="assistant", content="a0 " * 30),
                Message(role="user", content="q1"),
            ],
            token_budget=6,
            trim_policy="drop_pairs",
        )
        result = apply_trim_policy(convo, model="mock-model")
        assert result.trimmed_count >= 1

    def test_summarize_oldest_produces_summary_and_shortens_list(self) -> None:
        convo = self._make(
            [
                Message(role="system", content="Be brief."),
                Message(role="user", content="q1 " * 40),
                Message(role="assistant", content="a1 " * 40),
                Message(role="user", content="q2"),
                Message(role="assistant", content="a2"),
            ],
            token_budget=25,
            trim_policy="summarize_oldest",
        )
        result = apply_trim_policy(convo, model="mock-model")
        assert convo.summary is not None
        assert result.summarized_count > 0
        assert convo.messages[0].role == "system"

    def test_summarize_oldest_uses_provided_summarizer(self) -> None:
        seen: list[list[Message]] = []

        def summarizer(msgs: list[Message]) -> str:
            seen.append(msgs)
            return "SHORT SUMMARY"

        convo = self._make(
            [
                Message(role="user", content="q1 " * 30),
                Message(role="assistant", content="a1 " * 30),
                Message(role="user", content="q2"),
            ],
            token_budget=10,
            trim_policy="summarize_oldest",
        )
        apply_trim_policy(convo, model="mock-model", summarizer=summarizer)
        assert seen  # summarizer was invoked
        assert "SHORT SUMMARY" in (convo.summary or "")

    def test_no_budget_is_noop(self) -> None:
        convo = self._make(
            [Message(role="user", content="hi")],
            token_budget=None,
        )
        result = apply_trim_policy(convo, model="mock-model")
        assert result.trimmed_count == 0
        assert len(convo.messages) == 1

    def test_within_budget_is_noop(self) -> None:
        convo = self._make(
            [Message(role="user", content="hi")],
            token_budget=100,  # far above the message's token estimate
        )
        result = apply_trim_policy(convo, model="mock-model")
        assert result.trimmed_count == 0
        assert result.policy == ""

    def test_summarize_oldest_falls_back_to_concat(self) -> None:
        # No summarizer passed → policy folds oldest turns into a plain
        # concatenation prefixed with speakers.
        convo = self._make(
            [
                Message(role="user", content="q1 " * 30),
                Message(role="assistant", content="a1 " * 30),
                Message(role="user", content="q2"),
            ],
            token_budget=10,
            trim_policy="summarize_oldest",
        )
        apply_trim_policy(convo, model="mock-model")
        assert convo.summary
        # Fallback concatenation must include speaker markers.
        assert "user:" in (convo.summary or "")

    def test_summarize_oldest_does_not_drop_duplicate_retained_turn(self) -> None:
        # Regression: folded turns must be removed POSITIONALLY, not by
        # value-equality. A retained turn that is value-equal to a folded
        # one (repeated "yes"/"thanks"/"ok") must survive. Removing by
        # ``__eq__`` silently deleted the retained duplicate.
        convo = self._make(
            [
                Message(role="system", content="Be brief."),
                Message(role="user", content="yes"),
                Message(role="assistant", content="a1 " * 60),
                Message(role="user", content="yes"),  # value-equal to the folded one
                Message(role="assistant", content="keep me"),
            ],
            token_budget=30,
            trim_policy="summarize_oldest",
        )
        result = apply_trim_policy(convo, model="mock-model")
        # The system turn is pinned; exactly the leading folded turns go.
        assert convo.messages[0].role == "system"
        # The second "yes" and the final "keep me" must both survive.
        contents = [m.content for m in convo.messages]
        assert "keep me" in contents
        assert contents.count("yes") == 1  # one survived, one folded
        # trimmed_count reflects what was actually removed positionally.
        removed = 5 - len(convo.messages)
        assert result.trimmed_count == removed

    def test_summarize_over_pinned_prefix_is_noop(self) -> None:
        # Pinned system prefix already exceeds budget — nothing left to
        # fold; the policy must return without changing the summary.
        convo = self._make(
            [
                Message(role="system", content="verbose " * 100),
                Message(role="user", content="q1"),
            ],
            token_budget=5,
            trim_policy="summarize_oldest",
        )
        result = apply_trim_policy(convo, model="mock-model")
        assert result.summarized_count == 0
        assert convo.summary is None

    def test_pinned_prefix_over_budget_logs_warning(self) -> None:
        # System message alone exceeds the token budget; nothing to trim.
        convo = self._make(
            [Message(role="system", content="verbose " * 100)],
            token_budget=5,
        )
        result = apply_trim_policy(convo, model="mock-model")
        assert result.trimmed_count == 0

    def test_summarize_appends_to_existing_summary(self) -> None:
        convo = self._make(
            [
                Message(role="user", content="q1 " * 30),
                Message(role="assistant", content="a1 " * 30),
                Message(role="user", content="q2"),
            ],
            token_budget=10,
            trim_policy="summarize_oldest",
            summary="prev summary line",
        )
        apply_trim_policy(convo, model="mock-model")
        assert (convo.summary or "").startswith("prev summary line")


class TestLoadAndProviderShape:
    def test_load_from_none(self) -> None:
        c = load_conversation(None)
        assert isinstance(c, Conversation)
        assert c.messages == []

    def test_load_from_dict(self) -> None:
        c = load_conversation({"messages": [{"role": "user", "content": "hi"}]})
        assert len(c.messages) == 1

    def test_load_from_list_shorthand(self) -> None:
        c = load_conversation([{"role": "user", "content": "hi"}])
        assert isinstance(c, Conversation)
        assert c.messages[0].content == "hi"

    def test_load_from_conversation_instance_passes_through(self) -> None:
        base = Conversation(messages=[Message(role="user", content="hi")])
        assert load_conversation(base) is base

    def test_load_rejects_unknown_type(self) -> None:
        with pytest.raises(TypeError):
            load_conversation(42)

    def test_to_provider_messages_includes_summary_prefix(self) -> None:
        c = Conversation(
            messages=[Message(role="user", content="q")],
            summary="earlier context",
        )
        wire = to_provider_messages(c)
        assert wire[0]["role"] == "system"
        assert "earlier context" in wire[0]["content"]

    def test_to_provider_messages_carries_name_field(self) -> None:
        c = Conversation(messages=[Message(role="user", content="hi", name="alice")])
        assert to_provider_messages(c)[0]["name"] == "alice"

    def test_to_provider_messages_carries_tool_call_id(self) -> None:
        # A ``role="tool"`` turn must round-trip its ``tool_call_id`` so a
        # resumed multi-turn tool workflow stays OpenAI-wire-compatible.
        c = Conversation(
            messages=[
                Message(role="tool", content="42", tool_call_id="call_abc"),
            ]
        )
        wire = to_provider_messages(c)
        assert wire[0]["tool_call_id"] == "call_abc"
        assert wire[0]["role"] == "tool"

    def test_to_provider_messages_serializes_tool_calls(self) -> None:
        c = Conversation(
            messages=[
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCallSpec(id="c1", name="fn", arguments={"k": 1})],
                )
            ]
        )
        wire = to_provider_messages(c)
        assert wire[0]["tool_calls"] == [{"id": "c1", "name": "fn", "arguments": {"k": 1}}]


# ---- StateManager checkpoint roundtrip -------------------------------------


class TestConversationStateRoundtrip:
    async def test_conversation_state_serializes_roundtrip(self, tmp_path: Path) -> None:
        sm = StateManager(
            initial_state={
                "chat": Conversation(
                    messages=[
                        Message(role="system", content="You are helpful."),
                        Message(role="user", content="hi", name="alice"),
                    ]
                ).model_dump()
            }
        )
        checkpoint = tmp_path / "state.json"
        await sm.save_checkpoint(checkpoint)
        payload = json.loads(checkpoint.read_text())
        assert payload["state"]["chat"]["messages"][1]["name"] == "alice"

        restored = await StateManager.from_checkpoint(checkpoint)
        got = await restored.get("chat")
        c = Conversation.model_validate(got)
        assert c.messages[1].name == "alice"

    async def test_conversation_resumes_correctly_from_checkpoint(self, tmp_path: Path) -> None:
        # Simulate: a workflow ran two turns, checkpointed, and now we
        # resume — the loaded Conversation must carry both turns intact.
        initial = Conversation(
            messages=[
                Message(role="system", content="Be concise."),
                Message(role="user", content="q1"),
                Message(role="assistant", content="a1"),
            ],
            token_budget=200,
            trim_policy="drop_pairs",
        )
        sm = StateManager(initial_state={"chat": initial.model_dump()})
        checkpoint = tmp_path / "resume.json"
        await sm.save_checkpoint(checkpoint)

        restored = await StateManager.from_checkpoint(checkpoint)
        got = await restored.get("chat")
        c = Conversation.model_validate(got)
        assert c.trim_policy == "drop_pairs"
        assert c.token_budget == 200
        assert [m.role for m in c.messages] == ["system", "user", "assistant"]


# ---- llm_call integration ---------------------------------------------------


def _make_context(
    step_def: StepDefinition,
    *,
    state: dict[str, Any] | None = None,
    gateway: ProviderGateway | None = None,
    observer: Any | None = None,
) -> StepContext:
    return StepContext(
        step_definition=step_def,
        state_manager=StateManager(initial_state=state or {}),
        provider_gateway=gateway,
        observer=observer,
        workflow_model="mock-model",
    )


class TestLLMCallWithConversation:
    async def test_appends_user_and_assistant_messages(self) -> None:
        provider = MockProvider(responses={"What is 2+2?": "4"})
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        ctx = _make_context(
            StepDefinition(
                id="turn",
                type=StepType.LLM_CALL,
                prompt="What is 2+2?",
                conversation="state.chat",
            ),
            state={"chat": {"messages": []}},
            gateway=gw,
        )
        result = await LLMCallStep().execute(ctx)
        assert result.status == StepStatus.SUCCESS

        chat = await ctx.state_manager.get("chat")
        restored = Conversation.model_validate(chat)
        assert [m.role for m in restored.messages] == ["user", "assistant"]
        assert restored.messages[0].content == "What is 2+2?"
        assert restored.messages[1].content == "4"

    async def test_without_prompt_uses_existing_messages(self) -> None:
        provider = MockProvider(responses={"existing?": "yes"})
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        ctx = _make_context(
            StepDefinition(
                id="turn",
                type=StepType.LLM_CALL,
                conversation="state.chat",
            ),
            state={
                "chat": {
                    "messages": [{"role": "user", "content": "existing?"}],
                }
            },
            gateway=gw,
        )
        result = await LLMCallStep().execute(ctx)
        assert result.status == StepStatus.SUCCESS
        chat = Conversation.model_validate(await ctx.state_manager.get("chat"))
        # user preserved + assistant appended
        assert [m.role for m in chat.messages] == ["user", "assistant"]

    async def test_missing_conversation_creates_new(self) -> None:
        provider = MockProvider(responses={"hi": "hello"})
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        ctx = _make_context(
            StepDefinition(
                id="turn",
                type=StepType.LLM_CALL,
                prompt="hi",
                conversation="state.chat",
            ),
            state={},
            gateway=gw,
        )
        result = await LLMCallStep().execute(ctx)
        assert result.status == StepStatus.SUCCESS
        chat = Conversation.model_validate(await ctx.state_manager.get("chat"))
        assert len(chat.messages) == 2

    async def test_tool_calls_not_persisted_replay_safe(self) -> None:
        # Replay safety: an assistant turn carrying tool_calls is only valid
        # on the wire when followed by its paired tool-result turns. The
        # conversation drops those, so it must NOT persist the tool_calls
        # either — otherwise the next llm_call would render a malformed
        # request. Only the final text answer is stored.
        class ToolProvider(MockProvider):
            async def complete(
                self,
                messages: list[dict[str, Any]],
                model: str,
                temperature: float | None = None,
                max_tokens: int | None = None,
                **kwargs: Any,
            ) -> ProviderResponse:
                return ProviderResponse(
                    content="It is noon UTC.",
                    model=model,
                    provider="mock",
                    usage=TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
                    cost_usd=0.001,
                    tool_calls=[ToolCall(id="c1", name="get_time", arguments={"tz": "UTC"})],
                )

        provider = ToolProvider()
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        ctx = _make_context(
            StepDefinition(
                id="turn",
                type=StepType.LLM_CALL,
                prompt="what time is it?",
                conversation="state.chat",
            ),
            state={"chat": {"messages": []}},
            gateway=gw,
        )
        await LLMCallStep().execute(ctx)

        chat = Conversation.model_validate(await ctx.state_manager.get("chat"))
        assert chat.messages[-1].role == "assistant"
        assert chat.messages[-1].content == "It is noon UTC."
        # tool_calls are NOT persisted (replay safety).
        assert chat.messages[-1].tool_calls == []

    async def test_multi_agent_name_propagates_to_openai_payload(self) -> None:
        provider = MockProvider(responses={"who": "assistant reply"})
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        ctx = _make_context(
            StepDefinition(
                id="turn",
                type=StepType.LLM_CALL,
                prompt="who",
                conversation="state.chat",
            ),
            state={
                "chat": {
                    "messages": [
                        {"role": "user", "content": "prev", "name": "alice"},
                    ]
                }
            },
            gateway=gw,
        )
        await LLMCallStep().execute(ctx)
        # The provider's stored payload should carry the ``name`` from the
        # loaded conversation on the earliest message.
        wire = provider.calls[0]["messages"]
        assert wire[0]["name"] == "alice"

    async def test_summary_ridealong_survives_next_turn(self) -> None:
        provider = MockProvider(responses={"newq": "answer"})
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        ctx = _make_context(
            StepDefinition(
                id="turn",
                type=StepType.LLM_CALL,
                prompt="newq",
                conversation="state.chat",
            ),
            state={
                "chat": {
                    "messages": [],
                    "summary": "earlier: hello world",
                }
            },
            gateway=gw,
        )
        await LLMCallStep().execute(ctx)
        # The provider saw a synthetic system turn built from the summary.
        wire = provider.calls[0]["messages"]
        assert wire[0]["role"] == "system"
        assert "earlier: hello world" in wire[0]["content"]

    async def test_malformed_conversation_raises_step_error(self) -> None:
        from agentloom.exceptions import StepError

        provider = MockProvider()
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        ctx = _make_context(
            StepDefinition(
                id="turn",
                type=StepType.LLM_CALL,
                prompt="hi",
                conversation="state.chat",
            ),
            state={"chat": 42},  # not a dict/list/Conversation
            gateway=gw,
        )
        # Early guard: workflow-author error surfaces as StepError (same
        # class as ``no gateway`` / ``no prompt``), not a captured
        # StepStatus.FAILED. Consistent with attachment / template early
        # guards elsewhere in the executor.
        with pytest.raises(StepError, match="malformed"):
            await LLMCallStep().execute(ctx)

    async def test_no_prompt_no_conversation_still_raises(self) -> None:
        gw = ProviderGateway()
        gw.register(MockProvider(), models=["mock-model"])
        ctx = _make_context(StepDefinition(id="s", type=StepType.LLM_CALL), gateway=gw)
        with pytest.raises(Exception, match="requires a 'prompt'"):
            await LLMCallStep().execute(ctx)


# ---- Provider name propagation ---------------------------------------------


class TestProviderNameField:
    def test_openai_forwards_name(self) -> None:
        from agentloom.providers.openai import OpenAIProvider

        formatted = OpenAIProvider._format_messages(
            [{"role": "user", "content": "hi", "name": "alice"}]
        )
        assert formatted[0]["name"] == "alice"

    def test_openai_drops_name_when_absent(self) -> None:
        from agentloom.providers.openai import OpenAIProvider

        formatted = OpenAIProvider._format_messages([{"role": "user", "content": "hi"}])
        assert "name" not in formatted[0]

    def test_anthropic_prepends_name_to_content(self) -> None:
        from agentloom.providers.anthropic import AnthropicProvider

        _, formatted = AnthropicProvider._format_messages(
            [{"role": "user", "content": "hi", "name": "bob"}]
        )
        assert formatted[0]["content"].startswith("[bob]")

    def test_google_prepends_name_to_text_part(self) -> None:
        from agentloom.providers.google import GoogleProvider

        _, formatted = GoogleProvider._format_messages(
            [{"role": "user", "content": "hi", "name": "carla"}]
        )
        assert "[carla]" in formatted[0]["parts"][0]["text"]

    def test_ollama_prepends_name_to_content(self) -> None:
        from agentloom.providers.ollama import OllamaProvider

        formatted = OllamaProvider._format_messages(
            [{"role": "user", "content": "hi", "name": "dan"}]
        )
        assert formatted[0]["content"].startswith("[dan]")


# ---- Observer wiring -------------------------------------------------------


class RecordingObserver(WorkflowObserver):
    def __init__(self) -> None:
        super().__init__(tracing=None, metrics=None)
        self.conversation_events: list[dict[str, Any]] = []
        self.attached_events: list[tuple[str, str, dict[str, Any]]] = []

    def on_conversation_turn(self, **kwargs: Any) -> None:  # type: ignore[override]
        self.conversation_events.append(kwargs)

    def attach_step_event(self, step_id: str, event_name: str, attributes: dict[str, Any]) -> None:
        self.attached_events.append((step_id, event_name, attributes))


class TestObserverIntegration:
    async def test_conversation_turn_hook_fires(self) -> None:
        provider = MockProvider(responses={"hi": "hello"})
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        obs = RecordingObserver()
        ctx = _make_context(
            StepDefinition(
                id="turn",
                type=StepType.LLM_CALL,
                prompt="hi",
                conversation="state.chat",
            ),
            state={"chat": {"messages": []}},
            gateway=gw,
            observer=obs,
        )
        await LLMCallStep().execute(ctx)
        assert obs.conversation_events
        evt = obs.conversation_events[0]
        assert evt["conversation_key"] == "chat"
        assert evt["turn_count"] == 2

    async def test_conversation_span_event_attached(self) -> None:
        provider = MockProvider(responses={"hi": "hello"})
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        obs = RecordingObserver()
        ctx = _make_context(
            StepDefinition(
                id="turn",
                type=StepType.LLM_CALL,
                prompt="hi",
                conversation="state.chat",
            ),
            state={"chat": {"messages": []}},
            gateway=gw,
            observer=obs,
        )
        await LLMCallStep().execute(ctx)
        conv_events = [e for e in obs.attached_events if e[1] == SpanAttr.CONVERSATION_EVENT]
        assert conv_events
        _, _, attrs = conv_events[0]
        assert attrs[SpanAttr.CONVERSATION_TURN_COUNT] == 2

    def test_metrics_records_trim_and_message_count(self) -> None:
        # Fire the recorder directly against a MetricsManager to exercise
        # the counter/histogram wiring without spinning a full workflow.
        # ``enabled=True`` still degrades to noop when no backend is
        # installed — we only assert the method does not raise.
        mm = MetricsManager(enabled=True)
        mm.record_conversation_turn("chat", 4, trim_policy="drop_oldest", trimmed_count=2)
        mm.record_conversation_turn("chat", 6)  # no-trim path
        mm.record_conversation_turn(
            "chat", 3, trim_policy="drop_oldest", trimmed_count=0
        )  # trims counter must NOT fire

    def test_noop_observer_accepts_hook(self) -> None:
        from agentloom.observability.noop import NoopObserver

        NoopObserver().on_conversation_turn(
            step_id="s", conversation_key="chat", turn_count=1, token_count=1
        )


# ---- YAML parsing -----------------------------------------------------------


class TestConversationYAMLParsing:
    def test_conversation_field_accepted_in_step(self) -> None:
        import yaml as _yaml

        raw = _yaml.safe_load(
            """
name: conv-test
version: "1.0"
config:
  provider: mock
  model: mock-model
state:
  chat:
    messages:
      - role: system
        content: "Be nice."
steps:
  - id: turn
    type: llm_call
    conversation: state.chat
    prompt: "hi"
    output: reply
"""
        )
        wf = WorkflowParser.from_dict(raw)
        assert wf.steps[0].conversation == "state.chat"

    def test_state_conversation_shape_survives_workflow_definition(self) -> None:
        wf = WorkflowDefinition(
            name="t",
            config=WorkflowConfig(provider="mock", model="mock-model"),
            state={"chat": Conversation().model_dump()},
            steps=[
                StepDefinition(
                    id="turn",
                    type=StepType.LLM_CALL,
                    prompt="hi",
                    conversation="state.chat",
                )
            ],
        )
        assert wf.state["chat"]["messages"] == []


# ---- summarize_oldest via the gateway (HIGH-1 fix) --------------------------


class _CountingProvider(MockProvider):
    """MockProvider that records every step_id it was called with."""

    # The gateway strips ``step_id`` before calling a provider unless it
    # opts in — mirror the real MockProvider so the summariser call's
    # ``{step}::summarize`` id survives to ``complete``.
    accepts_step_id = True

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        super().__init__(responses=responses)
        self.summarize_called = 0

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        if kwargs.get("step_id", "").endswith("::summarize"):
            self.summarize_called += 1
            return ProviderResponse(
                content="LLM SUMMARY of earlier turns.",
                model=model,
                provider="mock",
                usage=TokenUsage(prompt_tokens=5, completion_tokens=6, total_tokens=11),
                cost_usd=0.0,
            )
        return await super().complete(messages, model, temperature, max_tokens, **kwargs)


class TestSummarizeOldestUsesGateway:
    async def test_summarize_policy_invokes_llm(self) -> None:
        provider = _CountingProvider(responses={"final": "done"})
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        # Long history that overflows the tiny budget → summarise fires.
        history = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "q1 " * 40},
            {"role": "assistant", "content": "a1 " * 40},
            {"role": "user", "content": "q2 " * 40},
            {"role": "assistant", "content": "a2 " * 40},
        ]
        ctx = _make_context(
            StepDefinition(
                id="turn",
                type=StepType.LLM_CALL,
                prompt="final",
                conversation="state.chat",
            ),
            state={
                "chat": {
                    "messages": history,
                    "token_budget": 30,
                    "trim_policy": "summarize_oldest",
                }
            },
            gateway=gw,
        )
        await LLMCallStep().execute(ctx)
        assert provider.summarize_called >= 1
        chat = Conversation.model_validate(await ctx.state_manager.get("chat"))
        assert chat.summary is not None
        assert "LLM SUMMARY" in chat.summary

    async def test_summarize_falls_back_when_gateway_summary_raises(self) -> None:
        class RaisingProvider(MockProvider):
            accepts_step_id = True

            async def complete(
                self,
                messages: list[dict[str, Any]],
                model: str,
                temperature: float | None = None,
                max_tokens: int | None = None,
                **kwargs: Any,
            ) -> ProviderResponse:
                if kwargs.get("step_id", "").endswith("::summarize"):
                    raise RuntimeError("summary provider down")
                return await super().complete(messages, model, temperature, max_tokens, **kwargs)

        provider = RaisingProvider(responses={"final": "done"})
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        ctx = _make_context(
            StepDefinition(
                id="turn",
                type=StepType.LLM_CALL,
                prompt="final",
                conversation="state.chat",
            ),
            state={
                "chat": {
                    "messages": [
                        {"role": "user", "content": "q1 " * 40},
                        {"role": "assistant", "content": "a1 " * 40},
                        {"role": "user", "content": "q2 " * 40},
                    ],
                    "token_budget": 20,
                    "trim_policy": "summarize_oldest",
                }
            },
            gateway=gw,
        )
        result = await LLMCallStep().execute(ctx)
        # The step still succeeds — trim degraded to concat fallback.
        assert result.status == StepStatus.SUCCESS
        chat = Conversation.model_validate(await ctx.state_manager.get("chat"))
        assert chat.summary  # deterministic concat fallback populated it


class TestApplyTrimPolicyAsync:
    async def test_async_awaits_summarizer(self) -> None:
        convo = Conversation(
            messages=[
                Message(role="user", content="q1 " * 30),
                Message(role="assistant", content="a1 " * 30),
                Message(role="user", content="q2"),
            ],
            token_budget=10,
            trim_policy="summarize_oldest",
        )

        async def summarizer(folded: list[Message]) -> str:
            return "ASYNC SUMMARY"

        result = await apply_trim_policy_async(convo, "mock-model", summarizer=summarizer)
        assert result.summarized_count > 0
        assert "ASYNC SUMMARY" in (convo.summary or "")

    async def test_async_delegates_non_summarize_policies(self) -> None:
        convo = Conversation(
            messages=[
                Message(role="system", content="sys"),
                Message(role="user", content="q1"),
                Message(role="assistant", content="a1"),
                Message(role="user", content="q2"),
            ],
            token_budget=12,
            trim_policy="drop_oldest",
        )
        result = await apply_trim_policy_async(convo, "mock-model")
        assert result.policy == "drop_oldest"
        assert convo.messages[0].role == "system"

    async def test_async_noop_when_within_budget(self) -> None:
        convo = Conversation(
            messages=[Message(role="user", content="hi")],
            token_budget=100,
        )
        result = await apply_trim_policy_async(convo, "mock-model")
        assert result.trimmed_count == 0

    def test_sync_apply_ignores_async_summarizer(self) -> None:
        # The sync entry-point must NOT await; passing an async summariser
        # degrades to the concat fallback with a warning rather than
        # returning a coroutine as the summary text.
        convo = Conversation(
            messages=[
                Message(role="user", content="q1 " * 30),
                Message(role="assistant", content="a1 " * 30),
                Message(role="user", content="q2"),
            ],
            token_budget=10,
            trim_policy="summarize_oldest",
        )

        async def summarizer(folded: list[Message]) -> str:  # pragma: no cover - never awaited
            return "SHOULD NOT APPEAR"

        apply_trim_policy(convo, "mock-model", summarizer=summarizer)
        assert "SHOULD NOT APPEAR" not in (convo.summary or "")
        assert convo.summary  # concat fallback still populated it


# ---- Multi-agent speaker (MEDIUM-3 fix) ------------------------------------


class TestSpeakerTagging:
    async def test_speaker_tags_user_and_assistant_turns(self) -> None:
        provider = MockProvider(responses={"go": "reply"})
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        ctx = _make_context(
            StepDefinition(
                id="turn",
                type=StepType.LLM_CALL,
                prompt="go",
                conversation="state.chat",
                speaker="alice",
            ),
            state={"chat": {"messages": []}},
            gateway=gw,
        )
        await LLMCallStep().execute(ctx)
        chat = Conversation.model_validate(await ctx.state_manager.get("chat"))
        assert chat.messages[0].name == "alice"  # user turn
        assert chat.messages[1].name == "alice"  # assistant reply

    async def test_speaker_system_prompt_not_persisted(self) -> None:
        # With a speaker, the system_prompt is a transient per-turn message,
        # never stored in the shared conversation.
        provider = MockProvider(responses={"go": "reply"})
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        ctx = _make_context(
            StepDefinition(
                id="turn",
                type=StepType.LLM_CALL,
                prompt="go",
                system_prompt="You are Alice.",
                conversation="state.chat",
                speaker="alice",
            ),
            state={"chat": {"messages": []}},
            gateway=gw,
        )
        await LLMCallStep().execute(ctx)
        chat = Conversation.model_validate(await ctx.state_manager.get("chat"))
        assert not any(m.role == "system" for m in chat.messages)
        # But the provider DID see the system message this turn.
        wire = provider.calls[0]["messages"]
        assert wire[0]["role"] == "system"
        assert "Alice" in wire[0]["content"]


# ---- Attachments through a conversation (HIGH-2 fix) -----------------------


class TestConversationAttachments:
    async def test_attachment_reaches_provider_with_conversation(self, tmp_path: Path) -> None:
        from agentloom.providers.multimodal import ImageBlock, extract_text_content

        # 1x1 transparent PNG.
        png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00"
            b"\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        import base64

        img = tmp_path / "pixel.png"
        img.write_bytes(png)

        captured: dict[str, Any] = {}

        class VisionProvider(MockProvider):
            async def complete(
                self,
                messages: list[dict[str, Any]],
                model: str,
                temperature: float | None = None,
                max_tokens: int | None = None,
                **kwargs: Any,
            ) -> ProviderResponse:
                captured["messages"] = messages
                return await super().complete(messages, model, temperature, max_tokens, **kwargs)

        provider = VisionProvider(responses={})
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        ctx = _make_context(
            StepDefinition(
                id="turn",
                type=StepType.LLM_CALL,
                prompt="What's in this image?",
                conversation="state.chat",
                attachments=[Attachment(type="image", source=str(img), fetch="local")],
            ),
            state={"chat": {"messages": []}},
            gateway=gw,
        )
        _ = base64  # keep import referenced for clarity
        result = await LLMCallStep().execute(ctx)
        assert result.status == StepStatus.SUCCESS
        # The provider's last message content must be a block list carrying
        # an ImageBlock — not the bare text prompt.
        last = captured["messages"][-1]["content"]
        assert isinstance(last, list)
        assert any(isinstance(b, ImageBlock) for b in last)
        # And the text part still carries the rendered prompt.
        assert "image" in extract_text_content(last).lower()

    async def test_attachments_not_persisted_in_conversation(self, tmp_path: Path) -> None:
        png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00"
            b"\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        img = tmp_path / "pixel.png"
        img.write_bytes(png)

        provider = MockProvider(responses={})
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        ctx = _make_context(
            StepDefinition(
                id="turn",
                type=StepType.LLM_CALL,
                prompt="describe",
                conversation="state.chat",
                attachments=[Attachment(type="image", source=str(img), fetch="local")],
            ),
            state={"chat": {"messages": []}},
            gateway=gw,
        )
        await LLMCallStep().execute(ctx)
        chat = Conversation.model_validate(await ctx.state_manager.get("chat"))
        # The persisted user turn stores plain text — no base64 blob bloats
        # the checkpoint.
        user_turn = chat.messages[0]
        assert user_turn.content == "describe"
        assert isinstance(user_turn.content, str)


# ---- Streaming + conversation (MEDIUM-4 fix) -------------------------------


class TestStreamingConversation:
    async def test_stream_persists_conversation(self) -> None:
        provider = MockProvider(responses={"stream me": "streamed reply"})
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        ctx = StepContext(
            step_definition=StepDefinition(
                id="turn",
                type=StepType.LLM_CALL,
                prompt="stream me",
                conversation="state.chat",
            ),
            state_manager=StateManager(initial_state={"chat": {"messages": []}}),
            provider_gateway=gw,
            workflow_model="mock-model",
            stream=True,
        )
        result = await LLMCallStep().execute(ctx)
        assert result.status == StepStatus.SUCCESS
        chat = Conversation.model_validate(await ctx.state_manager.get("chat"))
        assert [m.role for m in chat.messages] == ["user", "assistant"]
        assert chat.messages[1].content == "streamed reply"

    async def test_stream_with_speaker_tags_turns(self) -> None:
        provider = MockProvider(responses={"q": "a"})
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        ctx = StepContext(
            step_definition=StepDefinition(
                id="turn",
                type=StepType.LLM_CALL,
                prompt="q",
                conversation="state.chat",
                speaker="bob",
            ),
            state_manager=StateManager(initial_state={"chat": {"messages": []}}),
            provider_gateway=gw,
            workflow_model="mock-model",
            stream=True,
        )
        await LLMCallStep().execute(ctx)
        chat = Conversation.model_validate(await ctx.state_manager.get("chat"))
        assert chat.messages[0].name == "bob"
        assert chat.messages[1].name == "bob"


# ---- Copilot review fixes -------------------------------------------------


class TestSummaryPositioning:
    def test_summary_lands_after_system_prefix(self) -> None:
        # Copilot-2: a model-generated summary must not sit ahead of the
        # workflow's real system prompt (it could override the instruction).
        c = Conversation(
            messages=[
                Message(role="system", content="You are strict."),
                Message(role="user", content="q1"),
                Message(role="assistant", content="a1"),
            ],
            summary="earlier context",
        )
        wire = to_provider_messages(c)
        assert wire[0]["role"] == "system"
        assert wire[0]["content"] == "You are strict."  # real system prompt first
        assert wire[1]["role"] == "system"
        assert "earlier context" in wire[1]["content"]  # summary trails it
        assert wire[2]["role"] == "user"

    def test_summary_appended_when_all_system(self) -> None:
        # All-system conversation with a summary → summary appended at the end.
        c = Conversation(
            messages=[Message(role="system", content="sys only")],
            summary="ctx",
        )
        wire = to_provider_messages(c)
        assert wire[0]["content"] == "sys only"
        assert wire[-1]["role"] == "system"
        assert "ctx" in wire[-1]["content"]


class TestPromptOmittedGuard:
    async def test_raises_when_conversation_ends_on_assistant(self) -> None:
        from agentloom.exceptions import StepError

        gw = ProviderGateway()
        gw.register(MockProvider(), models=["mock-model"])
        ctx = _make_context(
            StepDefinition(id="turn", type=StepType.LLM_CALL, conversation="state.chat"),
            state={
                "chat": {
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "hello"},
                    ]
                }
            },
            gateway=gw,
        )
        with pytest.raises(StepError, match="does not end on a 'user' or 'tool' turn"):
            await LLMCallStep().execute(ctx)

    async def test_raises_when_conversation_empty_and_no_prompt(self) -> None:
        from agentloom.exceptions import StepError

        gw = ProviderGateway()
        gw.register(MockProvider(), models=["mock-model"])
        ctx = _make_context(
            StepDefinition(id="turn", type=StepType.LLM_CALL, conversation="state.chat"),
            state={"chat": {"messages": []}},
            gateway=gw,
        )
        with pytest.raises(StepError, match="last non-system role: none"):
            await LLMCallStep().execute(ctx)

    async def test_raises_on_unreadable_conversation_path(self) -> None:
        from agentloom.exceptions import StepError

        gw = ProviderGateway()
        gw.register(MockProvider(), models=["mock-model"])
        # An empty path segment (``state..chat``) makes the state resolver
        # raise ValueError, surfaced as a clear StepError rather than a crash.
        ctx = _make_context(
            StepDefinition(
                id="turn", type=StepType.LLM_CALL, prompt="hi", conversation="state..chat"
            ),
            state={"chat": {"messages": []}},
            gateway=gw,
        )
        with pytest.raises(StepError, match="unreadable"):
            await LLMCallStep().execute(ctx)

    async def test_ok_when_conversation_ends_on_tool_turn(self) -> None:
        provider = MockProvider(responses={})
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])
        ctx = _make_context(
            StepDefinition(id="turn", type=StepType.LLM_CALL, conversation="state.chat"),
            state={
                "chat": {
                    "messages": [
                        {"role": "user", "content": "what time?"},
                        {"role": "tool", "content": "12:00", "tool_call_id": "c1"},
                    ]
                }
            },
            gateway=gw,
        )
        result = await LLMCallStep().execute(ctx)
        assert result.status == StepStatus.SUCCESS


class TestSingleAgentSystemInsert:
    async def test_system_prompt_inserted_once_and_persisted(self) -> None:
        provider = MockProvider(responses={"go": "ok"})
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])
        ctx = _make_context(
            StepDefinition(
                id="turn",
                type=StepType.LLM_CALL,
                prompt="go",
                system_prompt="You are helpful.",
                conversation="state.chat",
            ),
            state={"chat": {"messages": []}},
            gateway=gw,
        )
        await LLMCallStep().execute(ctx)
        chat = Conversation.model_validate(await ctx.state_manager.get("chat"))
        # Single-agent: the system prompt IS persisted, exactly once, at head.
        assert chat.messages[0].role == "system"
        assert chat.messages[0].content == "You are helpful."
        assert sum(1 for m in chat.messages if m.role == "system") == 1

    async def test_attachments_ignored_without_user_turn(self, tmp_path: Path) -> None:
        # prompt omitted + attachments + a conversation ending on a user turn:
        # the attachment hoist must NOT overwrite the trailing user message.
        png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00"
            b"\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        img = tmp_path / "pixel.png"
        img.write_bytes(png)

        captured: dict[str, Any] = {}

        class SpyProvider(MockProvider):
            async def complete(
                self,
                messages: list[dict[str, Any]],
                model: str,
                temperature: float | None = None,
                max_tokens: int | None = None,
                **kwargs: Any,
            ) -> ProviderResponse:
                captured["messages"] = messages
                return await super().complete(messages, model, temperature, max_tokens, **kwargs)

        provider = SpyProvider(responses={})
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])
        ctx = _make_context(
            StepDefinition(
                id="turn",
                type=StepType.LLM_CALL,
                conversation="state.chat",
                attachments=[Attachment(type="image", source=str(img), fetch="local")],
            ),
            state={"chat": {"messages": [{"role": "user", "content": "keep this"}]}},
            gateway=gw,
        )
        result = await LLMCallStep().execute(ctx)
        assert result.status == StepStatus.SUCCESS
        # The trailing user text is preserved (not overwritten by the hoist).
        assert captured["messages"][-1]["content"] == "keep this"


class TestObserverConversationSpan:
    class _FakeSpan:
        def __init__(self) -> None:
            self.attrs: dict[str, Any] = {}

        def set_attribute(self, key: str, value: Any) -> None:
            self.attrs[key] = value

        def end(self) -> None:
            pass

    class _FakeTracing:
        def __init__(self, span: Any) -> None:
            self._span = span

        def start_span(self, name: str, attributes: dict[str, Any] | None = None) -> Any:
            return self._span

        def end_span(self, span: Any) -> None:
            pass

    def test_on_conversation_turn_stamps_span_and_metrics(self) -> None:
        span = self._FakeSpan()
        metrics = MetricsManager(enabled=True)
        obs = WorkflowObserver(tracing=self._FakeTracing(span), metrics=metrics)
        obs.on_step_start("turn", "llm_call")
        obs.on_conversation_turn(
            step_id="turn",
            conversation_key="chat",
            turn_count=4,
            token_count=120,
            trim_policy="drop_pairs",
            trimmed_count=2,
        )
        assert span.attrs[SpanAttr.CONVERSATION_TURN_COUNT] == 4
        assert span.attrs[SpanAttr.CONVERSATION_TOKEN_COUNT] == 120
        assert span.attrs[SpanAttr.CONVERSATION_TRIMMED_MESSAGES] == 2

    def test_on_conversation_turn_no_span_is_safe(self) -> None:
        # No step span registered → metrics still fire, no crash.
        obs = WorkflowObserver(tracing=None, metrics=MetricsManager(enabled=True))
        obs.on_conversation_turn(
            step_id="missing", conversation_key="chat", turn_count=1, token_count=1
        )


class TestJsonDefault:
    def test_pydantic_model_dumped(self) -> None:
        out = _json_default(Message(role="user", content="hi"))
        assert out == {
            "role": "user",
            "content": "hi",
            "name": None,
            "tool_call_id": None,
            "tool_calls": [],
            "metadata": {},
        }

    def test_non_model_falls_back_to_str(self) -> None:
        assert _json_default({1, 2}) in ("{1, 2}", "{2, 1}")


class TestTokenCountOverhead:
    def test_name_adds_overhead(self) -> None:
        plain = Message(role="user", content="hi").token_count()
        named = Message(role="user", content="hi", name="alice").token_count()
        assert named > plain

    def test_tool_call_id_adds_overhead(self) -> None:
        plain = Message(role="tool", content="ok").token_count()
        keyed = Message(role="tool", content="ok", tool_call_id="call_123").token_count()
        assert keyed > plain


class TestConversationCoerceEdges:
    def test_non_list_messages_passes_through_then_fails(self) -> None:
        with pytest.raises(ValueError):
            Conversation.model_validate({"messages": "not-a-list"})

    def test_non_dict_entry_passes_through_then_fails(self) -> None:
        with pytest.raises(ValueError):
            Conversation.model_validate({"messages": [123]})


class TestAsyncSummarizeOverPinned:
    async def test_async_folded_empty_over_pinned_prefix(self) -> None:
        # Async path: system prefix already over budget → nothing to fold.
        convo = Conversation(
            messages=[Message(role="system", content="verbose " * 100)],
            token_budget=5,
            trim_policy="summarize_oldest",
        )
        result = await apply_trim_policy_async(convo, "mock-model")
        assert result.summarized_count == 0
        assert convo.summary is None


class TestInternalTrimGuards:
    # The per-policy helpers carry a defensive ``token_budget is None`` guard
    # that ``apply_trim_policy`` normally short-circuits before reaching them.
    # Call them directly so the guard stays covered if a future caller skips
    # the top-level check.
    def test_drop_oldest_budget_none_guard(self) -> None:
        convo = Conversation(messages=[Message(role="user", content="hi")])
        assert _drop_oldest(convo, "m").policy == "drop_oldest"

    def test_drop_pairs_budget_none_guard(self) -> None:
        convo = Conversation(messages=[Message(role="user", content="hi")])
        assert _drop_pairs(convo, "m").policy == "drop_pairs"

    def test_summarize_budget_none_guard(self) -> None:
        convo = Conversation(messages=[Message(role="user", content="hi")])
        assert _summarize_oldest(convo, "m", None).policy == "summarize_oldest"

    def test_apply_summary_empty_folded_noop(self) -> None:
        convo = Conversation(messages=[Message(role="user", content="hi")])
        result = _apply_summary(convo, [], "unused")
        assert result.trimmed_count == 0

    def test_apply_trim_policy_unknown_policy_raises(self) -> None:
        # Bypass Pydantic validation to simulate an unwired future policy.
        convo = Conversation.model_construct(
            messages=[Message(role="user", content="q " * 40)],
            token_budget=5,
            trim_policy="teleport",
            summary=None,
            metadata={},
        )
        with pytest.raises(ValueError, match="Unknown trim_policy"):
            apply_trim_policy(convo, "mock-model")


class TestProviderNameMultimodal:
    def test_anthropic_prepends_name_on_multimodal(self) -> None:
        from agentloom.providers.anthropic import AnthropicProvider
        from agentloom.providers.multimodal import TextBlock

        _, formatted = AnthropicProvider._format_messages(
            [{"role": "user", "content": [TextBlock(text="hi")], "name": "bob"}]
        )
        assert formatted[0]["content"][0] == {"type": "text", "text": "[bob]"}

    def test_google_prepends_name_on_multimodal(self) -> None:
        from agentloom.providers.google import GoogleProvider
        from agentloom.providers.multimodal import TextBlock

        _, formatted = GoogleProvider._format_messages(
            [{"role": "user", "content": [TextBlock(text="hi")], "name": "carla"}]
        )
        assert formatted[0]["parts"][0] == {"text": "[carla]"}

    def test_ollama_prepends_name_on_multimodal(self) -> None:
        from agentloom.providers.multimodal import TextBlock
        from agentloom.providers.ollama import OllamaProvider

        formatted = OllamaProvider._format_messages(
            [{"role": "user", "content": [TextBlock(text="hi")], "name": "dan"}]
        )
        assert formatted[0]["content"].startswith("[dan]")
