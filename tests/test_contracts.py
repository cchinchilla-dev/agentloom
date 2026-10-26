"""Public contracts (#159): light import, stable re-exports, WorkflowHandle and its adapter."""

from __future__ import annotations

import importlib
import subprocess
import sys
import warnings
from typing import Any

import pytest

from agentloom.contracts import stable
from agentloom.contracts.stable import Message, ToolCall, WorkflowHandle
from agentloom.core.handle import WorkflowAdapter
from agentloom.core.models import (
    RetryConfig,
    StepDefinition,
    StepType,
    WorkflowConfig,
    WorkflowDefinition,
)
from agentloom.exceptions import WorkflowError
from agentloom.providers.gateway import ProviderGateway
from tests.conftest import MockProvider

RUNTIME = ("agentloom.core", "anyio", "httpx", "typer", "yaml", "jsonschema")


def _python(code: str, *flags: str) -> subprocess.CompletedProcess[str]:
    """A fresh interpreter, so modules imported by other tests do not count."""
    return subprocess.run(
        [sys.executable, *flags, "-c", code], capture_output=True, text=True, check=False
    )


def test_contracts_import_does_not_load_runtime() -> None:
    probe = _python(
        "import sys, agentloom.contracts, agentloom.contracts.stable\n"
        f"print([m for m in {RUNTIME!r} if m in sys.modules])"
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "[]"


def test_contracts_import_does_not_warn() -> None:
    probe = _python(
        "import agentloom.contracts, agentloom.contracts.stable, agentloom.observability.observer,"
        " agentloom.cli.main",
        "-W",
        "error::FutureWarning",
    )
    assert probe.returncode == 0, probe.stderr


def test_top_level_names_resolve_on_first_access() -> None:
    import agentloom
    from agentloom.core import graph

    assert agentloom.WorkflowGraph is graph.WorkflowGraph
    assert set(agentloom.__all__) <= set(dir(agentloom))
    with pytest.raises(AttributeError, match="no attribute 'missing'"):
        _ = agentloom.missing


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("agentloom.core.models", "Attachment"),
        ("agentloom.core.models", "Message"),
        ("agentloom.core.models", "ToolDefinition"),
        ("agentloom.core.results", "TokenUsage"),
        ("agentloom.providers.base", "ToolCall"),
        ("agentloom.providers.base", "ProviderResponse"),
        ("agentloom.providers.base", "ResponseMeta"),
    ],
)
def test_contracts_reexports_are_identical(module: str, name: str) -> None:
    assert getattr(importlib.import_module(module), name) is getattr(stable, name)


def test_semconv_reexports_are_identical() -> None:
    from agentloom.contracts import _semconv
    from agentloom.observability import schema

    for name in ("SpanName", "SpanAttr", "MetricName", "GenAIOperationName", "GenAIProviderName"):
        assert getattr(schema, name) is getattr(_semconv, name)


def test_inter_agent_reexports_are_identical() -> None:
    from agentloom.contracts import _inter_agent

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        experimental = importlib.import_module("agentloom.contracts.experimental")
    names = [n for n in experimental.__all__ if hasattr(_inter_agent, n)]
    assert len(names) == 9
    for name in names:
        assert getattr(experimental, name) is getattr(_inter_agent, name)


def test_message_tool_calls_json_roundtrip() -> None:
    call = ToolCall(id="c1", name="lookup", arguments={"id": 7, "tags": ["a"]})
    message = Message(role="assistant", tool_calls=[call])
    assert Message.model_validate_json(message.model_dump_json()) == message
    assert message.model_dump()["tool_calls"] == [
        {"id": "c1", "name": "lookup", "arguments": {"id": 7, "tags": ["a"]}}
    ]


def test_experimental_import_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "agentloom.contracts.experimental", raising=False)
    with pytest.warns(FutureWarning, match="semantic conventions"):
        importlib.import_module("agentloom.contracts.experimental")


def test_structural_class_satisfies_workflow_handle() -> None:
    class Echo:
        input_schema: dict[str, Any] = {"type": "object"}
        output_schema: dict[str, Any] = {"type": "object"}

        async def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
            return inputs

    assert isinstance(Echo(), WorkflowHandle)
    assert not isinstance(object(), WorkflowHandle)


def _workflow(*steps: StepDefinition, budget_usd: float | None = None) -> WorkflowDefinition:
    return WorkflowDefinition(
        name="handle-test",
        config=WorkflowConfig(provider="mock", model="mock-model", budget_usd=budget_usd),
        state={"question": "default"},
        steps=list(steps),
    )


def _ask(step_id: str, **extra: Any) -> StepDefinition:
    return StepDefinition(
        id=step_id, type=StepType.LLM_CALL, prompt="{state.question}", output=step_id, **extra
    )


def _gateway(provider: MockProvider | None = None) -> ProviderGateway:
    gateway = ProviderGateway()
    if provider is not None:
        gateway.register(provider, priority=0)
    return gateway


async def test_workflow_adapter_runs_as_handle() -> None:
    handle = WorkflowAdapter(_workflow(_ask("answer")), _gateway(MockProvider({"hi": "hello"})))
    assert isinstance(handle, WorkflowHandle)
    assert handle.input_schema["properties"] == {"question": {}}  # keys, never default values
    assert handle.output_schema == {"type": "object", "additionalProperties": True}
    outputs = await handle.run({"question": "hi"})
    assert outputs["answer"] == "hello"
    assert outputs["question"] == "hi"


async def test_workflow_adapter_runs_leave_the_workflow_state_untouched() -> None:
    step = StepDefinition(
        id="a", type=StepType.LLM_CALL, prompt="{state.question}", output="cfg.answer"
    )
    workflow = _workflow(step)
    workflow.state["cfg"] = {}
    handle = WorkflowAdapter(workflow, _gateway(MockProvider({"hi": "hello"})))
    for _ in range(2):
        assert (await handle.run({"question": "hi"}))["cfg"] == {"answer": "hello"}
    assert workflow.state == {"question": "default", "cfg": {}}


@pytest.mark.parametrize(
    ("steps", "budget_usd", "gateway"),
    [
        pytest.param([_ask("a", retry=RetryConfig(max_retries=0))], None, _gateway(), id="failed"),
        pytest.param(
            [_ask("a"), StepDefinition(id="gate", type=StepType.APPROVAL_GATE, depends_on=["a"])],
            None,
            _gateway(MockProvider()),
            id="paused",
        ),
        pytest.param(
            [_ask("a"), _ask("b", depends_on=["a"])],
            0.0001,
            _gateway(MockProvider()),
            id="over-budget",
        ),
    ],
)
async def test_workflow_adapter_raises_on_failure(
    steps: list[StepDefinition], budget_usd: float | None, gateway: ProviderGateway
) -> None:
    handle = WorkflowAdapter(_workflow(*steps, budget_usd=budget_usd), gateway)
    with pytest.raises(WorkflowError, match="handle-test"):
        await handle.run({"question": "hi"})
