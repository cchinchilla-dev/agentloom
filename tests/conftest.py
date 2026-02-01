"""Shared test fixtures for AgentLoom."""

from __future__ import annotations

from typing import Any

import pytest

from agentloom.core.models import (
    Condition,
    StepDefinition,
    StepType,
    WorkflowConfig,
    WorkflowDefinition,
)
from agentloom.core.results import TokenUsage
from agentloom.providers.base import BaseProvider, ProviderResponse
from agentloom.providers.gateway import ProviderGateway


class MockProvider(BaseProvider):
    """Mock LLM provider for testing."""

    name = "mock"

    def __init__(self, responses: dict[str, str] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.responses = responses or {}
        self.calls: list[dict[str, Any]] = []
        self._default_response = "Mock response"

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        self.calls.append({
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })

        last_msg = messages[-1]["content"] if messages else ""
        content = self.responses.get(last_msg, self._default_response)

        return ProviderResponse(
            content=content,
            model=model,
            provider="mock",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            cost_usd=0.001,
        )

    def supports_model(self, model: str) -> bool:
        return True


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider()




@pytest.fixture
def mock_gateway(mock_provider: MockProvider) -> ProviderGateway:
    gateway = ProviderGateway()
    gateway.register(mock_provider, priority=0)
    return gateway




@pytest.fixture
def simple_workflow() -> WorkflowDefinition:
    return WorkflowDefinition(
        name="test-workflow",
        config=WorkflowConfig(provider="mock", model="mock-model"),
        state={"question": "test question"},
        steps=[
            StepDefinition(
                id="answer",
                type=StepType.LLM_CALL,
                prompt="Answer: {state.question}",
                output="answer",
            ),
        ],
    )


@pytest.fixture
def parallel_workflow() -> WorkflowDefinition:
    return WorkflowDefinition(
        name="parallel-test",
        config=WorkflowConfig(provider="mock", model="mock-model"),
        state={"topic": "AI"},
        steps=[
            StepDefinition(
                id="step_a",
                type=StepType.LLM_CALL,
                prompt="Research A about {state.topic}",
                output="result_a",
            ),
            StepDefinition(
                id="step_b",
                type=StepType.LLM_CALL,
                prompt="Research B about {state.topic}",
                output="result_b",
            ),
            StepDefinition(
                id="merge",
                type=StepType.LLM_CALL,
                depends_on=["step_a", "step_b"],
                prompt="Merge: {state.result_a} and {state.result_b}",
                output="merged",
            ),
        ],
    )


@pytest.fixture
def router_workflow() -> WorkflowDefinition:
    return WorkflowDefinition(
        name="router-test",
        config=WorkflowConfig(provider="mock", model="mock-model"),
        state={"input": "test", "classification": ""},
        steps=[
            StepDefinition(
                id="classify",
                type=StepType.LLM_CALL,
                prompt="Classify: {state.input}",
                output="classification",
            ),
            StepDefinition(
                id="route",
                type=StepType.ROUTER,
                depends_on=["classify"],
                conditions=[
                    Condition(expression="state.classification == 'billing'", target="billing"),
                    Condition(expression="state.classification == 'technical'", target="technical"),
                ],
                default="general",
            ),
            StepDefinition(
                id="billing",
                type=StepType.LLM_CALL,
                depends_on=["route"],
                prompt="Handle billing: {state.input}",
                output="response",
            ),
            StepDefinition(
                id="technical",
                type=StepType.LLM_CALL,
                depends_on=["route"],
                prompt="Handle technical: {state.input}",
                output="response",
            ),
            StepDefinition(
                id="general",
                type=StepType.LLM_CALL,
                depends_on=["route"],
                prompt="Handle general: {state.input}",
                output="response",
            ),
        ],
    )


SIMPLE_YAML = """
name: yaml-test
version: "1.0"
config:
  provider: mock
  model: mock-model
state:
  question: "What is Python?"
steps:
  - id: answer
    type: llm_call
    prompt: "Answer: {state.question}"
    output: answer
"""

INVALID_YAML_CYCLE = """
name: cycle-test
steps:
  - id: a
    type: llm_call
    prompt: "a"
    depends_on: [b]
  - id: b
    type: llm_call
    prompt: "b"
    depends_on: [a]
"""

INVALID_YAML_MISSING_REF = """
name: missing-ref-test
steps:
  - id: a
    type: llm_call
    prompt: "a"
    depends_on: [nonexistent]
"""
