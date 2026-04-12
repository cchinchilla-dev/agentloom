# Python DSL

Build workflows programmatically with a fluent builder API.

## Basic usage

```python
from agentloom.core.dsl import workflow

wf = (
    workflow("simple-qa", provider="ollama", model="phi4")
    .set_state(question="What is Python?")
    .add_llm_step("answer", prompt="Answer: {question}", output="answer")
    .build()
)
```

The `workflow()` function returns a `WorkflowBuilder`. All methods return `self` for chaining. Call `.build()` to validate and produce a `WorkflowDefinition`.

## API reference

### `workflow()`

```python
workflow(name: str, description: str = "", **config_kwargs) -> WorkflowBuilder
```

Factory function. Config keyword arguments map to [workflow config options](workflow-yaml.md#config-options):

```python
wf = workflow(
    "my-workflow",
    provider="openai",
    model="gpt-4o-mini",
    budget_usd=0.50,
    stream=True,
)
```

### `.set_state(**kwargs)`

Initialize workflow state variables:

```python
.set_state(
    question="What is Python?",
    context="",
    items=[{"id": 1, "name": "Item A"}],
)
```

### `.add_llm_step()`

```python
.add_llm_step(
    step_id: str,
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,           # override workflow model
    output: str | None = None,
    depends_on: list[str] | None = None,
    attachments: list[Attachment] | None = None,
    stream: bool | None = None,
    **kwargs,                           # temperature, max_tokens, timeout, retry, etc.
)
```

### `.add_tool_step()`

```python
.add_tool_step(
    step_id: str,
    tool_name: str,
    tool_args: dict[str, Any] | None = None,
    output: str | None = None,
    depends_on: list[str] | None = None,
)
```

### `.add_router_step()`

```python
.add_router_step(
    step_id: str,
    conditions: list[tuple[str, str]],   # (expression, target) pairs
    default: str | None = None,
    depends_on: list[str] | None = None,
)
```

### `.add_subworkflow_step()`

```python
.add_subworkflow_step(
    step_id: str,
    workflow_path: str | None = None,
    workflow_inline: dict[str, Any] | None = None,
    output: str | None = None,
    depends_on: list[str] | None = None,
)
```

### `.build()`

Validates the workflow definition (checks for cycles, missing dependencies, etc.) and returns a `WorkflowDefinition`.

---

## Full example

A multi-step customer support workflow:

```python
from agentloom.core.dsl import workflow

wf = (
    workflow(
        "customer-support",
        provider="openai",
        model="gpt-4o-mini",
        budget_usd=0.10,
    )
    .set_state(
        user_message="My order arrived damaged",
        classification="",
        response="",
    )
    .add_llm_step(
        "classify",
        system_prompt="Respond with one word: billing, technical, or general.",
        prompt="Classify: {state.user_message}",
        output="classification",
    )
    .add_router_step(
        "route",
        depends_on=["classify"],
        conditions=[
            ("state.classification.strip().lower() == 'billing'", "handle_billing"),
            ("state.classification.strip().lower() == 'technical'", "handle_technical"),
        ],
        default="handle_general",
    )
    .add_llm_step(
        "handle_billing",
        depends_on=["route"],
        prompt="Help with billing issue: {state.user_message}",
        output="response",
    )
    .add_llm_step(
        "handle_technical",
        depends_on=["route"],
        prompt="Help with technical issue: {state.user_message}",
        output="response",
    )
    .add_llm_step(
        "handle_general",
        depends_on=["route"],
        prompt="Help with: {state.user_message}",
        output="response",
    )
    .build()
)
```

## Running a built workflow

```python
from agentloom.core.engine import WorkflowEngine

engine = WorkflowEngine(wf)
result = await engine.run()

print(result.status)            # success
print(result.final_state)       # {"user_message": "...", "response": "..."}
print(result.total_cost_usd)    # 0.002
```
