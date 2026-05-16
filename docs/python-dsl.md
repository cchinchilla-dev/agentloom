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

## State contract — atomic updates

`StateManager.set` and `StateManager.get` each take the state lock individually, but the natural `cur = await sm.get('counter'); await sm.set('counter', cur + 1)` pattern drops the lock between the two awaits. Anything that yields control between them (a tool call, an `await anyio.sleep(0)`) lets another writer race, and the slower writer overwrites the faster one's result — 50 parallel counter bumps deterministically collapse to 1.

Use `StateManager.update(key, fn)` for compound read-modify-write. It holds the lock across the full `fn(current)` invocation:

```python
# Atomic — 50 parallel bumps land 50 increments.
async def bump(sm: StateManager) -> None:
    await sm.update("counter", lambda c: (c or 0) + 1)

# Racy — collapses under any concurrency. Documented as such.
async def bump_racy(sm: StateManager) -> None:
    cur = await sm.get("counter")
    await anyio.sleep(0)
    await sm.set("counter", cur + 1)
```

`fn` must be **synchronous and side-effect-free** — it runs while the lock is held, so a blocking call stalls every other state operation. For async transformations, compute the new value outside the lock and pass a lambda that returns the already-computed value (this still races on read-modify-write, so for that case use `update` with a sync function).

If the key does not exist yet, `fn` receives `None`; the caller chooses how to seed it.

### Dotted writes refuse to overwrite scalars

Writing `output: "user.name"` when `state.user` was a scalar (`"alice"`) used to silently replace the string with `{"name": ...}` and lose the original value. The current contract raises `StateWriteError` with a message naming the traversed prefix and the existing type — pick a different output key, or overwrite `user` at the parent path first.

```python
sm = StateManager(initial_state={"user": "alice"})
await sm.set("user.name", "bob")
# StateWriteError: Cannot write to 'user.name': intermediate 'user' is a str
# (value='alice'), not a dict. Refusing to silently overwrite the scalar.
```

Missing intermediates still auto-create as dicts (`set("user.name", "bob")` on an empty state works), and existing dict / list intermediates traverse unchanged.
