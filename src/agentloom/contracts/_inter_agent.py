"""The inter-agent event log of a run (#135): event models and their JSON Schema.

Pydantic only, like ``_semconv``: the runtime (``core.inter_agent``) imports
the models from here without a warning, and ``agentloom.contracts.experimental``
re-exports them. The committed schema, ``inter_agent.schema.json`` next to this
module, is generated from the models; regenerate it with
``python -m agentloom.contracts._inter_agent``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import UUID5, BaseModel, ConfigDict, Field, TypeAdapter

SCHEMA_URI = "urn:agentloom:inter-agent-event:1"
"""The ``$id`` of the event schema; it changes when the event shape breaks."""

SCHEMA_PATH = Path(__file__).with_name("inter_agent.schema.json")
"""The committed JSON Schema of one event, as written in a log or a JSONL line."""


class _Event(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: UUID5 = Field(description="Derived from the step path, the type and the position.")
    agent_id: str = Field(description="The step's speaker, or the step itself when it has none.")
    step_id: str = Field(description="The step's path; a subworkflow's steps read parent.child.")
    parent_event_ids: list[UUID5] = Field(
        default_factory=list, description="The events this one causally follows, sorted."
    )
    payload_hash: str = Field(pattern=r"^[0-9a-f]{16}$", description="SHA-256, 16-char prefix.")
    timestamp_ns: int = Field(ge=0, description="Monotonic per process; never compared.")


class TaskSentEvent(_Event):
    """A step's result handed to a step that depends on it."""

    type: Literal["task_sent"]
    recipient_agent_id: str


class TaskReceivedEvent(_Event):
    """A step starts with what the steps it depends on handed it."""

    type: Literal["task_received"]


class ToolCallEvent(_Event):
    """A step calls a tool."""

    type: Literal["tool_call"]
    tool_name: str


class ToolReturnEvent(_Event):
    """A tool answers the call it follows."""

    type: Literal["tool_return"]
    tool_name: str


class ResponseSentEvent(_Event):
    """A step finishes with its output."""

    type: Literal["response_sent"]


class StateTransitionEvent(_Event):
    """A step's write to one state key."""

    type: Literal["state_transition"]
    key: str


InterAgentEvent = Annotated[
    TaskSentEvent
    | TaskReceivedEvent
    | ToolCallEvent
    | ToolReturnEvent
    | ResponseSentEvent
    | StateTransitionEvent,
    Field(discriminator="type"),
]
EVENT_ADAPTER: TypeAdapter[InterAgentEvent] = TypeAdapter(InterAgentEvent)


def event_schema() -> dict[str, Any]:
    """The JSON Schema of one event, as committed in ``SCHEMA_PATH``."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_URI,
        "title": "AgentLoom inter-agent event",
        **EVENT_ADAPTER.json_schema(),
    }


if __name__ == "__main__":  # pragma: no cover — regenerates the committed file
    SCHEMA_PATH.write_text(json.dumps(event_schema(), indent=2) + "\n", encoding="utf-8")
