"""CLI command: inspect an agent definition (#120)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
import yaml

if TYPE_CHECKING:
    from agentloom.core.models import AgentDefinition
    from agentloom.providers.pricing import ModelPricing
    from agentloom.tools.registry import ToolRegistry

agent_app = typer.Typer(help="Inspect agent definitions.", no_args_is_help=True)


class _Dumper(yaml.SafeDumper):
    """Writes a multi-line string as a literal block, as an agent file holds a system prompt."""

    def ignore_aliases(self, data: Any) -> bool:
        return True  # two agents calling one agent print it twice, not as ``&id001``


def _text(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    return dumper.represent_scalar(
        "tag:yaml.org,2002:str", value, style="|" if "\n" in value else None
    )


_Dumper.add_representer(str, _text)


@agent_app.command("inspect")
def inspect(
    agent_path: Path = typer.Argument(..., help="Path to the agent YAML file.", exists=True),
    as_json: bool = typer.Option(False, "--json", help="Print JSON instead of YAML."),
) -> None:
    """Show an agent as it loads: its model and prices, its tools with their parameters
    (agents used as tools included), memory, loop policy, output schema and constraints."""
    # Here, not at the top: every command of the CLI loads this module
    from agentloom.cli.run import _import_from_cwd
    from agentloom.core.agent import load_agent
    from agentloom.providers.pricing import load_pricing
    from agentloom.tools.builtins import register_builtins
    from agentloom.tools.registry import ToolRegistry

    _import_from_cwd()  # a pydantic output_schema names a module of the project
    builtins = ToolRegistry()
    register_builtins(builtins)
    try:
        described = describe(load_agent(agent_path), builtins, load_pricing())
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None
    if as_json:
        typer.echo(json.dumps(described, indent=2, ensure_ascii=False))
    else:
        typer.echo(
            yaml.dump(described, Dumper=_Dumper, sort_keys=False, allow_unicode=True), nl=False
        )


def describe(
    agent: AgentDefinition, builtins: ToolRegistry, pricing: dict[str, ModelPricing]
) -> dict[str, Any]:
    """*agent* as plain data, the agents it uses as tools nested under their tool."""
    from agentloom.providers.pricing import price_of
    from agentloom.steps._structured import schema_dict_for

    price = price_of(agent.model, pricing) if agent.model else None
    schema = agent.output_schema
    return {
        "name": agent.name,
        "description": agent.description,
        # None: the workflow's model, or the one passed to ``Agent``
        "model": agent.model,
        "pricing": {
            "input_per_1k": price.input_cost_per_1k,
            "output_per_1k": price.output_cost_per_1k,
        }
        if price
        else None,
        "system_prompt": agent.system_prompt,
        "temperature": agent.temperature,
        "max_tokens": agent.max_tokens,
        "tools": [_tool(entry, builtins, pricing) for entry in agent.tools],
        "memory": agent.memory.model_dump() if agent.memory else None,
        "loop_policy": agent.loop_policy.model_dump(),
        "output_schema": schema
        and {
            **schema.model_dump(by_alias=True, exclude_none=True),
            "json_schema": schema_dict_for(schema, agent.name),
        },
        "constraints": agent.constraints.model_dump(),
    }


def _tool(entry: Any, builtins: ToolRegistry, pricing: dict[str, ModelPricing]) -> dict[str, Any]:
    from agentloom.steps.agent import AGENT_TOOL_PARAMETERS

    if not isinstance(entry, str):
        return {
            "name": entry.name,
            "description": entry.description or entry.agent.description,
            "parameters": AGENT_TOOL_PARAMETERS,
            "agent": describe(entry.agent, builtins, pricing),
        }
    known = {t.name: t for t in builtins.list()}.get(entry)
    # Another tool is the workflow's to register, from Python: only its name is known here
    return {
        "name": entry,
        "builtin": known is not None,
        "description": known.description if known else None,
        "parameters": known.schema if known else None,
    }
