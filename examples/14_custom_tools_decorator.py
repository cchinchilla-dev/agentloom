#!/usr/bin/env python3
"""Custom tools using the @tool decorator — sentiment monitoring pipeline.

Demonstrates:
  - Defining custom tools with @tool decorator (auto-generates JSON Schema)
  - Registering custom tools alongside builtins
  - Loading a YAML workflow that uses custom tools
  - Realistic use case: customer feedback monitoring → sentiment analysis → alerting

Usage:
  uv run python examples/14_custom_tools_decorator.py
  uv run python examples/14_custom_tools_decorator.py --provider openai --model gpt-4.1-nano
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import anyio

from agentloom.core.engine import WorkflowEngine
from agentloom.core.parser import WorkflowParser
from agentloom.core.state import StateManager
from agentloom.providers.gateway import ProviderGateway
from agentloom.tools import ToolRegistry, tool
from agentloom.tools.builtins import register_builtins

# ---------------------------------------------------------------------------
# Custom tools — defined with the @tool decorator
# ---------------------------------------------------------------------------


@tool(name="query_database", description="Query customer feedback from database")
async def query_database(query: str, limit: int = 10) -> str:
    """Simulates a database query returning recent customer feedback."""
    # In production, this would connect to PostgreSQL/MySQL/etc.
    feedback = [
        {
            "id": 1,
            "customer": "Alice M.",
            "product": "DataFlow Pro",
            "rating": 2,
            "text": "The pipeline keeps failing on large datasets. Had 3 outages "
            "this week. Support took 48 hours to respond.",
            "date": "2026-03-18",
        },
        {
            "id": 2,
            "customer": "Bob K.",
            "product": "DataFlow Pro",
            "rating": 5,
            "text": "Incredible product. Cut our ETL processing time from 4 hours "
            "to 20 minutes. The Grafana integration is chef's kiss.",
            "date": "2026-03-17",
        },
        {
            "id": 3,
            "customer": "Carol S.",
            "product": "DataFlow Pro",
            "rating": 1,
            "text": "Billing charged me twice and nobody is answering my emails. "
            "Considering switching to Airbyte.",
            "date": "2026-03-18",
        },
        {
            "id": 4,
            "customer": "David R.",
            "product": "DataFlow Pro",
            "rating": 4,
            "text": "Solid product overall. The new real-time streaming feature "
            "is exactly what we needed. Minor UI bugs but nothing blocking.",
            "date": "2026-03-16",
        },
        {
            "id": 5,
            "customer": "Eva L.",
            "product": "DataFlow Pro",
            "rating": 1,
            "text": "Lost data during migration. This is unacceptable for a "
            "product that claims enterprise-grade reliability.",
            "date": "2026-03-19",
        },
    ]
    return json.dumps(feedback[:limit], indent=2)


@tool(name="send_slack_message", description="Send a message to a Slack channel")
async def send_slack_message(channel: str, message: str, priority: str = "normal") -> str:
    """Simulates sending a Slack message via webhook."""
    # In production, this would POST to Slack's API
    timestamp = datetime.now(UTC).isoformat()
    result = {
        "ok": True,
        "channel": channel,
        "priority": priority,
        "timestamp": timestamp,
        "message_preview": message[:100] + "..." if len(message) > 100 else message,
    }
    print(f"  [Slack] → #{channel} ({priority}): {result['message_preview']}")
    return json.dumps(result)


@tool(name="create_ticket", description="Create a support ticket in the ticketing system")
async def create_ticket(title: str, body: str, priority: str = "medium") -> str:
    """Simulates creating a Jira/Linear ticket."""
    ticket = {
        "id": "SUPPORT-1847",
        "title": title,
        "priority": priority,
        "status": "open",
        "created": datetime.now(UTC).isoformat(),
    }
    print(f"  [Ticket] Created {ticket['id']}: {title} (P: {priority})")
    return json.dumps(ticket)


# ---------------------------------------------------------------------------
# Main — parse args, set up tools + gateway, run workflow
# ---------------------------------------------------------------------------

WORKFLOW_PATH = Path(__file__).parent / "14_custom_tools_decorator.yaml"


def _setup_gateway(provider: str) -> ProviderGateway:
    """Set up providers (same pattern as CLI)."""
    gateway = ProviderGateway()

    if os.environ.get("OPENAI_API_KEY"):
        from agentloom.providers.openai import OpenAIProvider

        gateway.register(
            OpenAIProvider(),
            priority=0 if provider == "openai" else 10,
            models=["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4o-mini"],
        )

    if os.environ.get("ANTHROPIC_API_KEY"):
        from agentloom.providers.anthropic import AnthropicProvider

        gateway.register(
            AnthropicProvider(),
            priority=0 if provider == "anthropic" else 10,
            models=["claude-haiku-4-5-20251001"],
        )

    if os.environ.get("GOOGLE_API_KEY"):
        from agentloom.providers.google import GoogleProvider

        gateway.register(
            GoogleProvider(),
            priority=0 if provider == "google" else 10,
            models=["gemini-2.0-flash", "gemini-2.5-flash"],
        )

    from agentloom.providers.ollama import OllamaProvider

    gateway.register(
        OllamaProvider(),
        priority=0 if provider == "ollama" else 100,
        is_fallback=True,
    )

    return gateway


async def _main(provider: str, model: str | None) -> None:
    # 1. Set up custom tool registry
    tool_registry = ToolRegistry()
    register_builtins(tool_registry)  # keep builtins available

    # Register our custom tools
    tool_registry.register(query_database)
    tool_registry.register(send_slack_message)
    tool_registry.register(create_ticket)

    print("Registered tools:")
    for t in tool_registry.list():
        print(f"  - {t.name}: {t.description}")
    print()

    # 2. Load workflow from YAML
    workflow = WorkflowParser.from_yaml(WORKFLOW_PATH)

    if provider:
        workflow.config.provider = provider
    if model:
        workflow.config.model = model

    # 3. Set up provider gateway
    gateway = _setup_gateway(workflow.config.provider)

    # 4. Run
    state = StateManager(initial_state=dict(workflow.state))
    engine = WorkflowEngine(
        workflow=workflow,
        state_manager=state,
        provider_gateway=gateway,
        tool_registry=tool_registry,
    )

    provider_info = f"{workflow.config.provider}/{workflow.config.model}"
    print(f"Running: {workflow.name} ({provider_info})")
    print("=" * 60)

    result = await engine.run()

    print(f"\n{'=' * 60}")
    print(f"Status: {result.status.value}")
    print(f"Duration: {result.total_duration_ms:.0f}ms")
    print(f"Cost: ${result.total_cost_usd:.4f}")
    print("\nSteps:")
    for step_id, sr in result.step_results.items():
        status = sr.status.value.upper()
        print(f"  [{status}] {step_id} ({sr.duration_ms:.0f}ms)")

    await gateway.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment monitoring with custom tools")
    parser.add_argument("--provider", default="ollama", help="LLM provider (default: ollama)")
    parser.add_argument("--model", default=None, help="Override model")
    args = parser.parse_args()

    anyio.run(_main, args.provider, args.model)
