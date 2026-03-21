#!/usr/bin/env python3
"""Custom tools using BaseTool subclass — customer data enrichment pipeline.

Demonstrates:
  - Defining custom tools by subclassing BaseTool (full control over schema)
  - Loading a YAML workflow that references the custom tools
  - Registering custom tools and running the workflow programmatically
  - Realistic use case: enrich customer profile → risk scoring → personalized action

Usage:
  uv run python examples/15_custom_tools_subclass.py
  uv run python examples/15_custom_tools_subclass.py --provider openai --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import anyio

from agentloom.core.engine import WorkflowEngine
from agentloom.core.parser import WorkflowParser
from agentloom.core.state import StateManager
from agentloom.providers.gateway import ProviderGateway
from agentloom.tools import BaseTool, ToolRegistry
from agentloom.tools.builtins import register_builtins

# ---------------------------------------------------------------------------
# Custom tools — defined by subclassing BaseTool
# ---------------------------------------------------------------------------


class GeocodingTool(BaseTool):
    """Geocodes an address into coordinates and enriched location data."""

    name = "geocode_address"
    description = "Geocode an address into lat/lon coordinates and location metadata"
    parameters_schema = {
        "type": "object",
        "properties": {
            "address": {
                "type": "string",
                "description": "Full street address to geocode",
            },
            "include_demographics": {
                "type": "boolean",
                "description": "Whether to include area demographics",
                "default": False,
            },
        },
        "required": ["address"],
    }

    # Simulated geocoding results
    _MOCK_RESULTS = {
        "default": {
            "lat": 37.7749,
            "lon": -122.4194,
            "city": "San Francisco",
            "state": "CA",
            "country": "US",
            "zip": "94105",
            "timezone": "America/Los_Angeles",
            "metro_area": "San Francisco-Oakland-Berkeley",
            "demographics": {
                "median_income": 112449,
                "population": 873965,
                "tech_employment_pct": 18.4,
                "business_density_per_sqmi": 342,
            },
        }
    }

    async def execute(self, **kwargs: Any) -> Any:
        address = kwargs["address"]
        include_demographics = kwargs.get("include_demographics", False)

        # In production: call Google Maps / Mapbox / Nominatim API
        result = dict(self._MOCK_RESULTS["default"])
        result["input_address"] = address

        if not include_demographics:
            result.pop("demographics", None)

        return json.dumps(result, indent=2)


class CRMLookupTool(BaseTool):
    """Looks up a customer profile in the CRM system."""

    name = "crm_lookup"
    description = "Retrieve customer profile and history from CRM"
    parameters_schema = {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "Customer email address",
            },
            "include_history": {
                "type": "boolean",
                "description": "Include purchase and interaction history",
                "default": True,
            },
        },
        "required": ["email"],
    }

    async def execute(self, **kwargs: Any) -> Any:
        email = kwargs["email"]
        include_history = kwargs.get("include_history", True)

        # In production: query Salesforce / HubSpot / custom CRM
        profile = {
            "email": email,
            "name": "Sarah Chen",
            "company": "TechCorp Inc.",
            "title": "VP of Engineering",
            "industry": "SaaS / Developer Tools",
            "company_size": "150-500 employees",
            "annual_revenue": "$25M-$50M",
            "account_tier": "enterprise",
            "customer_since": "2024-06-15",
            "health_score": 72,
            "nps_score": 7,
            "assigned_csm": "Jordan Park",
        }

        if include_history:
            profile["history"] = {
                "total_spent": 84000,
                "active_licenses": 45,
                "support_tickets_90d": 8,
                "avg_ticket_resolution_hours": 18.5,
                "last_interaction": "2026-03-15",
                "last_interaction_type": "support_ticket",
                "renewal_date": "2026-06-15",
                "expansion_opportunities": [
                    "Team plan → Enterprise (12 seats unused)",
                    "Add-on: Advanced Analytics module",
                ],
                "risk_signals": [
                    "Support ticket volume up 60% in 90 days",
                    "NPS dropped from 9 to 7 in last survey",
                    "Champion (CTO) left the company last month",
                ],
            }

        return json.dumps(profile, indent=2)


class RiskScoreTool(BaseTool):
    """Computes a numeric churn risk score from structured inputs."""

    name = "compute_risk_score"
    description = "Compute a numeric churn risk score (0-100) from customer signals"
    parameters_schema = {
        "type": "object",
        "properties": {
            "health_score": {
                "type": "integer",
                "description": "Current account health score (0-100)",
            },
            "nps_score": {
                "type": "integer",
                "description": "Latest NPS score (0-10)",
            },
            "ticket_volume_trend": {
                "type": "string",
                "description": "Ticket volume trend: increasing, stable, or decreasing",
            },
            "days_to_renewal": {
                "type": "integer",
                "description": "Days until contract renewal",
            },
            "champion_departed": {
                "type": "boolean",
                "description": "Whether the internal champion has left",
            },
        },
        "required": [
            "health_score",
            "nps_score",
            "ticket_volume_trend",
            "days_to_renewal",
        ],
    }

    async def execute(self, **kwargs: Any) -> Any:
        # Deterministic scoring algorithm (not ML, but production-realistic)
        health = kwargs.get("health_score", 50)
        nps = kwargs.get("nps_score", 5)
        trend = kwargs.get("ticket_volume_trend", "stable")
        days_renewal = kwargs.get("days_to_renewal", 365)
        champion_left = kwargs.get("champion_departed", False)

        # Base risk from health score (inverted)
        risk = 100 - health

        # NPS adjustment (-20 to +10)
        if nps <= 3:
            risk += 20
        elif nps <= 6:
            risk += 10
        elif nps >= 9:
            risk -= 10

        # Ticket trend
        if trend == "increasing":
            risk += 15
        elif trend == "decreasing":
            risk -= 5

        # Renewal proximity amplifier
        if days_renewal < 90:
            risk = int(risk * 1.3)

        # Champion departure
        if champion_left:
            risk += 20

        # Clamp and add slight variance
        risk = max(0, min(100, risk + random.randint(-3, 3)))

        result = {
            "risk_score": risk,
            "risk_level": "high" if risk >= 70 else "medium" if risk >= 40 else "low",
            "factors": {
                "health_contribution": 100 - health,
                "nps_contribution": "negative" if nps <= 6 else "positive",
                "trend_impact": trend,
                "renewal_urgency": "high" if days_renewal < 90 else "normal",
                "champion_risk": champion_left,
            },
        }

        return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Main — parse args, set up tools + gateway, run workflow
# ---------------------------------------------------------------------------

WORKFLOW_PATH = Path(__file__).parent / "15_custom_tools_subclass.yaml"


def _setup_gateway(provider: str) -> ProviderGateway:
    gateway = ProviderGateway()

    if os.environ.get("OPENAI_API_KEY"):
        from agentloom.providers.openai import OpenAIProvider

        gateway.register(
            OpenAIProvider(),
            priority=0 if provider == "openai" else 10,
            models=["gpt-4o-mini", "gpt-4o", "gpt-4.1"],
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
    # 1. Set up tool registry with custom tools
    tool_registry = ToolRegistry()
    register_builtins(tool_registry)

    # Register custom tools (subclass instances)
    tool_registry.register(GeocodingTool())
    tool_registry.register(CRMLookupTool())
    tool_registry.register(RiskScoreTool())

    print("Registered tools:")
    for t in tool_registry.list():
        schema_fields = list(t.schema.get("properties", {}).keys())
        print(f"  - {t.name}: {t.description}")
        print(f"    params: {schema_fields}")
    print()

    # 2. Load workflow from YAML
    workflow = WorkflowParser.from_yaml(WORKFLOW_PATH)

    if provider:
        workflow.config.provider = provider
    if model:
        workflow.config.model = model

    print(f"Workflow: {workflow.name}")
    print(f"Steps: {len(workflow.steps)}")
    print(f"Provider: {workflow.config.provider} / {workflow.config.model}")
    print("=" * 60)

    # 3. Run
    gateway = _setup_gateway(workflow.config.provider)
    state = StateManager(initial_state=dict(workflow.state))
    engine = WorkflowEngine(
        workflow=workflow,
        state_manager=state,
        provider_gateway=gateway,
        tool_registry=tool_registry,
    )

    result = await engine.run()

    print(f"\n{'=' * 60}")
    print(f"Status: {result.status.value}")
    print(f"Duration: {result.total_duration_ms:.0f}ms")
    print(f"Cost: ${result.total_cost_usd:.4f}")
    print("\nSteps:")
    for step_id, sr in result.step_results.items():
        status = sr.status.value.upper()
        print(f"  [{status}] {step_id} ({sr.duration_ms:.0f}ms)")

    # Print final strategy
    final = result.final_state.get("retention_strategy", "")
    if final:
        print(f"\n{'=' * 60}")
        print("RETENTION STRATEGY:")
        print(final[:500] + ("..." if len(str(final)) > 500 else ""))

    await gateway.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer enrichment with custom tools")
    parser.add_argument("--provider", default="ollama", help="LLM provider (default: ollama)")
    parser.add_argument("--model", default=None, help="Override model")
    args = parser.parse_args()

    anyio.run(_main, args.provider, args.model)
