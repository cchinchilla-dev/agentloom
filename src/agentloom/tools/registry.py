"""Tool registry for managing available tools."""

from __future__ import annotations

from typing import Any

from agentloom.tools.base import BaseTool


class ToolInfo:
    """Metadata about a registered tool."""

    def __init__(self, name: str, description: str, schema: dict[str, Any]) -> None:
        self.name = name
        self.description = description
        self.schema = schema


class ToolRegistry:
    """Registry of tools available to workflow steps."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        if not tool.name:
            raise ValueError("Tool must have a 'name' attribute")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        """Get a tool by name.

        Raises:
            ToolNotFoundError: If tool is not registered. Subclass of
                ``KeyError`` so pre-0.5.0 ``except KeyError`` call sites
                keep working; the change is purely additive — the
                resilience layer now sees a non-retryable marker on the
                exception and skips retries.
        """
        if name not in self._tools:
            # Lazy import keeps ``tools/registry.py`` free of an import
            # edge into ``exceptions.py`` at module load.
            from agentloom.exceptions import ToolNotFoundError

            raise ToolNotFoundError(name, list(self._tools.keys()))
        return self._tools[name]

    def list(self) -> list[ToolInfo]:
        """List all registered tools."""
        return [
            ToolInfo(
                name=t.name,
                description=t.description,
                schema=t.parameters_schema,
            )
            for t in self._tools.values()
        ]

    def to_provider_format(self, provider: str) -> list[dict[str, Any]]:
        """Convert tools to the function-calling format of a provider.

        Args:
            provider: Provider name ('openai', 'anthropic', 'google').

        Returns:
            List of tool definitions in provider-specific format.
        """
        if provider in ("openai", "google"):
            return [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters_schema,
                    },
                }
                for t in self._tools.values()
            ]
        elif provider == "anthropic":
            return [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters_schema,
                }
                for t in self._tools.values()
            ]
        else:
            return self.to_provider_format("openai")
