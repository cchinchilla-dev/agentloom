"""Base tool interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Abstract base class for tools that can be invoked by workflow steps."""

    name: str = ""
    description: str = ""
    parameters_schema: dict[str, Any] = {}

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with the given arguments.

        Args:
            **kwargs: Tool-specific arguments.

        Returns:
            Tool execution result.
        """
        ...
