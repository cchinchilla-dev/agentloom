"""Public contracts of AgentLoom, importable without loading the runtime.

- ``agentloom.contracts.stable``: shared models (``Message``, ``ToolCall``,
  ``ProviderResponse``, ...) and the ``WorkflowHandle`` protocol, under
  semantic versioning.
- ``agentloom.contracts.experimental``: telemetry names that track the
  OpenTelemetry GenAI semantic conventions, still in development upstream;
  importing it emits a ``FutureWarning``.

This package imports neither submodule, so ``import agentloom.contracts``
never warns.
"""
