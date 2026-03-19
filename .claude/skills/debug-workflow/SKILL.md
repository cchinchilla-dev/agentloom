---
name: debug-workflow
description: Debug a workflow YAML file — validate structure, trace execution path, identify potential issues before running.
---

Debug the workflow at the path given in $ARGUMENTS (or ask for one).

1. `uv run agentloom validate $ARGUMENTS` — check YAML syntax and DAG structure
2. Read the YAML file and trace the execution:
   - What steps run in which order?
   - Are there router conditions that could be ambiguous?
   - Are all `depends_on` references valid?
   - Are tool_name references pointing to registered tools?
   - Are there state variables referenced in prompts that might not exist?
3. `uv run agentloom visualize $ARGUMENTS` — show the DAG
4. Report: what will work, what might fail, and what to watch out for at runtime
