# CLI reference

The `agentloom` command is installed with the package. Every command accepts `--help`.

| Command | Purpose |
|---------|---------|
| [`run`](#run) | Execute a workflow |
| [`replay`](#replay) | Re-run a workflow offline against a recording |
| [`verify-determinism`](#verify-determinism) | Replay a recording several times and check that every replay gives the same result |
| [`resume`](#resume) | Continue a checkpointed or paused run |
| [`runs`](#runs) | List checkpointed runs |
| [`history`](#history) | Query per-run history records |
| [`validate`](#validate) | Check a workflow file without running it |
| [`visualize`](#visualize) | Draw the workflow DAG |
| [`info`](#info) | Show version, dependencies, and provider configuration |
| [`callback-server`](#callback-server) | Serve approval callbacks and webhook notifications |

## `run`

```bash
agentloom run WORKFLOW.yaml [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `-s, --state key=value` | Set an initial state value; repeat for several keys |
| `-p, --provider NAME` | Override the default provider |
| `-m, --model NAME` | Override the default model |
| `-b, --budget USD` | Maximum budget for the run |
| `--stream` | Stream LLM output as it arrives |
| `--checkpoint` | Write checkpoints so the run can be resumed |
| `--checkpoint-dir DIR` | Checkpoint directory (default `.agentloom/checkpoints`) |
| `--record FILE` | Record provider responses, and the [inter-agent event log](testing-and-replay.md#inter-agent-events), to a JSON file for later replay |
| `--record-tools` | With `--record`, also record tool results, for `replay --tools` (see [Recording tool results](testing-and-replay.md#recording-tool-results)) |
| `--mock-responses FILE` | Answer from a recording instead of calling providers |
| `--lite` | Disable observability |
| `--json` | Print the result as JSON |

`--record` and `--mock-responses` are mutually exclusive.

## `replay`

```bash
agentloom replay WORKFLOW.yaml --recording FILE [OPTIONS]
```

Runs the workflow against a recording with no network calls, unless a cut runs part of it live (see [Replaying up to a cut point](testing-and-replay.md#replaying-up-to-a-cut-point)). Replay is strict: a request that is not in the recording fails the run (see [Testing & Replay](testing-and-replay.md#strict-mode)).

| Option | Description |
|--------|-------------|
| `-r, --recording FILE` | Recording to replay (required) |
| `-s, --state key=value` | Set an initial state value |
| `--allow-default-fallback` | Return a placeholder response on a recording miss instead of failing |
| `--tools live\|replay\|verify` | Tool calls: `live` (default) runs them, `replay` serves the results recorded with `--record-tools`, `verify` runs them and fails when a result differs |
| `--live-from STEP` | Serve the steps before `STEP` from the recording, and run it and every step that depends on it live, with the workflow's providers |
| `--force-live STEP` | Run `STEP` live even when the recording matches it; repeat for several steps |
| `--record FILE` | With `--live-from` or `--force-live`, record the run to a new file |
| `--record-tools` | With `--record`, also record tool results |
| `--observability` | Enable observability (off by default in replay) |
| `--json` | Print the result as JSON |

## `verify-determinism`

!!! info "New in 0.5.0 (unreleased)"
    Available on `main`; the 0.4.0 release on PyPI does not include it yet.

```bash
agentloom verify-determinism WORKFLOW.yaml --recording FILE [OPTIONS]
```

Replays the recording strictly several times and compares the results and the [inter-agent event log](testing-and-replay.md#inter-agent-events) across the replays and with the recorded log. Exits 1 with a unified diff on stderr on any difference.

| Option | Description |
|--------|-------------|
| `-r, --recording FILE` | Recording to replay (required) |
| `--runs N` | How many replays to compare (default 2, at least 2) |
| `-s, --state key=value` | Set an initial state value |
| `--tools live\|replay\|verify` | How tool calls run, as in `replay` (default `live`) |
| `--events-out FILE` | Write the event log of the first replay as JSONL |

## `resume`

```bash
agentloom resume RUN_ID [OPTIONS]
```

Continues a run from its last checkpoint. For a run paused at an [`approval_gate`](workflow-yaml.md#approval_gate), pass the decision:

| Option | Description |
|--------|-------------|
| `--approve` / `--reject` | Decision for a pending approval gate |
| `--checkpoint-dir DIR` | Checkpoint directory (default `.agentloom/checkpoints`) |
| `-p, --provider NAME` / `-m, --model NAME` | Override provider or model |
| `--stream`, `--lite`, `--json` | Same as in `run` |

## `runs`

```bash
agentloom runs [--checkpoint-dir DIR] [--json]
```

Lists checkpointed runs with their workflow, status, and last update.

## `history`

!!! info "New in 0.5.0 (unreleased)"
    Available on `main`; the 0.4.0 release on PyPI does not include it yet.

```bash
agentloom history [OPTIONS]
```

Queries the per-run records described in [Observability](observability.md#per-run-history-records).

| Option | Description |
|--------|-------------|
| `--runs-dir DIR` | Records directory (default `./agentloom_runs`, or `AGENTLOOM_RUNS_DIR`) |
| `--workflow NAME` | Filter by workflow name |
| `--provider NAME` | Filter by provider |
| `--since DATE` / `--until DATE` | Filter by timestamp (`YYYY-MM-DD` or ISO 8601) |
| `--min-cost USD` / `--max-cost USD` | Filter by total cost |
| `--limit N` | Maximum records to show (default 20) |
| `--json` | Print as JSON |

## `validate`

```bash
agentloom validate WORKFLOW.yaml
```

Parses the workflow and builds its DAG without calling any provider, then prints the step count, execution layers, provider, and model. Any schema or dependency error exits with status 1.

## `visualize`

```bash
agentloom visualize WORKFLOW.yaml [--format ascii|mermaid]
```

Prints the DAG as an ASCII diagram (default) or as a Mermaid graph.

## `info`

```bash
agentloom info
```

Shows the AgentLoom and Python versions, core and optional dependencies, and which provider credentials are configured.

## `callback-server`

```bash
agentloom callback-server [--port 8642] [--host 0.0.0.0] [--token SECRET]
```

Serves `approve` / `reject` callbacks for paused runs and receives approval-gate webhooks. Endpoints and security notes are in [Deployment](deployment.md#approval-callback-server).
