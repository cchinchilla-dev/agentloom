# AI-assisted development

**Last updated:** 2026-10-17
**Applies to:** agentloom up to 0.5.0

This project is built with AI-assisted tooling. This document describes how AI
tools are used, how the work is split between human and AI, and what is done to
keep the result correct.

The maintainer is accountable for all code, tests, documentation, and design
decisions in this repository, whether or not they were produced with AI
assistance. AI tools do not hold authorship of this project.

---

## Scope of AI assistance

AI tools are used for software-engineering tasks: code generation, test
scaffolding, documentation drafting, code review, and release automation.

Architectural decisions, system design, component boundaries, and API design
are the maintainer's own work.

---

## Tools

The process is not tied to a specific assistant. Project rules live in
[`AGENTS.md`](AGENTS.md), which any compatible coding assistant reads
([`CLAUDE.md`](CLAUDE.md) imports it for Claude Code), and reusable task prompts
live in [`.agents/skills/`](.agents/skills/). The tools change over time; at the
time of writing:

| Tool | Role |
|---|---|
| Claude Code (Anthropic) | Coding assistant: implementation, tests, documentation, local review passes |
| Codex (OpenAI) | Coding assistant and second-vendor review during development |
| CodeRabbit | Light automated review on pull requests |
| GitHub Copilot code review | Automated pull-request review through 0.4.0; no longer used |

The underlying models are current frontier models from Anthropic (Claude) and
OpenAI (GPT). Versions change as vendors release successors and are not
tracked per commit.

### Skills

Each skill encapsulates a repeatable engineering task that an assistant runs on
demand. The list lives in [`.agents/skills/`](.agents/skills/); each `SKILL.md`
opens with a one-line description of what it does and when to use it.

---

## Development workflow

```
                    human                          AI-assisted
               ─────────────                  ─────────────────────
                     │                                │
              Design & scope                          │
              Define roadmap                          │
                     │                                │
                     ├───────────────────────> Issue creation (issue)
                     │                                │
              Implementation planning ◄──────► Implementation planning
                     │                                │
                     ├───────────────────────> Code generation
                     │                         Test scaffolding
                     │                         Documentation drafts
                     │                                │
              Functional testing               Quality gates (check)
              Exploratory testing              Deep review passes
              Infrastructure validation          (review skill +
                     │                            second vendor)
                     │                                │
              Triage review findings                  │
                     │                         PR creation (pr)
                     │                         CI + CodeRabbit pass
                     │                                │
              Merge decision                          │
                     │                         Release preparation (release)
              Final validation                 CI pipeline execution
                     │                                │
                     v                                v
                              Production
```

1. **Scope** — Features are designed and scoped by the maintainer. GitHub
   issues are created in batches via the `issue` skill from human-defined
   specifications and roadmap priorities.
2. **Plan** — Before writing code, maintainer and AI collaborate on an
   implementation plan: component structure, API surface, error handling
   strategy, and test approach. Technical decisions are driven by the
   maintainer.
3. **Implement** — Feature branches are developed with an AI coding assistant,
   using the skills above for continuous quality assurance.
4. **Test** — Each feature is tested hands-on: workflow YAML round-trips,
   CLI flows (`run`, `replay`, `validate`, `resume`, `history`), provider
   integrations against real APIs (OpenAI, Anthropic, Google, Ollama), and
   observability output (OTel traces, metrics, dashboards) in local
   Kubernetes (kind) and Docker.
5. **Review** — The deep review happens during development, before a pull
   request is opened: AI review passes (the `review` skill) against the
   *Review guidelines* in `AGENTS.md`, plus a pass with a model from a
   different vendor than the one that wrote the change when the change is not
   trivial. On GitHub, CI gates every pull request and CodeRabbit adds a light
   automated review, mostly useful for external contributions. The maintainer
   triages all findings.
6. **Ship** — Releases are prepared with the `release` skill and validated
   through the CI pipeline.

---

## How the work is split

Approximate, qualitative:

| Area | Predominantly | Notes |
|---|---|---|
| Architecture, module boundaries, public API | Human | AI used as sounding board and implementation partner. |
| Implementation (engine, providers, steps, observability, resilience, record/replay) | AI-assisted, human-reviewed | Generated and iterated under maintainer direction; merged after human review and CI. |
| Test suite (unit, integration, contract suites, e2e Ollama) | AI-assisted, human-curated | Edge cases and failure modes added by the maintainer during exploratory testing. |
| Documentation (README, MkDocs site, examples, CHANGELOG, deployment) | AI-drafted, human-edited | Structure and framing set by the maintainer. |
| Infrastructure (Dockerfile, Helm chart, Kustomize overlays, ArgoCD, Grafana dashboards) | AI-assisted, human-validated | Validated against a local kind cluster before merge. |
| `AGENTS.md`, skill definitions, review configuration | Predominantly human | These govern how AI is used. |
| Debugging root-cause analysis | Human-led | AI used for log inspection and hypothesis generation; root-cause calls by the maintainer. |

Line-level attribution is not tracked.

---

## Verification

AI-generated code passes through multiple layers before reaching `main`:

- **Automated quality gates** — pytest suite (unit + integration + e2e
  Ollama), `ruff` lint, `mypy --strict` type check, `version-linearity`
  gate (`pyproject.toml` ↔ `CHANGELOG.md`). Run locally via the `check`
  skill and in CI on every pull request; the e2e Ollama job runs on release
  branches, on pull requests labelled `e2e`, or on demand (9 GitHub Actions
  workflows: `ci`, `auto-tag`, `release`, `labels`, `pr-labeler`, `stale`,
  `docker`, `docs`, `e2e-ollama`).
- **Independent AI review** — Review passes during development use, where it
  matters, a model from a different vendor than the one that generated the
  change. CodeRabbit reviews pull requests against the same guidelines. Through
  0.4.0, GitHub Copilot played this role on pull requests.
- **Human review** — The maintainer reviews all generated code for
  correctness and design coherence before merging.
- **Hands-on validation** — CLI workflows (`agentloom run`, `replay`,
  `validate`, `resume`, `history`), provider integrations against real
  APIs, Kubernetes deployments via Helm and Kustomize, OTel traces in
  Jaeger and metrics in Grafana — all exercised end-to-end.

### Shared blind spots

When AI generates both production code and tests, both can miss the same
failure mode. This project treats that as a real risk and mitigates it three
ways:

1. **Hands-on exploratory testing by the maintainer** for conditions AI
   rarely covers unprompted — concurrent flush in record/replay, hash-key
   collision avoidance under provider parameter changes, atomic-rename on
   partial writes, circuit-breaker state transitions under load, sandbox
   bypass attempts, reasoning-token cost propagation, multi-turn
   tool-calling loops.
2. **Integration tests against real components** — `respx` per provider
   with realistic vendor-shaped payloads (OpenAI, Anthropic, Google, Ollama),
   Ollama running in CI for the `e2e-ollama` job, Kustomize and Helm
   manifests validated in CI and exercised by hand on a local kind cluster,
   and real-API smoke runs (OpenAI, Anthropic, Google) against CLI, Docker,
   and Kubernetes before releases.
3. **Review by a different vendor's model** — changes written with one
   vendor's model are reviewed with another's during development.

The risk is reduced, not eliminated.

---

## Traceability

Full development history is public: issues, pull requests, commits, CI runs,
releases. Most changes link back to an issue and a pull request.

The rules that govern AI behaviour in this project are themselves versioned:

- [`AGENTS.md`](AGENTS.md) — project rules and review guidelines read by coding
  assistants and by the pull-request reviewer.
- [`.agents/skills/`](.agents/skills/) — skills and their prompts.
- [`.coderabbit.yaml`](.coderabbit.yaml) — pull-request review configuration.
- `.github/copilot-instructions.md` — Copilot review rules through
  0.4.0; removed afterwards, kept in the git history.

A reader can reconstruct **what changed, when, why, and under which rules**.
What cannot be reconstructed: which exact prompt produced which exact diff —
see below.

---

## What this document does not cover

- Line-level or function-level AI vs. human authorship. Not tracked.
- Exact tool or model per commit. Not tracked.
- Prompt-to-diff mapping. Not preserved.
- "Human-reviewed" means the maintainer read the change and judged it fit to
  merge; understands it well enough to defend, extend, and debug it. It does
  not mean formal verification or line-by-line inspection.
- AI tools can produce plausible-looking but wrong code. The verification
  layers above reduce that risk; they do not eliminate it. Anything that
  ships is the maintainer's responsibility.
