# AI-assisted development

**Last updated:** 2026-04-28
**Applies to:** agentloom up to 0.4.0

This project is built with AI-assisted tooling. This document describes which
tools were used, how the work was split between human and AI, and what was
done to keep the result correct.

The maintainer is accountable for all code, tests, documentation, and design
decisions in this repository, whether or not they were produced with AI
assistance. AI tools do not hold authorship of this project.

---

## Scope of AI assistance

AI tools were used for software-engineering tasks: code generation, test
scaffolding, documentation drafting, code review, and release automation.

Architectural decisions, system design, component boundaries, and API design
are the maintainer's own work.

---

## Tools

### Claude Code (Anthropic)

General-purpose development assistant used for code generation, test
scaffolding, documentation drafting, and iterative problem-solving throughout
the development lifecycle.

The workflow relies on a set of custom skills defined in
[`.claude/skills/`](.claude/skills/), each encapsulating a repeatable
engineering task that Claude Code executes on demand:

| Skill | Purpose |
|---|---|
| [`analyze`](.claude/skills/analyze/) | Deep pre-commit analysis: logic bugs, architectural violations, security, performance |
| [`check`](.claude/skills/check/) | Full quality gate: format, lint (`ruff`), tests, type check (`mypy`) |
| [`ci-status`](.claude/skills/ci-status/) | Inspect GitHub Actions runs, fetch logs, and suggest fixes on failure |
| [`commit`](.claude/skills/commit/) | Stage and commit changes following the project's observed flow |
| [`coverage`](.claude/skills/coverage/) | Raise patch coverage above threshold with real edge cases |
| [`debug-workflow`](.claude/skills/debug-workflow/) | Validate workflow YAML, trace execution order, simulate state flow |
| [`gen-tests`](.claude/skills/gen-tests/) | Generate tests targeting edge cases and failure modes |
| [`issue`](.claude/skills/issue/) | Create, view, or triage GitHub issues from the terminal |
| [`pr`](.claude/skills/pr/) | Create pull requests with auto-generated description and label suggestions |
| [`release`](.claude/skills/release/) | Version bump, changelog generation, tagging, and push |
| [`review`](.claude/skills/review/) | Review staged/unstaged changes for bugs, style drift, and missing coverage |
| [`roadmap`](.claude/skills/roadmap/) | Create release-tracking issue organising open issues into a phased plan |

### GitHub Copilot (GitHub / Microsoft)

Automated code review integrated in the GitHub pull-request workflow.
Project-specific review rules are defined in
[`.github/copilot-instructions.md`](.github/copilot-instructions.md).
Review findings appear in the git history as *"address review"* or
*"fix review findings"* follow-up commits.

---

## Models used

| Tool | Model family | Period |
|---|---|---|
| Claude Code | Anthropic Claude — Sonnet 4.5 / 4.6 and Opus 4.6 / 4.7 | 2025 – present |
| GitHub Copilot review | GitHub's current review model | 2025 – present |

Model versions have changed over time as vendors released successors.

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
              Functional testing                      │
              Exploratory testing              Quality gates (check)
              Infrastructure validation        Pre-commit analysis (analyze)
                     │                                │
                     │                         PR creation (pr)
                     │                         Copilot code review
                     │                                │
              Review AI output                        │
              Address feedback                        │
              Merge decision                          │
                     │                                │
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
3. **Implement** — Feature branches are developed with Claude Code as coding
   assistant, using the skills above for continuous quality assurance.
4. **Test** — Each feature is tested hands-on: workflow YAML round-trips,
   CLI flows (`run`, `replay`, `validate`, `resume`, `history`), provider
   integrations against real APIs (OpenAI, Anthropic, Google, Ollama), and
   observability output (OTel traces, metrics, dashboards) in local
   Kubernetes (kind) and Docker.
5. **Review** — Pull requests are reviewed by GitHub Copilot against
   project-specific standards. Feedback is addressed before merging.
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
| `CLAUDE.md`, skill definitions, Copilot instructions | Predominantly human | These govern how AI is used. |
| Infrastructure (Dockerfile, Helm chart, Kustomize overlays, ArgoCD, Grafana dashboards) | AI-assisted, human-validated | Validated against a local kind cluster before merge. |
| Debugging root-cause analysis | Human-led | AI used for log inspection and hypothesis generation; root-cause calls by the maintainer. |

Line-level attribution is not tracked.

---

## Verification

AI-generated code passes through multiple layers before reaching `main`:

- **Automated quality gates** — pytest suite (unit + integration + e2e
  Ollama), `ruff` lint, `mypy --strict` type check, `version-linearity`
  gate (`pyproject.toml` ↔ `CHANGELOG.md`). Run on every commit via the
  `check` skill and the CI pipeline (9 GitHub Actions workflows: `ci`,
  `auto-tag`, `release`, `labels`, `pr-labeler`, `stale`, `docker`,
  `docs`, `e2e-ollama`).
- **Independent AI code review** — GitHub Copilot (different vendor and
  model from the one used for generation) reviews every pull request
  against project-specific standards.
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
   Ollama running in CI for the `e2e-ollama` job, kind cluster smokes for
   Helm and Kustomize validation, real-API smoke matrix (OpenAI + Anthropic
   + Google) against CLI, Docker, and Kubernetes before each release.
3. **Code review by a different vendor's model** — GitHub Copilot reviews
   code generated via Anthropic Claude, and vice-versa for review
   suggestions.

The risk is reduced, not eliminated.

---

## Traceability

Full development history is public: issues, pull requests, commits, CI runs,
releases. Every change links back to an issue and a pull request.

The rules that govern AI behaviour in this project are themselves versioned:

- [`.claude/skills/`](.claude/skills/) — custom skills and their prompts.
- [`.github/copilot-instructions.md`](.github/copilot-instructions.md) —
  Copilot review rules.
- [`CLAUDE.md`](CLAUDE.md) — contribution rules read by Claude Code on every
  invocation.

A reader can reconstruct **what changed, when, why, and under which rules**.
What cannot be reconstructed: which exact prompt produced which exact diff —
see below.

---

## What this document does not cover

- Line-level or function-level AI vs. human authorship. Not tracked.
- Exact model per commit. Not tracked.
- Prompt-to-diff mapping. Not preserved.
- "Human-reviewed" means the maintainer read the change and judged it fit to
  merge; understands it well enough to defend, extend, and debug it. It does
  not mean formal verification or line-by-line inspection.
- AI tools can produce plausible-looking but wrong code. The verification
  layers above reduce that risk; they do not eliminate it. Anything that
  ships is the maintainer's responsibility.
