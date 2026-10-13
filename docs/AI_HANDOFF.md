# Antigravity and AI handoff workflow

Use AI coding tools one phase at a time. The repository root `AGENTS.md` contains the project constraints. Start every prompt by asking the tool to read `AGENTS.md`, `README.md`, the phase plan, and the experiment specification.

## Prompt for the next task: complete Phase 1 review

Copy this prompt to Antigravity:

```text
Work only on Phase 1 (signal and feature foundation) of this project. First read AGENTS.md, README.md, docs/PROJECT_PLAN.md, and docs/EXPERIMENT_SPEC.md.

Inspect the existing Python files and generated preview outputs. Do not implement the three detector methods, the experiment runner, Wokwi firmware, or the report.

Review Phase 1 for correctness and reproducibility. In particular:
1. Confirm a fixed seed reproduces the same signal/metadata and that changing SNR changes noise while preserving the underlying clean recipe for each group/case.
2. Confirm group IDs are disjoint between development and test, and all variants from a group stay together.
3. Check labels, SNR construction, frequency targets, waveform/spectrum preview, and feature calculations on known signals.
4. Fix only concrete Phase 1 defects. Avoid changing the scientific scope or silently changing definitions; document any necessary proposal first.
5. Run appropriate checks or small reproducibility/feature checks, and report exact commands and outcomes.

At the end, list changed files, checks run and results, remaining limitations, and any decision needed from the team. Do not claim physical validation or energy measurements.
```

## What to bring back for review

After Antigravity responds, paste its complete summary and, where available, its diff or changed-file list. Include any check output and questions it raises. We will review the changes against the project rules before starting Phase 2.

## Standard prompt pattern for later phases

Replace `<PHASE>` with the exact phase and its deliverables from `PROJECT_PLAN.md`:

```text
Read AGENTS.md, README.md, docs/PROJECT_PLAN.md, and docs/EXPERIMENT_SPEC.md. Work only on <PHASE>. Preserve the frozen signal definitions, grouped split, and development/test boundary. Do not start later phases or write the report. If an open methodological choice blocks implementation, explain the options and tradeoffs before silently choosing. Implement the smallest complete phase slice, run relevant checks, and report changed files, exact commands/results, limitations, and decisions needed. Never claim real hardware behavior or energy use from simulation.
```
