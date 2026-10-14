# Antigravity and AI handoff workflow

Use AI coding tools one phase at a time. The repository root `AGENTS.md` contains the project constraints. Start every prompt by asking the tool to read `AGENTS.md`, `README.md`, the phase plan, and the experiment specification.

## Current Phase and Completed Milestones

- **Phases 1–5:** Completed. Signal generation, fixed baselines, adaptive selector, development review, CV study, protocol freezes (v1.0.0, v1.1.0, v1.2.0), and held-out test evaluations completed.
- **Phase 6:** Embedded firmware demonstration port completed in `esp32_demo/` with 100% desktop C++ parity; Wokwi live simulation execution remains pending manual launch.
- **Engineering Audit & Corrections:** Single-pass rFFT spectral extraction implemented; fair benchmarking completed across all methods under uniform warm timing; difficult-case investigation and independent validation of Candidate v1.2 completed; 5 report-ready figures and tables generated; formal prior art review documented in `docs/PRIOR_ART.md`.
- **Phase 7 (Research Report):** Deferred until explicitly resumed by user.

## Prompt pattern for future work

## What to bring back for review

After Antigravity responds, paste its complete summary and, where available, its diff or changed-file list. Include any check output and questions it raises. We will review the changes against the project rules before starting Phase 2.

## Standard prompt pattern for later phases

Replace `<PHASE>` with the exact phase and its deliverables from `PROJECT_PLAN.md`:

```text
Read AGENTS.md, README.md, docs/PROJECT_PLAN.md, and docs/EXPERIMENT_SPEC.md. Work only on <PHASE>. Preserve the frozen signal definitions, grouped split, and development/test boundary. Do not start later phases or write the report. If an open methodological choice blocks implementation, explain the options and tradeoffs before silently choosing. Implement the smallest complete phase slice, run relevant checks, and report changed files, exact commands/results, limitations, and decisions needed. Never claim real hardware behavior or energy use from simulation.
```
