# Instructions for coding agents

## Project context

This is a software-only acoustic signal-processing research mini-project. Python is the experimental platform. A later Wokwi ESP32 sketch will demonstrate the adaptive decision flow. The report is deferred.

## Before changing code

1. Read `README.md`, `docs/PROJECT_PLAN.md`, and `docs/EXPERIMENT_SPEC.md`.
2. Check the current phase and do only the requested phase or milestone.
3. Preserve reproducibility: use the configured random seeds, record settings, and keep source groups intact across splits.

## Method rules

- The three methods must see the same labelled windows and obey the same development/test boundary.
- A feature calculation is not a detection rule. Specify exactly how each method classifies a window and how its thresholds are calibrated.
- Do not tune on held-out test data.
- Record adaptive stage/escalation counts and include selector overhead in its work accounting.
- Report Python runtime and operation-count proxy as separate quantities. Neither represents measured ESP32 energy.
- Frequency error is evaluated only for cases with an unambiguous defined frequency target.
- Synthetic cases are not real machine recordings. Do not claim real-world validation, measured power savings, publication acceptance, or patentability.

## Change hygiene

- Keep dependencies minimal and explain any new dependency.
- Do not silently change signal definitions, labels, splits, thresholds, or evaluation metrics. Update the experiment specification and call out the impact.
- Do not write the research report unless the user explicitly resumes that phase.
- When completing a requested change, summarize files changed, commands run, outcomes, limitations, and the next decision needed. Do not claim checks that were not run.
