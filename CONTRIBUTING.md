# Contributing to Adaptive Acoustic Monitoring

Thank you for contributing to this research study. We follow a careful, phase-by-phase workflow to maintain scientific reproducibility and fair method comparison.

## Getting Started

1. **Clone or navigate to the repository:**
   ```powershell
   git status
   ```
2. **Environment setup:**
   Ensure Python 3.10+ is installed. Install required Phase 1 dependencies:
   ```powershell
   py -m pip install -r requirements.txt
   ```
3. **Run Phase 1 generation & preview:**
   ```powershell
   py python/generate_signals.py --out results/preview --groups 8
   ```

## Development and Review Guidelines

1. **One Phase at a Time:**
   - Follow the phases outlined in [PROJECT_PLAN.md](docs/PROJECT_PLAN.md).
   - Do not implement subsequent phases or jump ahead without protocol freeze.
2. **Method Fairness & Protocol Integrity:**
   - Group-based splitting is strictly enforced. Never split noise variants of the same source group across development and test splits.
   - All thresholds, rules, and calibrations must be derived exclusively from the development split.
   - Keep operation counts, Python execution runtime, simulated firmware time, and energy metrics distinct.
3. **No Unsubstantiated Claims:**
   - Do not claim physical machine validation, measured ESP32 energy savings, publication acceptance, or patentability based on synthetic simulation.
4. **Documentation & Handoff:**
   - Any proposed changes to signal definitions, split policy, or evaluation metrics must be documented in [EXPERIMENT_SPEC.md](docs/EXPERIMENT_SPEC.md) before or alongside code updates.
   - If using AI coding tools, adhere strictly to the prompts and constraints in [AGENTS.md](AGENTS.md) and [AI_HANDOFF.md](docs/AI_HANDOFF.md).
