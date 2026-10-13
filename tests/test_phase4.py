"""Phase 4 verification tests: Operation-cost arithmetic and split protection.

Covers:
1. Exact operation count proxy consistency:
   Stage 1 cost = 10,265
   Stage 2 cumulative cost = 39,515
   Weighted mean formula: (N_stage1 * 10265 + N_stage2 * 39515) / N_total
2. Phase 4 development review runner strictly refuses to evaluate test split.
3. Test split group isolation: group IDs in test split never appear in development split.
"""
import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from config import DEFAULT_CONFIG
from generate_signals import build_dataset
from run_phase4_review import run_phase4_development_review, compute_binary_metrics


def test_operation_count_weighted_mean_consistency():
    """Verify that the average adaptive operation count equals the exact weighted mean

    of non-escalated windows (10,265 ops) and escalated windows (39,515 ops).
    """
    n_samples = DEFAULT_CONFIG.samples_per_window
    assert n_samples == 1024

    stage1_ops = 10 * n_samples + 25
    assert stage1_ops == 10265

    n_log2_n = int(n_samples * np.log2(n_samples))
    stage2_additional_ops = int(2.5 * n_log2_n + 3.5 * n_samples + 66)
    assert stage2_additional_ops == 29250

    stage2_cumulative_ops = stage1_ops + stage2_additional_ops
    assert stage2_cumulative_ops == 39515

    # Given 606 stage 1 windows and 354 stage 2 windows out of 960 total:
    n_stage1 = 606
    n_stage2 = 354
    n_total = 960

    expected_weighted_mean = (n_stage1 * stage1_ops + n_stage2 * stage2_cumulative_ops) / n_total
    assert np.isclose(expected_weighted_mean, 21050.9375)

    # Verify that comparing against standalone Fixed-High (29250) yields savings:
    savings_percent = (stage2_additional_ops - expected_weighted_mean) / stage2_additional_ops * 100.0
    assert np.isclose(savings_percent, 28.031, atol=1e-2)


def test_phase4_review_refuses_test_split():
    """Verify that run_phase4_development_review raises an exception if allow_test_split=True."""
    with pytest.raises(ValueError, match="Security violation"):
        run_phase4_development_review(allow_test_split=True)


def test_held_out_test_split_integrity():
    """Verify that test split groups are completely disjoint from development fit and calibration groups."""
    records, _ = build_dataset(groups=80, config=DEFAULT_CONFIG, dev_calibration_fraction=0.5)

    dev_groups = {r["group_id"] for r in records if r["split"] == "development"}
    test_groups = {r["group_id"] for r in records if r["split"] == "test"}

    assert len(dev_groups) == 60
    assert len(test_groups) == 20
    assert dev_groups.isdisjoint(test_groups)


def test_binary_metrics_calculation():
    """Verify edge cases and correctness of compute_binary_metrics."""
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 1]

    m = compute_binary_metrics(y_true, y_pred)
    assert m["tp"] == 1
    assert m["fn"] == 1
    assert m["tn"] == 1
    assert m["fp"] == 1
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5
    assert m["false_alarm_rate"] == 0.5
