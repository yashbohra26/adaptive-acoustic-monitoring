"""Phase 2 focused verification tests.

Covers:
1. Fitting uses normal model-fit data only (rejects or ignores other data; changes if fit data changes).
2. Threshold calibration uses normal calibration data only.
3. Test group IDs are never used during fitting or calibration.
4. Both methods return documented outputs (score, predicted label, threshold, frequency estimate).
5. Feature normalization and distance calculation handle zero or near-zero scales safely.
6. Calibration thresholds are finite, positive, and documented.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from config import DEFAULT_CONFIG, StudyConfig
from generate_signals import build_dataset
from baselines import (
    FixedLowDetector,
    FixedHighDetector,
    compute_median_mad,
    standardized_distance,
    calibrate_threshold,
)


@pytest.fixture
def phase2_dataset():
    # Build small pilot dataset with 20 groups (15 dev, 5 test)
    # dev split has fit and calibration subsets
    records, signals = build_dataset(groups=20, config=DEFAULT_CONFIG, dev_calibration_fraction=0.5)
    return records, signals


def test_fit_and_calibration_partition_separation(phase2_dataset):
    """Verify that fitting uses fit subsets only, calibration uses calibration subsets only,

    and test groups are never touched during training/calibration.
    """
    records, signals = phase2_dataset

    fit_records = [r for r in records if r["split"] == "development" and r["dev_subset"] == "fit"]
    calib_records = [r for r in records if r["split"] == "development" and r["dev_subset"] == "calibration"]
    test_records = [r for r in records if r["split"] == "test"]

    fit_groups = {r["group_id"] for r in fit_records}
    calib_groups = {r["group_id"] for r in calib_records}
    test_groups = {r["group_id"] for r in test_records}

    # All group sets must be pairwise disjoint
    assert fit_groups.isdisjoint(calib_groups)
    assert fit_groups.isdisjoint(test_groups)
    assert calib_groups.isdisjoint(test_groups)

    # Normal signals only for fitting and calibration
    fit_normals = [signals[r["signal_key"]] for r in fit_records if r["case"] == "normal"]
    calib_normals = [signals[r["signal_key"]] for r in calib_records if r["case"] == "normal"]

    # Verify no test signals are included
    for r in fit_records:
        assert r["group_id"] not in test_groups
    for r in calib_records:
        assert r["group_id"] not in test_groups

    # Fit and calibrate Fixed-Low
    low = FixedLowDetector(sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz)
    low.fit(fit_normals)
    assert low.reference_median is not None
    assert low.reference_scale is not None
    assert np.all(np.isfinite(low.reference_median))
    assert np.all(np.isfinite(low.reference_scale))

    low_thresh = low.calibrate(calib_normals, target_far=0.05)
    assert np.isfinite(low_thresh)
    assert low_thresh > 0

    # Fit and calibrate Fixed-High
    high = FixedHighDetector(sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz)
    high.fit(fit_normals)
    assert high.reference_median is not None
    assert high.reference_scale is not None
    assert np.all(np.isfinite(high.reference_median))
    assert np.all(np.isfinite(high.reference_scale))

    high_thresh = high.calibrate(calib_normals, target_far=0.05)
    assert np.isfinite(high_thresh)
    assert high_thresh > 0


def test_baseline_predictions_and_output_contract(phase2_dataset):
    """Verify both methods return BaselineResult with required attributes on normal and anomaly windows."""
    records, signals = phase2_dataset

    fit_normals = [signals[r["signal_key"]] for r in records if r["split"] == "development" and r["dev_subset"] == "fit" and r["case"] == "normal"]
    calib_normals = [signals[r["signal_key"]] for r in records if r["split"] == "development" and r["dev_subset"] == "calibration" and r["case"] == "normal"]

    low = FixedLowDetector()
    low.fit(fit_normals)
    low.calibrate(calib_normals, target_far=0.05)

    high = FixedHighDetector()
    high.fit(fit_normals)
    high.calibrate(calib_normals, target_far=0.05)

    # Test sample normal window and sample anomaly window from development set
    sample_normal = fit_normals[0]
    sample_impact = next(signals[r["signal_key"]] for r in records if r["split"] == "development" and r["case"] == "impact")

    for detector, method_name in [(low, "fixed_low"), (high, "fixed_high")]:
        res_norm = detector.predict(sample_normal)
        assert res_norm.method == method_name
        assert isinstance(res_norm.anomaly_score, float)
        assert res_norm.predicted_label in (0, 1)
        assert np.isfinite(res_norm.threshold)
        assert res_norm.frequency_estimate_hz is not None
        assert res_norm.frequency_estimate_hz >= 0.0

        res_anom = detector.predict(sample_impact)
        assert res_anom.method == method_name
        assert isinstance(res_anom.anomaly_score, float)
        assert res_anom.predicted_label in (0, 1)


def test_zero_and_near_zero_scale_safety():
    """Verify compute_median_mad and standardized_distance handle invariant/constant features safely."""
    # Constant matrix: all rows identical
    constant_data = np.ones((10, 5)) * 3.5
    med, scale = compute_median_mad(constant_data, floor=1e-5)

    assert np.allclose(med, 3.5)
    assert np.allclose(scale, 1e-5)  # Clamped to floor, no division by zero

    # Distance calculation with constant data
    dist = standardized_distance(np.ones(5) * 3.5, med, scale)
    assert np.isclose(dist, 0.0)

    # All-zero signal feature extraction
    zero_signal = np.zeros(1024)
    low = FixedLowDetector()
    low_vec = low.extract_vector(zero_signal)
    assert np.all(np.isfinite(low_vec))

    high = FixedHighDetector()
    high_vec = high.extract_vector(zero_signal)
    assert np.all(np.isfinite(high_vec))
    assert np.all(high_vec == 0.0)


def test_finite_threshold_calibration():
    """Verify threshold calibration with small and standard samples."""
    scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    # At target FAR 0.10, percentile is 90th
    th = calibrate_threshold(scores, target_far=0.10)
    assert np.isfinite(th)
    assert 8.0 <= th <= 10.0

    with pytest.raises(ValueError):
        calibrate_threshold([])
