"""Central settings for the controlled acoustic signal study."""
from dataclasses import dataclass


@dataclass(frozen=True)
class StudyConfig:
    sample_rate_hz: int = 4000
    duration_s: float = 0.256
    snr_db_levels: tuple[float, ...] = (30.0, 20.0, 10.0, 0.0)
    normal_frequency_range_hz: tuple[float, float] = (180.0, 260.0)
    shift_ratio_range: tuple[float, float] = (1.35, 1.60)
    impact_decay_s: float = 0.009
    impact_carrier_range_hz: tuple[float, float] = (500.0, 900.0)
    test_fraction: float = 0.25
    seed: int = 20261013

    @property
    def samples_per_window(self) -> int:
        return round(self.sample_rate_hz * self.duration_s)


DEFAULT_CONFIG = StudyConfig()
SIGNAL_CASES = ("normal", "frequency_shift", "added_harmonic", "impact")
