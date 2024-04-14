"""
Main project configuration
"""

from dataclasses import dataclass
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "datasets"

DATA_PARAMS = {
    # columns to calculate rolling window statisctics with
    "calc_rolling_metrics": ["usd_xr"],
    # dict with rolling period for each column
    "rolling_period": {"usd_xr": [7]},
}


@dataclass
class KliepConfig:
    sigma: float
    z: int
    window_size: int
    verbose: bool = False


INCOME_KLIEP_CONGIF = KliepConfig(
    sigma=0.03,
    z=76,
    window_size=50,
)

OUTCOME_KLIEP_CONGIF = KliepConfig(
    sigma=0.05,
    z=53,
    window_size=60,
)
