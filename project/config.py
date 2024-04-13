"""
Main project configuration
"""

from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / 'datasets'

DATA_PARAMS = {
    # columns to calculate rolling window statisctics with
    "calc_rolling_metrics": ["usd_xr"],
    # dict with rolling period for each column
    "rolling_period": {
        "usd_xr": [7]
    }
}
