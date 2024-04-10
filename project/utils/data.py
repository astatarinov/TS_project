"""
Model utilities
"""
import pandas as pd

from project.config import DATA_PATH


def load_balances_data() -> pd.DataFrame:
    """
    Return processed raw data as DataFrame
    """
    data = pd.read_csv(DATA_PATH / 'balances.csv').set_index("date")
    return data
