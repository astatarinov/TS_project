"""
Metrics for model validation
"""
from typing import Iterable

import numpy as np
import pandas as pd


def _calculate_rate_share(
    cbr_key_rate: float,
    rate_diff: float = 0.5,
    period: int = 1,
) -> float:
    """
    Calculate share base on rate and period in days
    """
    annual_yield = (cbr_key_rate + rate_diff) / 100
    period_share = annual_yield / 365 * period
    return period_share


def calculate_daily_profit(
    prediction: float,
    target: float,
    cbr_key_rate: float,
    profit_rate_diff: float = 0.5,
    on_deposit_rate_diff: float = -0.9,
    on_loan_rate_diff: float = 1.0,
) -> float:
    """
    Calculate daily yield in % based on predictions and rates
    """
    deviation = prediction - target
    earnings = (
        prediction * (1 + _calculate_rate_share(cbr_key_rate, profit_rate_diff))
        + min(deviation, 0) * (1 + _calculate_rate_share(cbr_key_rate, on_deposit_rate_diff))
        - max(deviation, 0) * _calculate_rate_share(cbr_key_rate, on_loan_rate_diff)
    )
    profit = (earnings - target) / target
    return profit


def calculate_total_earnings(
    predictions: Iterable[float],
    targets: Iterable[float],
    cbr_key_rate: float,
    profit_rate_diff: float = 0.5,
    on_deposit_rate_diff: float = -0.9,
    on_loan_rate_diff: float = 1.0,
) -> float:
    """
    Calculate total profit over the period
    """
    earnings = sum(
        target * calculate_daily_profit(
            pred,
            target,
            cbr_key_rate,
            profit_rate_diff,
            on_deposit_rate_diff,
            on_loan_rate_diff
        )
        for pred, target in zip(predictions, targets)
    )
    return earnings


def check_business_requirements(
    predictions: pd.Series | np.array,
    targets: pd.Series | np.array,
    max_deviation: float = 0.42,
) -> bool:
    """
    Check if model predictions satisfy business requirements
    """
    return (
        np.abs(predictions - targets) <= max_deviation
    ).all()
