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


def calculate_add_margin(
    prediction: float,
    target: float,
    cbr_key_rate: float,
    profit_rate_diff: float = 0.5,
    on_deposit_rate_diff: float = -0.9,
    on_loan_rate_diff: float = 1.0,
) -> float:
    """
    Calculates additional margin that can be earned by using model
    """
    # if deficit predicted than we do nothing
    if prediction <= 0:
        return 0
    deviation = prediction - target
    add_earnings = (
        (
            # gi from derivatives
            prediction
            * _calculate_rate_share(cbr_key_rate, profit_rate_diff)
        )
        - (
            # gi that would be earned by OverNight deposit anyway
            target
            * _calculate_rate_share(cbr_key_rate, on_deposit_rate_diff)
        )
        - (
            # prediction mistake "fee"
            # if pred > target > 0, need to loan ON to cover difference
            max(deviation, 0)
            * _calculate_rate_share(cbr_key_rate, on_loan_rate_diff)
        )
    )
    return add_earnings


def calculate_total_add_margin(
    predictions: Iterable[float],
    targets: Iterable[float],
    cbr_key_rates: Iterable[float],
    profit_rate_diff: float = 0.5,
    on_deposit_rate_diff: float = -0.9,
    on_loan_rate_diff: float = 1.0,
) -> float:
    """
    Calculate total additional profit over the period
    """
    earnings = sum(
        target
        * calculate_add_margin(
            pred,
            target,
            key_rate,
            profit_rate_diff,
            on_deposit_rate_diff,
            on_loan_rate_diff,
        )
        for pred, target, key_rate in zip(predictions, targets, cbr_key_rates)
    )
    return earnings


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
        + min(deviation, 0)
        * (1 + _calculate_rate_share(cbr_key_rate, on_deposit_rate_diff))
        - max(deviation, 0) * _calculate_rate_share(cbr_key_rate, on_loan_rate_diff)
    )
    profit = (earnings - target) / target
    return profit


def calculate_total_earnings(
    predictions: Iterable[float],
    targets: Iterable[float],
    cbr_key_rates: Iterable[float],
    profit_rate_diff: float = 0.5,
    on_deposit_rate_diff: float = -0.9,
    on_loan_rate_diff: float = 1.0,
) -> float:
    """
    Calculate total profit over the period
    """
    earnings = sum(
        target
        * calculate_daily_profit(
            pred,
            target,
            key_rate,
            profit_rate_diff,
            on_deposit_rate_diff,
            on_loan_rate_diff,
        )
        for pred, target, key_rate in zip(predictions, targets, cbr_key_rates)
    )
    return earnings


def check_business_requirements_for_sample(
    prediction: float,
    target: float,
    max_deviation: float = 0.42,
) -> bool:
    """
    Check if model predictions satisfy business requirements
    """
    return abs(prediction - target) <= max_deviation


def check_business_requirements(
    predictions: pd.Series | np.ndarray,
    targets: pd.Series | np.ndarray,
    max_deviation: float = 0.42,
) -> bool:
    """
    Check if model predictions satisfy business requirements
    """
    return (np.abs(predictions - targets) <= max_deviation).all()
