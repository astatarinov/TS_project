"""
Model prod pipeline
"""
from dataclasses import asdict
from datetime import datetime

import pandas as pd

from project.utils.data import load_extended_data
from project.utils.metrics import calculate_daily_profit

from .config import INCOME_KLIEP_CONGIF, OUTCOME_KLIEP_CONGIF
from .kliep import change_series, perform_kliep


def get_raw_data(
    start_date: pd.Timestamp = None, current_date: pd.Timestamp = None
) -> pd.DataFrame:
    """
    Load raw data
    """
    raw_data = load_extended_data()
    start_date = start_date or raw_data.index[0]
    current_date = current_date or pd.Timestamp.now()

    return raw_data.loc[start_date:current_date]


def detect_change_point(current_date: pd.Timestamp) -> pd.Timestamp:
    """
    Detect data drifts so trained model is unusable.
    Returns last detected changepoint. If no changepoints were detected returns the first date.
    """
    print("Starting changepoint detector")

    data_till_today = get_raw_data(current_date=current_date)

    data_till_today_changed = change_series(
        series=data_till_today["income"].values, type_change="lin"
    )
    income_stats, income_changepoints = perform_kliep(
        data_val=data_till_today_changed,
        **asdict(INCOME_KLIEP_CONGIF),
    )
    data_till_today_changed = change_series(
        series=data_till_today["outcome"].values, type_change="lin"
    )
    outcome_stats, outcome_changepoints = perform_kliep(
        data_val=data_till_today_changed,
        **asdict(OUTCOME_KLIEP_CONGIF),
    )

    if len(income_changepoints) == 0 and len(outcome_changepoints) == 0:
        return data_till_today.index[0]

    last_income_changepoint = data_till_today.index[income_changepoints[-1]].date()
    print("last_income_changepoint ".upper(), last_income_changepoint)
    last_outcome_changepoint = data_till_today.index[outcome_changepoints[-1]].date()
    print("last_outcome_changepoint ".upper(), last_outcome_changepoint)

    return max(last_income_changepoint, last_outcome_changepoint)


def build_features(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Build features based on data
    """
    # todo: implement
    return raw_data


def select_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run features selection
    """
    # todo: implement
    return features_df


def split_data(dataset: pd.DataFrame, test_size: float = 0.2) -> pd.DataFrame:
    """
    Split data for train/test
    """
    # todo: implement
    # todo: pay attention to the time, shuffling, random state, stratification


def run_model_training():
    """
    Run model training
    """
    # todo: load SOTA model
    # todo: implement
    # todo: save model state to binary file
    # todo: print training logs


def run_model_validation():
    """
    Run model validation
    """
    # todo: implement
    # todo: print validation logs
    # todo: return validation results


def run_model_pipeline(start_date: pd.Timestamp, current_date: pd.Timestamp):
    """
    Main model pipeline
    """
    print("Starting model pipeline")

    print("Loading raw data")
    raw_data = get_raw_data(start_date, current_date)
    print(f"Raw data loaded. Shape: {raw_data.shape}")

    print("Building features from raw data")
    features_df = build_features(raw_data)
    print(f"Features dataset built. Shape: {features_df.shape}")

    print("Selecting features for training")
    dataset = select_features(raw_data)
    print(f"Training features selected. Shape: {dataset.shape}")

    print("Splitting dataset into train/val")
    X_train, X_test, y_train, y_test = split_data(dataset)
    print("Dataset split")

    print("Run model training")
    model = run_model_training(X_train, y_train)
    print("Model trained")

    print("Run model validation")
    val_results = run_model_validation(model)
    print("Model validated")

    print(val_results)


def get_today_data(current_date: pd.Timestamp):
    data = get_raw_data(current_date=current_date + pd.DateOffset(1))
    return data.iloc[-1]["balance"]


def get_today_(current_date: pd.Timestamp):
    data = get_raw_data(current_date=current_date + pd.DateOffset(1))
    return data.iloc[-1]


def run_full_pipeline(current_date: pd.Timestamp):
    """
    Это основная функция, которую будем вызывать для каждой даты в тестовом промежутке.
    (При смене дня в предыдущий подкладывается реальный показатель, поскольку день закончен и баланс известен)
    Она должа:
    1. детектить разладку (находит дату последнего изменения)
    2. брать данные с последней разладки
    3. на этих данных прогонять model_pipeline
    4. делать предикт на сегодня
    5. отписывать логи, насколько мы ошиблись, сколько денег заработали / потеряли и тд.
    """
    last_changepoint = detect_change_point(current_date)
    print('-' * 50)
    run_model_pipeline(start_date=last_changepoint, current_date=current_date)
    print('-' * 50)
    prediction = run_model_validation(current_date)  # предсказываем сегодняшний день [!ЭТО НУЖНО ДОПИСАТЬ!]
    today_data = get_today_data(current_date)
    today_metric = calculate_daily_profit(
        prediction, target=today_data["balance"], cbr_key_rate=today_data["key_rate"]
    )
    print(f"date: {current_date}, metric: {today_metric}")
