"""
Model prod pipeline
"""

from dataclasses import asdict

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from tsfresh import extract_relevant_features
from tsfresh.feature_extraction import settings

from project.utils.data import load_extended_data, create_target_features, select_topN_mutual_info
from .config import INCOME_KLIEP_CONGIF, OUTCOME_KLIEP_CONGIF, DATA_PARAMS
from .kliep import change_series, perform_kliep
from .model.catboost import catboost_ts_model_fit
from .utils.metrics import calculate_add_margin


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
    Returns last detected changepoint. If no change points were detected returns the first date.
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
    Build features based on data via tsfresh
    """
    X = raw_data.drop(columns="balance")
    data_long = pd.DataFrame(
        {
            "values": X.values.flatten(),
            "id": np.arange(X.shape[0]).repeat(X.shape[1]),
        }
    )

    y_indexed = raw_data["balance"].copy()
    y_indexed.index = np.arange(X.shape[0])

    ts_settings = settings.ComprehensiveFCParameters()

    features_df = extract_relevant_features(
        data_long,
        y_indexed,
        column_id="id",
        default_fc_parameters=ts_settings,
    )
    features_df.index = X.index

    features_df = features_df.join(raw_data["balance"])
    return features_df


def run_model_validation(model, sample) -> float:
    """
    Run model validation
    """
    return model.predict(sample)


def train_model(
    start_date: pd.Timestamp,
    current_date: pd.Timestamp,
    min_samples_for_training: int,
    use_tsfresh: bool = False,
):
    """
    Main model pipeline
    Return trained model object
    """
    print("Starting model pipeline")

    print("Loading raw data")
    raw_data = get_raw_data(start_date, current_date)
    print(f"Raw data loaded. Shape: {raw_data.shape}")
    target_columns = ["balance", "income", "outcome", "income - outcome"]

    if raw_data.shape[0] < min_samples_for_training:
        print("Too small dataset for training. Skip training")
        return

    if use_tsfresh:
        print("Building features from raw data using tsfresh")
        features_df = build_features(raw_data)
        print(f"Features dataset built. Shape: {features_df.shape}")
    else:
        print("tsfresh is disabled. Pre-built features loaded")
        features_df = create_target_features(raw_data, target_name="balance")

        print("FS using Mutual Information")
        selected_features = select_topN_mutual_info(
            y = features_df["balance"],
            X=features_df.drop(columns=target_columns),
            N=DATA_PARAMS["max_features"]
        )

    print("Run model training")

    model = CatBoostRegressor(verbose=0,)

    param_grid = {
        'iterations': [50, 100, 200,],
        'learning_rate': [0.1, 1],
        'depth': [4, 5, 7],
    }
    if use_tsfresh:
        param_grid = {"depth": [5], "iterations": [200], "learning_rate": [0.1]}
    best_model, mae_test, additional_metric_result, best_params = catboost_ts_model_fit(
        target=features_df["balance"],
        features=features_df[selected_features],
        params_grid=param_grid, model_class=model, cv_window='expanding',
    )
    print("Model trained")
    print(f"Best hyperparameters: {best_params}")
    print(f"MAE on validation: {mae_test}")

    return best_model, selected_features


def get_today_data(current_date: pd.Timestamp):
    data = get_raw_data(current_date=current_date + pd.DateOffset(1))
    return data.iloc[-1]["balance"]


def get_today_(current_date: pd.Timestamp):
    data = get_raw_data(current_date=current_date + pd.DateOffset(1))
    return data.iloc[-1]


def run_full_pipeline(
    current_date: pd.Timestamp,
    min_samples_for_training: int = 50,
    min_days_after_change_point: int = 50,
    use_tsfresh: bool = False,
):
    """
    Это основная функция, которую будем вызывать для каждой даты в тестовом промежутке.
    (При смене дня в предыдущий подкладывается реальный показатель, поскольку день закончен и баланс известен)
    Назначение функции:
    1. детектить разладку (находить дату последнего изменения)
    2. брать данные с последней разладки
    3. на этих данных прогонять model_pipeline
    4. делать предикт на сегодня
    5. отписывать логи, насколько мы ошиблись, сколько денег заработали / потеряли и тд.
    """
    last_changepoint = detect_change_point(current_date)

    days_after_cp = (current_date - pd.Timestamp(last_changepoint)).days
    if days_after_cp < min_days_after_change_point:
        print(
            f"Not enough samples to train model after change point ({days_after_cp}/{min_days_after_change_point}). Use manual model."
        )

    print('-' * 50)
    model, _ = train_model(
        start_date=last_changepoint,
        current_date=current_date,
        min_samples_for_training=min_samples_for_training,
        use_tsfresh=use_tsfresh,
    )
    if model is None:
        return
    print("-" * 50)

    today_observation = get_today_(current_date=current_date)

    if use_tsfresh:
        today_observation = build_features(pd.DataFrame(today_observation).T)

    prediction = run_model_validation(model=model, sample=today_observation)
    print(f"Prediction: {prediction}")

    real_balance = get_today_data(current_date)
    today_metric = calculate_add_margin(
        prediction=prediction, target=real_balance, cbr_key_rate=today_observation["key_rate"]
    )
    print(f"date: {current_date}, add margin: {today_metric}")
