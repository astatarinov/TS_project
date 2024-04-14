"""
Run CatBoost Regressor
"""

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


def catboost_ts_model_fit(
    target,
    features,
    params_grid,
    model_class,
    cv_window="expanding",
    n_splits=5,
    additional_metric=None,
    test_size=0.2,
):
    # Step 1: Split time series data into train and test
    split_index = int(len(target) * (1 - test_size))
    target_train, target_test = target[:split_index], target[split_index:]
    features_train, features_test = features[:split_index], features[split_index:]

    # Step 2: Define Time Series Cross-Validation
    if cv_window == "expanding":
        cv = TimeSeriesSplit(n_splits=n_splits if n_splits else len(target_train))
    elif cv_window == "rolling":
        cv = TimeSeriesSplit(n_splits=n_splits)

    # Step 3: Run Grid Search CV on the train data
    grid_search = GridSearchCV(
        model_class, params_grid, cv=cv, scoring="neg_mean_absolute_error"
    )
    grid_search.fit(features_train, target_train)

    best_model = grid_search.best_estimator_

    # Step 4: Evaluate the model on the test data
    predictions_test = best_model.predict(features_test)
    mae_test = mean_absolute_error(target_test, predictions_test)

    # Step 5: Fit the best model on all data
    best_model.fit(features, target)

    additional_metric_result = None
    if additional_metric:
        additional_metric_result = additional_metric(target_test, predictions_test)

    return best_model, mae_test, additional_metric_result, grid_search.best_params_
