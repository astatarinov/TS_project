"""
Model prod pipeline
"""
import pandas as pd

from project.utils.data import load_extended_data


def detect_change_point():
    """
    Detect data drifts so trained model is unusable
    """
    print('Starting change point detector')

    change_point_flag = True  # todo: run kliep

    if not change_point_flag:
        print('No change point detected, its OK to use predictions')
        return

    print('Change point detected, use the model carefully')
    print('Rerun model training...')

    run_model_pipeline()


def get_raw_data(current_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Load raw data
    """
    current_date = current_date or pd.Timestamp.now()
    raw_data = load_extended_data()
    return raw_data.loc[:current_date]


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


def run_model_pipeline(current_date: pd.Timestamp = None):
    """
    Main model pipeline
    """
    current_date = current_date or pd.Timestamp.now()

    print('Starting model pipeline')

    print('Loading raw data')
    raw_data = get_raw_data(current_date)
    print(f'Raw data loaded. Shape: {raw_data.shape}')

    print('Building features from raw data')
    features_df = build_features(raw_data)
    print(f'Features dataset built. Shape: {features_df.shape}')

    print('Selecting features for training')
    dataset = select_features(raw_data)
    print(f'Training features selected. Shape: {dataset.shape}')

    print('Splitting dataset into train/val')
    X_train, X_test, y_train, y_test = split_data(dataset)
    print('Dataset split')

    print('Run model training')
    model = run_model_training(X_train, y_train)
    print('Model trained')

    print('Run model validation')
    val_results = run_model_validation(model)
    print('Model validated')

    print(val_results)
