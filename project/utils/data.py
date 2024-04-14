"""
Model utilities
"""
import os

import pandas as pd

from project.config import DATA_PARAMS, DATA_PATH


def load_balances_data() -> pd.DataFrame:
    """
    Return processed raw data as DataFrame
    """
    data = pd.read_csv(
        DATA_PATH / 'balances.csv',
        parse_dates=['date'],
    ).set_index('date')
    return data


def create_lag_features(df, column_name, lag_levels):
    for lag in lag_levels:
        new_column_name = f'L{lag}_{column_name}'
        df[new_column_name] = df[column_name].shift(lag)

    return df


def create_calendar_features(df: pd.DataFrame):
    """
    Creates basic calendar features
    """
    assert isinstance(df.index, pd.core.indexes.datetimes.DatetimeIndex), "Index must be timestamp"

    # Create dummy columns for weekdays  
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day in weekdays:
        df[day] = (df.index.day_name() == day).astype(int)

        # Create dummy columns for start and end of month
    df['start_of_month'] = df.index.is_month_start.astype(int)
    df['end_of_month'] = df.index.is_month_end.astype(int)

    # Create dummy columns for years  
    # unique_years = df.index.year.unique()  
    unique_years = [2017, 2018, 2019, 2020, 2021]
    for year in unique_years:
        df[f'year_{year}'] = (df.index.year == year).astype(int)

    # Create dummy columns for months
    months = [
        'January', 'February', 'March', 'April', 'May', 'June', 'July',
        'August', 'September', 'October', 'November', 'December'
    ]
    for month in months:
        df[month] = (df.index.month_name() == month).astype(int)

    return df


def add_rolling_features(df: pd.DataFrame, column_name: str, days_num=30):
    """
    Adds rolling window metrics for <column_name> series by last <days_num> days
    """
    df['{}_mean_{}'.format(column_name, days_num)] = df[column_name].rolling(window=days_num).mean()
    df['{}_std_{}'.format(column_name, days_num)] = df[column_name].rolling(window=days_num).std()
    df['{}_min_{}'.format(column_name, days_num)] = df[column_name].rolling(window=days_num).min()
    df['{}_max_{}'.format(column_name, days_num)] = df[column_name].rolling(window=days_num).max()
    df['{}_median_{}'.format(column_name, days_num)] = df[column_name].rolling(window=days_num).median()

    return df


# put create_rolling & rolling_period to config?
def add_features_to_balances(
    balances: pd.DataFrame,
    create_rolling=DATA_PARAMS['calc_rolling_metrics'],
    rolling_periods=DATA_PARAMS['rolling_period'],
) -> pd.DataFrame:
    """
    Return balances data with added 
    1) calendar features:
        - weekdays
        - months
        - years
        - holidays (Russian calendar)
        - tax days
    2) CBR rate
    """
    balances['income - outcome'] = balances['income'] - balances['outcome']

    # calendar of russian holidays and weekends 
    custom_calendar = (
        pd.read_csv(os.path.join(DATA_PATH, 'custom_calendar.csv'))
        .rename(columns={'reporting_date': 'date'})
    )
    custom_calendar['date'] = custom_calendar['date'].apply(pd.Timestamp)
    custom_calendar.set_index('date', inplace=True)

    # cbr rate
    cbr_rate = pd.read_csv(os.path.join(DATA_PATH, 'cbr_rate.csv'))
    cbr_rate['date'] = cbr_rate['date'].apply(pd.Timestamp)
    cbr_rate.set_index('date', inplace=True)

    # MOSPRIME 
    mosprime = pd.read_csv(DATA_PATH / 'mosprime.csv')
    mosprime['date'] = mosprime['date'].apply(pd.Timestamp)
    mosprime.set_index('date', inplace=True)

    # USD exchage rate
    usd_xr = pd.read_csv(DATA_PATH / 'usd_xr.csv')
    usd_xr['date'] = usd_xr['date'].apply(pd.Timestamp)
    usd_xr.set_index('date', inplace=True)

    res_df = (
        balances
        .merge(custom_calendar, left_index=True, right_index=True, how='left')
        .merge(cbr_rate, left_index=True, right_index=True, how='left')
        .merge(mosprime, left_index=True, right_index=True, how='left')
        .merge(usd_xr, left_index=True, right_index=True, how='left')
    )

    # rolling features 
    for column_name in create_rolling:
        if column_name in res_df.columns:
            for period in rolling_periods[column_name]:
                res_df = add_rolling_features(res_df, column_name, period)
        else:
            print("there if no", column_name)

    res_df['day_before_holiday'] = res_df['isholiday'].shift().fillna(0)
    res_df['day_after_holidays'] = (
        (res_df['isholiday'].shift().fillna(0) == 1) & (res_df['isholiday'] == 0)
    ).astype(int)

    # tax day
    res_df['tax'] = (res_df.index.day == 28).astype(int)
    res_df['day_before_tax'] = (res_df.index.day == 27).astype(int)

    # add basic calendar features by index and return 
    return create_calendar_features(res_df)


def overwrite_extended_df():
    """
    Writes extended data to csv
    (Adds 'income - outcome' and features from add_features_to_balances()) 
    """
    balances = load_balances_data()
    df = add_features_to_balances(balances).reset_index(drop=False)
    path = os.path.join(DATA_PATH, 'extended_data.csv')
    df.to_csv(path, index=False)
    print(path, 'is overwritten')


def load_extended_data() -> pd.DataFrame:
    """
    Return ultimate dataset
    """
    data = pd.read_csv(
        DATA_PATH / 'extended_data.csv',
        parse_dates=['date'],
    ).set_index('date').sort_index().dropna()
    return data


def load_tsfresh_data() -> pd.DataFrame:
    """
    Return tsfresh features dataframe
    """
    data = pd.read_csv(
        DATA_PATH / 'tsfresh_features.csv',
        parse_dates=['date'],
    ).set_index('date')
    return data


def load_target_data() -> pd.Series:
    """
    Return target values
    """
    data = pd.read_csv(
        DATA_PATH / 'target.csv',
        parse_dates=['date'],
    ).set_index('date')
    return data
