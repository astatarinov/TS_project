"""
Model utilities
"""
import pandas as pd
import os
from project.config import DATA_PATH


def load_balances_data() -> pd.DataFrame:
    """
    Return processed raw data as DataFrame
    """
    data = pd.read_csv(
        DATA_PATH / 'balances.csv',
        parse_dates=['date'],
    ).set_index('date')
    return data


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
    unique_years = df.index.year.unique()  
    for year in unique_years:  
        df[f'year_{year}'] = (df.index.year == year).astype(int)  
  
    # Create dummy columns for months  
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
              'August', 'September', 'October', 'November', 'December']  
    for month in months:  
        df[month] = (df.index.month_name() == month).astype(int)  
  
    return df  


def add_features_to_balances() -> pd.DataFrame:
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

    balances = load_balances_data()
    
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

    res_df = (
        balances
        .merge(custom_calendar, left_index=True, right_index=True, how='left')
        .merge(cbr_rate, left_index=True, right_index=True, how='left')
    )

    res_df['day_before_holiday'] =  res_df['isholiday'].shift().fillna(0)

    # tax day
    res_df['tax'] =  (res_df.index.day == 28).astype(int)
    res_df['day_before_tax'] =  (res_df.index.day == 27).astype(int)

    # add basic calendar features and return 

    return create_calendar_features(res_df)


def overwrite_featured_df() -> pd.DataFrame:

    df = add_features_to_balances().reset_index(drop=False)
    path = os.path.join(DATA_PATH, 'extended_data.csv')
                        
    df.to_csv(path, index=False)

    print(path, "is overwritten")

def load_extended_data() -> pd.DataFrame:

    data = pd.read_csv(
        DATA_PATH / 'extended_data.csv',
        parse_dates=['date'],
    ).set_index('date')
    
    return data