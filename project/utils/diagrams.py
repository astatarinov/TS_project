import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def plot_time_series(df: pd.DataFrame, col_name: str, figsize=(12, 8)):

    plt.figure(figsize=figsize)

    plt.title('Balance Time Series')
    plt.xlabel('date')
    plt.ylabel(col_name)
    
    plt.plot(df[col_name])
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.grid()
    plt.show();
