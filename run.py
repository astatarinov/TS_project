"""
Run model
"""

import datetime
import time

import schedule

from project.pipeline import detect_anomalies, run_model_pipeline

if __name__ == "__main__":
    print(f'Start serving at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    schedule.every().day.at("07:00").do(run_model_pipeline)
    schedule.every().hour.at(":00").do(detect_anomalies)

    while True:
        schedule.run_pending()
        time.sleep(1)
