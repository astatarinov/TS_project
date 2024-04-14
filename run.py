"""
Run model
"""

import datetime
import time

import schedule

from project.pipeline import detect_change_point, run_model_pipeline

if __name__ == "__main__":
    print(f'Start serving at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    # schedule.every().second.do(run_model_pipeline)  # for testing
    schedule.every().day.at('07:00').do(run_model_pipeline)
    schedule.every().hour.at(':00').do(detect_change_point)

    while True:
        schedule.run_pending()
        time.sleep(1)
