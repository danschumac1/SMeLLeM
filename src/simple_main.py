'''
python ./src/simple_main.py
'''

import json
import os
from utils.data_management import Prompter, TSDataGenerator
from datetime import datetime

def main():
    prompter = Prompter(
        model="gpt-4o-mini",
        temperature=0.7
    )
    data_gen = TSDataGenerator(
        low_thresh=-0.2,
        high_thresh=0.2,
        num_points=100
    )

    start = "2023-01-01 00:00:00"
    end = "2023-01-02 00:00:00"

    data_gen.create_time_series_data(
        slope=0.1,
        intercept=0.5,
        noise=0.1,  # ✅ Renamed from `error` to `noise`
        start=start,
        end=end,
        num_points=20,
        gen_graph=True
    )

    trend = prompter.analyze_trend(data_gen.data_str)
    print(f"Predicted Trend: {trend.direction}")
    print(f"Ground Truth: {data_gen.label}")

if __name__ == '__main__':
    main()