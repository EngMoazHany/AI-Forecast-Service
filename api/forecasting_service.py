import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
from statsmodels.tsa.holtwinters import ExponentialSmoothing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "global_expense_model.pkl")

bundle = joblib.load(MODEL_PATH)
ml_model = bundle["model"]
category_mapping = bundle["category_mapping"]

MODEL_VERSION = "hybrid_pro_v1"

W_TREND = 0.4
W_ML = 0.4
W_PERSONAL = 0.2


def _next_months(last_month: str, horizon: int):
    last = datetime.strptime(last_month, "%Y-%m").replace(day=1)
    return [
        (last + pd.DateOffset(months=i)).strftime("%Y-%m")
        for i in range(1, horizon + 1)
    ]


def _trend_forecast(values, horizon):
    model = ExponentialSmoothing(values, trend='add', seasonal=None)
    fit = model.fit()
    return fit.forecast(horizon)


def run_forecast(series_dict: Dict[str, List[Any]], horizon: int):

    forecast_out = {}
    total = None

    for category, data in series_dict.items():

        if category not in category_mapping:
            raise ValueError(f"Unknown category: {category}")

        data_sorted = sorted(data, key=lambda x: x["month"])
        values = [float(p["amount"]) for p in data_sorted]

        if len(values) < 3:
            continue

        months = _next_months(data_sorted[-1]["month"], horizon)
        category_code = category_mapping[category]

        # 1️⃣ Trend
        trend_preds = _trend_forecast(values, horizon)

        # 2️⃣ ML
        temp_values = values.copy()
        ml_preds = []

        for i in range(horizon):
            lag1 = temp_values[-1]
            lag2 = temp_values[-2]
            lag3 = temp_values[-3]
            rolling_mean = np.mean(temp_values[-3:])
            month_num = datetime.strptime(months[i], "%Y-%m").month

            X = np.array([[lag1, lag2, lag3, rolling_mean, month_num, category_code]])
            pred = ml_model.predict(X)[0]
            ml_preds.append(pred)
            temp_values.append(pred)

        # 3️⃣ Personal bias
        personal_mean = np.mean(values[-3:])
        global_mean = np.mean(values)
        personal_bias = personal_mean - global_mean

        final_preds = []
        for i in range(horizon):
            final = (
                W_TREND * trend_preds[i] +
                W_ML * ml_preds[i] +
                W_PERSONAL * personal_bias
            )
            final = max(final, 0)
            final_preds.append(final)

        forecast_out[category] = [
            {"month": months[i], "amount": round(final_preds[i], 2)}
            for i in range(horizon)
        ]

        if total is None:
            total = final_preds.copy()
        else:
            total = [total[i] + final_preds[i] for i in range(horizon)]

    total_output = []
    if total:
        any_cat = next(iter(forecast_out.values()))
        for i in range(horizon):
            total_output.append({
                "month": any_cat[i]["month"],
                "amount": round(total[i], 2)
            })

    return {
        "forecast": forecast_out,
        "total_forecast": total_output,
        "model_version": MODEL_VERSION
    }