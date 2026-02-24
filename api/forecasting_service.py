import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any

# -------------------------
# SAFE MODEL LOADING
# -------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "global_expense_model.pkl")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
category_mapping = bundle["category_mapping"]

MODEL_VERSION = "global_ml_hybrid_v1"
ALPHA = 0.5  # Personal weight


def _next_months(last_month: str, horizon: int):
    last = datetime.strptime(last_month, "%Y-%m").replace(day=1)
    return [
        (last + pd.DateOffset(months=i)).strftime("%Y-%m")
        for i in range(1, horizon + 1)
    ]


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

        preds = []
        temp_values = values.copy()

        global_mean = np.mean(values[-6:]) if len(values) >= 6 else np.mean(values)

        for i in range(horizon):

            lag1 = temp_values[-1]
            lag2 = temp_values[-2]
            lag3 = temp_values[-3]
            rolling_mean = np.mean(temp_values[-3:])
            month_num = datetime.strptime(months[i], "%Y-%m").month

            X = np.array([[lag1, lag2, lag3, rolling_mean, month_num, category_code]])

            global_pred = model.predict(X)[0]

            # Personal Adjustment
            personal_mean = np.mean(temp_values[-3:])
            personal_bias = personal_mean - global_mean

            final_pred = global_pred + ALPHA * personal_bias
            final_pred = max(final_pred, 0)

            preds.append(final_pred)
            temp_values.append(final_pred)

        forecast_out[category] = [
            {"month": months[i], "amount": round(preds[i], 2)}
            for i in range(horizon)
        ]

        if total is None:
            total = preds.copy()
        else:
            total = [total[i] + preds[i] for i in range(horizon)]

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