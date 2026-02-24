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

MODEL_VERSION = bundle.get("model_version", "rf_global_v1")

ALPHA = 0.5  # Personal weight (0..1)


def _next_months(last_month: str, horizon: int):
    last = datetime.strptime(last_month, "%Y-%m").replace(day=1)
    return [
        (last + pd.DateOffset(months=i)).strftime("%Y-%m")
        for i in range(1, horizon + 1)
    ]


def _get_month(p: Any) -> str:
    # dict
    if isinstance(p, dict):
        return p["month"]
    # pydantic object or any object with attribute
    return getattr(p, "month")


def _get_amount(p: Any) -> float:
    if isinstance(p, dict):
        return float(p["amount"])
    return float(getattr(p, "amount"))


def run_forecast(series_dict: Dict[str, List[Any]], horizon: int):

    forecast_out = {}
    total = None

    for category, data in (series_dict or {}).items():

        if category not in category_mapping:
            raise ValueError(f"Unknown category: {category}")

        if not data or len(data) < 3:
            continue

        data_sorted = sorted(data, key=_get_month)
        values = [_get_amount(p) for p in data_sorted]

        months = _next_months(_get_month(data_sorted[-1]), horizon)
        category_code = category_mapping[category]

        preds = []
        temp_values = values.copy()

        global_mean = np.mean(values[-6:]) if len(values) >= 6 else np.mean(values)

        for i in range(horizon):
            lag1 = temp_values[-1]
            lag2 = temp_values[-2]
            lag3 = temp_values[-3]
            rolling_mean = float(np.mean(temp_values[-3:]))
            month_num = datetime.strptime(months[i], "%Y-%m").month

            X = np.array([[lag1, lag2, lag3, rolling_mean, month_num, category_code]], dtype=float)

            global_pred = float(model.predict(X)[0])

            # Personal Adjustment
            personal_mean = float(np.mean(temp_values[-3:]))
            personal_bias = personal_mean - float(global_mean)

            final_pred = global_pred + (ALPHA * personal_bias)
            final_pred = max(final_pred, 0.0)

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
    if total and forecast_out:
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