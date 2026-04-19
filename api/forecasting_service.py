from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any

import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "global_expense_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Forecast model file not found: {MODEL_PATH}")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]

if hasattr(model, "n_jobs"):
    model.n_jobs = 1

category_mapping = bundle["category_mapping"]

MODEL_VERSION = bundle.get("model_version", "rf_global_v2")


MAX_MOM_CHANGE = 0.25
SMOOTHING_LAMBDA = 0.6
FALLBACK_CATEGORY = "Others"

CATEGORY_ALIASES = {
    "other": "Others",
    "others": "Others",
    "misc": "Others",
    "miscellaneous": "Others",
}


def _next_months(last_month: str, horizon: int):
    last = datetime.strptime(last_month, "%Y-%m").replace(day=1)
    return [
        (last + pd.DateOffset(months=i)).strftime("%Y-%m")
        for i in range(1, horizon + 1)
    ]


def _get_month(x):
    return x["month"] if isinstance(x, dict) else x.month


def _get_amount(x):
    return float(x["amount"]) if isinstance(x, dict) else float(x.amount)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _resolve_supported_category(category: str) -> str:
    if category in category_mapping:
        return category

    cat = str(category).strip()
    cat_lower = cat.lower()

    for supported in category_mapping.keys():
        if supported.lower() == cat_lower:
            return supported

    alias_target = CATEGORY_ALIASES.get(cat_lower)
    if alias_target and alias_target in category_mapping:
        return alias_target

    if FALLBACK_CATEGORY in category_mapping:
        return FALLBACK_CATEGORY

    raise ValueError(
        f"Unknown category: {category}. Supported categories: {list(category_mapping.keys())}"
    )


def _normalize_and_merge_series(series_dict: Dict[str, List[Any]]) -> Dict[str, List[dict]]:
    merged: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for raw_category, data in (series_dict or {}).items():
        normalized_category = _resolve_supported_category(raw_category)

        for point in data:
            month = _get_month(point)
            amount = _get_amount(point)
            merged[normalized_category][month] += amount

    normalized_output: Dict[str, List[dict]] = {}

    for category, month_map in merged.items():
        normalized_output[category] = [
            {"month": month, "amount": round(amount, 2)}
            for month, amount in sorted(month_map.items(), key=lambda x: x[0])
        ]

    return normalized_output


def run_forecast(series_dict: Dict[str, List[Any]], horizon: int):
    if horizon <= 0:
        raise ValueError("forecast_horizon must be greater than 0.")

    normalized_series = _normalize_and_merge_series(series_dict)

    forecast_out = {}
    total = None

    feature_names = list(
        getattr(
            model,
            "feature_names_in_",
            ["lag1", "lag2", "lag3", "rolling_mean", "month_num", "category_code"]
        )
    )

    for category, data in normalized_series.items():
        if category not in category_mapping:
            raise ValueError(f"Unknown category after normalization: {category}")

        if not data or len(data) < 3:
            continue

        data_sorted = sorted(data, key=_get_month)
        values = [_get_amount(p) for p in data_sorted]

        last_month = _get_month(data_sorted[-1])
        months = _next_months(last_month, horizon)
        category_code = category_mapping[category]

        preds = []
        temp_values = values.copy()

        last_val = float(temp_values[-1])
        mean3 = float(np.mean(temp_values[-3:]))
        mean6 = float(np.mean(temp_values[-6:])) if len(temp_values) >= 6 else mean3

        for i in range(horizon):
            lag1 = float(temp_values[-1])
            lag2 = float(temp_values[-2])
            lag3 = float(temp_values[-3])
            rolling_mean = float(np.mean(temp_values[-3:]))
            month_num = datetime.strptime(months[i], "%Y-%m").month

            X_df = pd.DataFrame([{
                feature_names[0]: lag1,
                feature_names[1]: lag2,
                feature_names[2]: lag3,
                feature_names[3]: rolling_mean,
                feature_names[4]: month_num,
                feature_names[5]: category_code,
            }])

            raw_pred = float(model.predict(X_df)[0])

            lo_mom = last_val * (1.0 - MAX_MOM_CHANGE)
            hi_mom = last_val * (1.0 + MAX_MOM_CHANGE)

            volatility = (
                float(np.std(temp_values[-6:]))
                if len(temp_values) >= 6
                else float(np.std(temp_values))
            )

            band = max(0.15 * mean6, 1.5 * volatility)
            lo_mean = mean6 - band
            hi_mean = mean6 + band

            lo = max(0.0, lo_mom, lo_mean)
            hi = max(lo + 1e-6, hi_mom, hi_mean)

            clamped = _clamp(raw_pred, lo, hi)

            final_pred = (SMOOTHING_LAMBDA * last_val) + ((1.0 - SMOOTHING_LAMBDA) * clamped)
            final_pred = max(final_pred, 0.0)

            preds.append(final_pred)

            temp_values.append(final_pred)
            last_val = final_pred
            mean3 = float(np.mean(temp_values[-3:]))
            mean6 = float(np.mean(temp_values[-6:])) if len(temp_values) >= 6 else mean3

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
        "model_version": MODEL_VERSION,
        "meta": {
            "guardrails": {
                "max_mom_change": MAX_MOM_CHANGE,
                "smoothing_lambda": SMOOTHING_LAMBDA
            },
            "fallback_category": FALLBACK_CATEGORY if FALLBACK_CATEGORY in category_mapping else None,
            "supported_categories": list(category_mapping.keys())
        }
    }