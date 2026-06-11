from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "global_expense_model.pkl"

DEFAULT_MODEL_VERSION = "rf_global_finexa_user_category_v5_log_target"

FEATURE_COLUMNS = [
    "lag1",
    "lag2",
    "lag3",
    "rolling_mean",
    "month_num",
    "category_code",
]

MAX_MOM_CHANGE = 0.25
SMOOTHING_LAMBDA = 0.6

AI_CATEGORY_ALIASES: Dict[str, str] = {
    # Food group
    "food": "Food",
    "foodd": "Food",
    "foods": "Food",
    "fod": "Food",
    "drink": "Food",
    "drinks": "Food",
    "coffee": "Food",
    "cafe": "Food",
    "tea": "Food",
    "juice": "Food",
    "grocery": "Food",
    "groceries": "Food",
    "supermarket": "Food",
    "market": "Food",
    "restaurant": "Food",
    "restaurants": "Food",
    "dining": "Food",
    "meal": "Food",
    "meals": "Food",

    # Transport group
    "transport": "Transport",
    "transportation": "Transport",
    "uber": "Transport",
    "taxi": "Transport",
    "fuel": "Transport",
    "gasoline": "Transport",
    "bus": "Transport",
    "metro": "Transport",
    "travel": "Transport",
    "traval": "Transport",
    "travl": "Transport",
    "trip": "Transport",
    "flight": "Transport",
    "hotel": "Transport",

    # Shopping group
    "shopping": "Shopping",
    "shop": "Shopping",
    "clothes": "Shopping",
    "fashion": "Shopping",
    "accessories": "Shopping",
    "electronics": "Shopping",
    "electronic": "Shopping",
    "devices": "Shopping",
    "device": "Shopping",
    "mobile": "Shopping",
    "phone device": "Shopping",
    "laptop": "Shopping",
    "computer": "Shopping",

    # Bills group
    "bill": "Bills",
    "bills": "Bills",
    "utility": "Bills",
    "utilities": "Bills",
    "utilties": "Bills",
    "utlities": "Bills",
    "electricity": "Bills",
    "water": "Bills",
    "gas": "Bills",
    "internet": "Bills",
    "phone": "Bills",
    "rent": "Bills",
    "rentt": "Bills",
    "rnt": "Bills",
    "housing": "Bills",
    "house": "Bills",
    "home": "Bills",
    "subscription": "Bills",
    "subscriptions": "Bills",
    "netflix": "Bills",
    "spotify": "Bills",
    "youtube": "Bills",
    "software": "Bills",
    "saas": "Bills",
    "gym": "Bills",
    "fitness": "Bills",
    "receipt": "Bills",
    "receipts": "Bills",

    # Entertainment
    "entertainment": "Entertainment",
    "entrtnmnt": "Entertainment",
    "fun": "Entertainment",
    "movies": "Entertainment",
    "movie": "Entertainment",
    "games": "Entertainment",
    "gaming": "Entertainment",

    # Health
    "health": "Health",
    "helth": "Health",
    "medical": "Health",
    "medicine": "Health",
    "pharmacy": "Health",
    "doctor": "Health",

    # Education
    "education": "Education",
    "educaton": "Education",
    "school": "Education",
    "course": "Education",
    "courses": "Education",
    "tuition": "Education",
    "university": "Education",

    # Other Expense
    "other": "Other Expense",
    "others": "Other Expense",
    "other expense": "Other Expense",
    "misc": "Other Expense",
    "miscellaneous": "Other Expense",
    "unknown": "Other Expense",
}

EXCLUDED_AI_CATEGORIES = {
    "saving",
    "savings",
    "goal",
    "goals",
    "balance adjustment",
}

CATEGORY_GROUPING = {
    "Drinks": "Food",
    "Groceries": "Food",
    "Electronics": "Shopping",
    "Subscriptions": "Bills",
    "Gym": "Bills",
    "Receipt": "Bills",
    "Rent": "Bills",
    "Travel": "Transport",
    "Other": "Other Expense",
    "Others": "Other Expense",
}


def _load_model_bundle() -> Dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Forecast model file not found: {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict):
        raise RuntimeError("Invalid model bundle format. Expected dictionary.")

    required_keys = {"model", "category_mapping"}
    missing = required_keys - set(bundle.keys())

    if missing:
        raise RuntimeError(f"Model bundle missing keys: {sorted(missing)}")

    return bundle


_MODEL_BUNDLE = _load_model_bundle()
_MODEL = _MODEL_BUNDLE["model"]
_CATEGORY_MAPPING: Dict[str, int] = _MODEL_BUNDLE["category_mapping"]
_MODEL_FEATURES: List[str] = _MODEL_BUNDLE.get("feature_names", FEATURE_COLUMNS)
MODEL_VERSION: str = _MODEL_BUNDLE.get("model_version", DEFAULT_MODEL_VERSION)


def normalize_ai_category(category: str) -> Optional[str]:
    raw = str(category).strip()

    if not raw:
        return "Other Expense"

    lower = raw.lower().strip()

    if lower in EXCLUDED_AI_CATEGORIES:
        return None

    if lower in AI_CATEGORY_ALIASES:
        return AI_CATEGORY_ALIASES[lower]

    for trained_category in _CATEGORY_MAPPING.keys():
        if trained_category.lower() == lower:
            return trained_category

    return "Other Expense"


def _normalize_month(value: Any) -> Optional[str]:
    try:
        parsed = pd.to_datetime(value, errors="coerce")

        if pd.isna(parsed):
            return None

        return parsed.to_period("M").strftime("%Y-%m")

    except Exception:
        return None


def _clean_points(points: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    clean_points: List[Dict[str, float]] = []

    for point in points:
        month = _normalize_month(point.get("month"))
        amount = pd.to_numeric(point.get("amount", 0.0), errors="coerce")

        if month is None or pd.isna(amount):
            continue

        amount = float(amount)

        if amount < 0:
            continue

        clean_points.append(
            {
                "month": month,
                "amount": amount,
            }
        )

    return sorted(clean_points, key=lambda x: x["month"])


def normalize_forecast_series(
    series: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, float]]]:
    """
    Internal normalization for the ML model.

    The model sees canonical categories:
    Drinks + Groceries + Food => Food
    Electronics + Shopping => Shopping
    Subscriptions + Gym + Receipt + Rent + Bills => Bills

    But the public API response will still return the original request names.
    """

    normalized: Dict[str, Dict[str, float]] = {}

    for original_category, points in series.items():
        mapped_category = normalize_ai_category(original_category)

        if mapped_category is None:
            continue

        if mapped_category not in _CATEGORY_MAPPING:
            if "Other Expense" in _CATEGORY_MAPPING:
                mapped_category = "Other Expense"
            else:
                continue

        clean_points = _clean_points(points)

        if not clean_points:
            continue

        if mapped_category not in normalized:
            normalized[mapped_category] = {}

        for point in clean_points:
            month = point["month"]
            amount = float(point["amount"])

            normalized[mapped_category][month] = (
                normalized[mapped_category].get(month, 0.0) + amount
            )

    result: Dict[str, List[Dict[str, float]]] = {}

    for category, month_values in normalized.items():
        rows = [
            {
                "month": month,
                "amount": round(float(amount), 2),
            }
            for month, amount in sorted(month_values.items())
        ]

        if rows:
            result[category] = rows

    return result


def build_original_response_allocation(
    series: Dict[str, List[Dict[str, Any]]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str]]:
    """
    Builds allocation weights to return the same category names sent in request.

    Example:
    Request has Food + Drinks.
    Internally both are forecasted as Food.
    Response returns Food and Drinks by splitting the canonical Food forecast
    based on each category's recent contribution.
    """

    totals_by_canonical: Dict[str, Dict[str, float]] = {}
    original_to_canonical: Dict[str, str] = {}

    for original_category, points in series.items():
        mapped_category = normalize_ai_category(original_category)

        if mapped_category is None:
            continue

        if mapped_category not in _CATEGORY_MAPPING:
            if "Other Expense" in _CATEGORY_MAPPING:
                mapped_category = "Other Expense"
            else:
                continue

        clean_points = _clean_points(points)

        if not clean_points:
            continue

        recent_points = clean_points[-3:]
        recent_total = sum(float(point["amount"]) for point in recent_points)

        original_to_canonical[original_category] = mapped_category

        if mapped_category not in totals_by_canonical:
            totals_by_canonical[mapped_category] = {}

        totals_by_canonical[mapped_category][original_category] = recent_total

    allocation: Dict[str, List[Dict[str, Any]]] = {}

    for canonical_category, original_totals in totals_by_canonical.items():
        total = sum(original_totals.values())
        original_names = list(original_totals.keys())

        allocation[canonical_category] = []

        if total > 0:
            for original_name, value in original_totals.items():
                allocation[canonical_category].append(
                    {
                        "response_category": original_name,
                        "weight": float(value / total),
                    }
                )
        else:
            equal_weight = 1.0 / len(original_names)

            for original_name in original_names:
                allocation[canonical_category].append(
                    {
                        "response_category": original_name,
                        "weight": equal_weight,
                    }
                )

    return allocation, original_to_canonical


def _next_months(last_month: str, horizon: int) -> List[str]:
    start = pd.Period(last_month, freq="M")
    return [str(start + i) for i in range(1, horizon + 1)]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _build_feature_row(
    values: List[float],
    forecast_month: str,
    category_code: int,
) -> pd.DataFrame:
    lag1 = float(values[-1])
    lag2 = float(values[-2])
    lag3 = float(values[-3])

    rolling_mean = float(np.mean(values[-3:]))
    month_num = pd.Period(forecast_month, freq="M").month

    return pd.DataFrame(
        [
            {
                "lag1": lag1,
                "lag2": lag2,
                "lag3": lag3,
                "rolling_mean": rolling_mean,
                "month_num": month_num,
                "category_code": category_code,
            }
        ],
        columns=_MODEL_FEATURES,
    )


def _apply_prediction_guardrails(raw_pred: float, values: List[float]) -> float:
    raw_pred = max(float(raw_pred), 0.0)

    last_value = float(values[-1])
    mean3 = float(np.mean(values[-3:]))
    mean6 = float(np.mean(values[-6:])) if len(values) >= 6 else mean3

    lo_mom = last_value * (1.0 - MAX_MOM_CHANGE)
    hi_mom = last_value * (1.0 + MAX_MOM_CHANGE)

    recent_values = values[-6:] if len(values) >= 6 else values
    volatility = float(np.std(recent_values))

    band = max(0.15 * mean6, 1.5 * volatility)

    lo_mean = mean6 - band
    hi_mean = mean6 + band

    lo = max(0.0, lo_mom, lo_mean)
    hi = max(lo + 1e-6, hi_mom, hi_mean)

    clamped = _clamp(raw_pred, lo, hi)

    final_pred = (SMOOTHING_LAMBDA * last_value) + ((1.0 - SMOOTHING_LAMBDA) * clamped)

    return max(final_pred, 0.0)


def _forecast_single_category(
    category: str,
    points: List[Dict[str, Any]],
    forecast_horizon: int,
) -> List[Dict[str, float]]:
    if category not in _CATEGORY_MAPPING:
        return []

    clean_points = _clean_points(points)

    if len(clean_points) < 3:
        return []

    values = [float(point["amount"]) for point in clean_points]
    last_month = clean_points[-1]["month"]
    future_months = _next_months(last_month, forecast_horizon)

    category_code = _CATEGORY_MAPPING[category]

    predictions: List[Dict[str, float]] = []
    temp_values = values.copy()

    for forecast_month in future_months:
        X = _build_feature_row(
            values=temp_values,
            forecast_month=forecast_month,
            category_code=category_code,
        )

        raw_pred = float(_MODEL.predict(X)[0])
        final_pred = _apply_prediction_guardrails(raw_pred, temp_values)

        predictions.append(
            {
                "month": forecast_month,
                "amount": round(final_pred, 2),
            }
        )

        temp_values.append(final_pred)

    return predictions


def _convert_canonical_forecast_to_original_response(
    canonical_forecast: Dict[str, List[Dict[str, float]]],
    allocation: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, float]]]:
    """
    Converts forecast keys from canonical model categories back to original request category names.
    """

    response_forecast: Dict[str, List[Dict[str, float]]] = {}

    for canonical_category, forecast_points in canonical_forecast.items():
        category_allocations = allocation.get(canonical_category)

        if not category_allocations:
            response_forecast[canonical_category] = forecast_points
            continue

        for allocation_item in category_allocations:
            response_category = allocation_item["response_category"]
            weight = float(allocation_item["weight"])

            response_forecast[response_category] = [
                {
                    "month": point["month"],
                    "amount": round(float(point["amount"]) * weight, 2),
                }
                for point in forecast_points
            ]

    return response_forecast


def _build_total_forecast(
    forecast: Dict[str, List[Dict[str, float]]],
) -> List[Dict[str, float]]:
    total_by_month: Dict[str, float] = {}

    for category_points in forecast.values():
        for point in category_points:
            month = point["month"]
            amount = float(point["amount"])

            total_by_month[month] = total_by_month.get(month, 0.0) + amount

    return [
        {
            "month": month,
            "amount": round(amount, 2),
        }
        for month, amount in sorted(total_by_month.items())
    ]


def run_forecast(
    series: Dict[str, List[Dict[str, Any]]],
    forecast_horizon: int = 3,
) -> Dict[str, Any]:
    """
    Main Forecast API function.

    The ML model internally uses canonical categories.
    The API response returns the same category names sent in the request.
    """

    forecast_horizon = int(forecast_horizon)

    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon must be greater than zero.")

    if forecast_horizon > 12:
        raise ValueError("forecast_horizon must not exceed 12 months.")

    normalized_series = normalize_forecast_series(series)
    allocation, original_to_canonical = build_original_response_allocation(series)

    if not normalized_series:
        return {
            "forecast": {},
            "total_forecast": [],
            "model_version": MODEL_VERSION,
            "meta": {
                "message": "No valid expense categories found after normalization.",
                "category_grouping": CATEGORY_GROUPING,
            },
        }

    canonical_forecast: Dict[str, List[Dict[str, float]]] = {}

    for canonical_category, points in normalized_series.items():
        category_forecast = _forecast_single_category(
            category=canonical_category,
            points=points,
            forecast_horizon=forecast_horizon,
        )

        if category_forecast:
            canonical_forecast[canonical_category] = category_forecast

    if not canonical_forecast:
        return {
            "forecast": {},
            "total_forecast": [],
            "model_version": MODEL_VERSION,
            "meta": {
                "message": "Not enough historical data. Each category group needs at least 3 monthly points.",
                "normalized_categories": list(normalized_series.keys()),
                "category_grouping": CATEGORY_GROUPING,
            },
        }

    response_forecast = _convert_canonical_forecast_to_original_response(
        canonical_forecast=canonical_forecast,
        allocation=allocation,
    )

    total_forecast = _build_total_forecast(response_forecast)

    return {
        "forecast": response_forecast,
        "total_forecast": total_forecast,
        "model_version": MODEL_VERSION,
        "meta": {
            "response_category_mode": "same_as_request",
            "normalized_categories": list(normalized_series.keys()),
            "original_to_canonical": original_to_canonical,
            "category_grouping": CATEGORY_GROUPING,
            "split_strategy": "canonical forecast split back to original request categories using recent 3-month spending contribution",
            "guardrails": {
                "max_mom_change": MAX_MOM_CHANGE,
                "smoothing_lambda": SMOOTHING_LAMBDA,
            },
        },
    }