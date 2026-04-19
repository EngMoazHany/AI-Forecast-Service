from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "income",
    "goal_amount",
    "months",
    "forecast_horizon",
    "required_monthly_saving",
    "hist_avg_expense",
    "hist_std_expense",
    "hist_min_expense",
    "hist_max_expense",
    "hist_trend",
    "forecast_avg_expense",
    "forecast_std_expense",
    "forecast_min_expense",
    "forecast_max_expense",
    "forecast_trend",
    "hist_avg_free_cash",
    "forecast_avg_free_cash",
    "safe_monthly_saving_capacity",
    "expense_to_income_ratio",
    "free_cash_to_income_ratio",
    "goal_pressure_ratio",
    "volatility_ratio",
    "top1_category_share",
    "top3_category_share",
    "fixed_ratio",
    "flexible_ratio",
    "category_count",
    "avg_category_spend",
    "max_category_spend",
    "months_of_history",
]

FIXED_CATEGORY_KEYWORDS = {
    "bills",
    "bill",
    "rent",
    "utilities",
    "electricity",
    "water",
    "gas",
    "insurance",
    "health",
    "education",
    "school",
    "tuition",
}

FLEXIBLE_CATEGORY_KEYWORDS = {
    "food",
    "shopping",
    "entertainment",
    "transport",
    "travel",
    "dining",
    "restaurant",
    "coffee",
    "lifestyle",
    "other",
    "others",
    "misc",
}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_div(a: float, b: float) -> float:
    if abs(b) < 1e-9:
        return 0.0
    return float(a / b)


def _linear_slope(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def aggregate_monthly_totals(series: Dict[str, List[dict]]) -> List[dict]:
    monthly_totals: Dict[str, float] = {}

    for _, points in series.items():
        for point in points:
            month = str(point.get("month", "")).strip()
            if not month:
                continue
            amount = _to_float(point.get("amount"))
            monthly_totals[month] = monthly_totals.get(month, 0.0) + amount

    result = [
        {"month": month, "amount": round(amount, 2)}
        for month, amount in sorted(monthly_totals.items(), key=lambda x: x[0])
    ]
    return result


def extract_average_forecast_by_category(forecast_result: Dict[str, Any]) -> Dict[str, float]:
    forecast_by_cat = forecast_result.get("forecast", {}) or {}
    avg_by_category: Dict[str, float] = {}

    for category, arr in forecast_by_cat.items():
        amounts = [_to_float(item.get("amount")) for item in arr]
        if amounts:
            avg_by_category[category] = float(np.mean(amounts))
        else:
            avg_by_category[category] = 0.0

    return avg_by_category


def compute_safe_monthly_capacity(
    free_cash: float,
    save_ratio: float = 0.70,
    buffer_ratio: float = 0.20
) -> float:
    free_cash = _to_float(free_cash)

    if free_cash <= 0:
        return 0.0

    by_save_ratio = free_cash * save_ratio
    by_buffer = free_cash * (1.0 - buffer_ratio)

    return round(max(0.0, min(by_save_ratio, by_buffer)), 2)


def build_goal_strategy(
    goal_amount: float,
    months: int,
    recommended_monthly_saving: float
) -> dict | None:
    goal_amount = _to_float(goal_amount)
    recommended_monthly_saving = _to_float(recommended_monthly_saving)

    if months <= 0 or recommended_monthly_saving <= 0:
        return None

    max_goal = recommended_monthly_saving * months
    recommended_timeline_months = int(math.ceil(goal_amount / recommended_monthly_saving))

    return {
        "max_possible_goal_in_timeframe": round(max_goal, 2),
        "recommended_timeline_months": recommended_timeline_months,
        "recommended_monthly_saving": round(recommended_monthly_saving, 2),
    }


def build_monthly_plan(
    total_forecast: List[dict],
    income: float,
    recommended_monthly_saving: float,
    months: int
) -> List[dict]:
    income = _to_float(income)
    recommended_monthly_saving = _to_float(recommended_monthly_saving)

    plan: List[dict] = []

    for item in total_forecast[:months]:
        month = str(item.get("month"))
        expense = _to_float(item.get("amount"))
        free_cash = income - expense
        month_capacity = compute_safe_monthly_capacity(free_cash)
        save_amount = max(0.0, min(recommended_monthly_saving, month_capacity))

        plan.append(
            {
                "month": month,
                "save": round(save_amount, 2),
                "expected_free_cash": round(free_cash, 2),
            }
        )

    return plan


def build_planning_features(
    series: Dict[str, List[dict]],
    forecast_result: Dict[str, Any],
    income: float,
    goal_amount: float,
    months: int,
    forecast_horizon: int
) -> pd.DataFrame:
    income = _to_float(income)
    goal_amount = _to_float(goal_amount)
    months = int(months)
    forecast_horizon = int(forecast_horizon)

    historical_totals = aggregate_monthly_totals(series)
    historical_values = [_to_float(item["amount"]) for item in historical_totals]

    total_forecast = forecast_result.get("total_forecast", []) or []
    planning_window = total_forecast[:months] if months > 0 else total_forecast
    forecast_values = [_to_float(item.get("amount")) for item in planning_window]

    if not forecast_values and total_forecast:
        forecast_values = [_to_float(item.get("amount")) for item in total_forecast]

    avg_by_category = extract_average_forecast_by_category(forecast_result)

    hist_avg = float(np.mean(historical_values)) if historical_values else 0.0
    hist_std = float(np.std(historical_values)) if historical_values else 0.0
    hist_min = float(np.min(historical_values)) if historical_values else 0.0
    hist_max = float(np.max(historical_values)) if historical_values else 0.0
    hist_trend = _linear_slope(historical_values)

    forecast_avg = float(np.mean(forecast_values)) if forecast_values else 0.0
    forecast_std = float(np.std(forecast_values)) if forecast_values else 0.0
    forecast_min = float(np.min(forecast_values)) if forecast_values else 0.0
    forecast_max = float(np.max(forecast_values)) if forecast_values else 0.0
    forecast_trend = _linear_slope(forecast_values)

    required_monthly_saving = goal_amount / months if months > 0 else 0.0
    hist_avg_free_cash = income - hist_avg
    forecast_avg_free_cash = income - forecast_avg
    safe_capacity = compute_safe_monthly_capacity(forecast_avg_free_cash)

    total_avg_category_spend = sum(avg_by_category.values())
    category_count = len(avg_by_category)
    avg_category_spend = _safe_div(total_avg_category_spend, category_count)
    max_category_spend = max(avg_by_category.values()) if avg_by_category else 0.0

    sorted_amounts = sorted(avg_by_category.values(), reverse=True)
    top1_share = _safe_div(sorted_amounts[0], total_avg_category_spend) if sorted_amounts else 0.0
    top3_share = _safe_div(sum(sorted_amounts[:3]), total_avg_category_spend) if sorted_amounts else 0.0

    fixed_sum = 0.0
    flexible_sum = 0.0

    for category, amount in avg_by_category.items():
        name = str(category).strip().lower()

        if any(word in name for word in FIXED_CATEGORY_KEYWORDS):
            fixed_sum += amount

        if any(word in name for word in FLEXIBLE_CATEGORY_KEYWORDS):
            flexible_sum += amount

    fixed_ratio = _safe_div(fixed_sum, total_avg_category_spend)
    flexible_ratio = _safe_div(flexible_sum, total_avg_category_spend)

    feature_dict = {
        "income": income,
        "goal_amount": goal_amount,
        "months": float(months),
        "forecast_horizon": float(forecast_horizon),
        "required_monthly_saving": required_monthly_saving,
        "hist_avg_expense": hist_avg,
        "hist_std_expense": hist_std,
        "hist_min_expense": hist_min,
        "hist_max_expense": hist_max,
        "hist_trend": hist_trend,
        "forecast_avg_expense": forecast_avg,
        "forecast_std_expense": forecast_std,
        "forecast_min_expense": forecast_min,
        "forecast_max_expense": forecast_max,
        "forecast_trend": forecast_trend,
        "hist_avg_free_cash": hist_avg_free_cash,
        "forecast_avg_free_cash": forecast_avg_free_cash,
        "safe_monthly_saving_capacity": safe_capacity,
        "expense_to_income_ratio": _safe_div(forecast_avg, income),
        "free_cash_to_income_ratio": _safe_div(forecast_avg_free_cash, income),
        "goal_pressure_ratio": _safe_div(required_monthly_saving, income),
        "volatility_ratio": _safe_div(forecast_std, forecast_avg),
        "top1_category_share": top1_share,
        "top3_category_share": top3_share,
        "fixed_ratio": fixed_ratio,
        "flexible_ratio": flexible_ratio,
        "category_count": float(category_count),
        "avg_category_spend": avg_category_spend,
        "max_category_spend": max_category_spend,
        "months_of_history": float(len(historical_values)),
    }

    row = {
        col: float(feature_dict.get(col, 0.0))
        for col in FEATURE_COLUMNS
    }

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)