from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from api.schemas import CategorySummary, SavingPlanRequest


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "saving_plan_model.pkl"

DEFAULT_MODEL_VERSION = "saving_plan_advisor_summary_v1"


ESSENTIAL_KEYWORDS = {
    "rent",
    "bill",
    "bills",
    "utilities",
    "electricity",
    "water",
    "gas",
    "health",
    "medical",
    "education",
    "school",
    "tuition",
    "insurance",
    "basic transport",
    "transportation basic",
    "groceries",
    "basic groceries",
}

FLEXIBLE_KEYWORDS = {
    "shopping",
    "entertainment",
    "subscriptions",
    "subscription",
    "coffee",
    "gaming",
    "luxury",
    "delivery",
    "restaurant",
    "restaurants",
    "food outside",
    "dining",
    "accessories",
    "travel",
    "other",
    "others",
}


PLAN_RULES = {
    "Easy": {
        "flexible_base": 0.08,
        "flexible_max": 0.10,
        "essential_base": 0.01,
        "essential_max": 0.03,
        "difficulty": "Low",
    },
    "Balanced": {
        "flexible_base": 0.15,
        "flexible_max": 0.20,
        "essential_base": 0.025,
        "essential_max": 0.05,
        "difficulty": "Medium",
    },
    "Aggressive": {
        "flexible_base": 0.27,
        "flexible_max": 0.35,
        "essential_base": 0.04,
        "essential_max": 0.08,
        "difficulty": "High",
    },
}


MIN_ESSENTIAL_RECOMMENDATION_AMOUNT = 100.0


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _round_money(value: Any) -> float:
    return round(max(0.0, _to_float(value)), 2)


def _mean(values: List[float]) -> float:
    clean = [_to_float(v) for v in values]
    if not clean:
        return 0.0
    return float(np.mean(clean))


def _safe_div(a: float, b: float) -> float:
    if abs(b) < 1e-9:
        return 0.0
    return float(a / b)


def _format_money(value: float, currency: str) -> str:
    return f"{round(value):,} {currency}"


def _linear_slope(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0

    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)

    return float(np.polyfit(x, y, 1)[0])


def _forecast_next_value(values: List[float]) -> float:
    clean = [_to_float(v) for v in values if _to_float(v) >= 0]

    if not clean:
        return 0.0

    if len(clean) == 1:
        return round(clean[0], 2)

    avg_all = _mean(clean)
    last_value = clean[-1]
    recent_values = clean[-min(3, len(clean)):]
    recent_avg = _mean(recent_values)
    trend = _linear_slope(clean[-min(6, len(clean)):])

    raw_forecast = (
        (0.45 * last_value)
        + (0.35 * recent_avg)
        + (0.20 * avg_all)
        + (0.50 * trend)
    )

    if last_value > 0:
        low = last_value * 0.75
        high = last_value * 1.25
        raw_forecast = max(low, min(high, raw_forecast))

    return round(max(0.0, raw_forecast), 2)


@lru_cache(maxsize=1)
def _load_model_bundle() -> Any:
    if not MODEL_PATH.exists():
        return None

    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def get_saving_plan_model_version() -> str:
    bundle = _load_model_bundle()

    if isinstance(bundle, dict):
        return str(bundle.get("model_version", DEFAULT_MODEL_VERSION))

    if bundle is not None:
        return "saving_plan_ai_single_model_v1"

    return DEFAULT_MODEL_VERSION


def _empty_response(
    dto: SavingPlanRequest,
    plan_status: str,
    summary_message: str,
    average_income: float = 0.0,
    average_expenses: float = 0.0,
    current_average_saving: float = 0.0,
    forecasted_income: float = 0.0,
    forecasted_expenses: float = 0.0,
    forecasted_saving: float = 0.0,
    difficulty: str = "Low",
    insights: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    safe_current_saving = max(0.0, current_average_saving)

    return {
        "analysisPeriodMonths": int(dto.months),
        "currency": dto.currency,
        "averageIncome": round(average_income, 2),
        "averageExpenses": round(average_expenses, 2),
        "currentAverageSaving": round(current_average_saving, 2),
        "forecastedIncome": round(forecasted_income, 2),
        "forecastedExpenses": round(forecasted_expenses, 2),
        "forecastedSaving": round(forecasted_saving, 2),
        "recommendedMonthlySaving": round(safe_current_saving, 2),
        "extraSavingOpportunity": 0.0,
        "difficulty": difficulty,
        "planStatus": plan_status,
        "summaryMessage": summary_message,
        "recommendations": [],
        "insights": insights or [],
        "warnings": warnings or [],
    }


def _resolve_category_type(category: CategorySummary) -> str:
    supplied_type = str(category.categoryType or "").strip()

    if supplied_type in {"Essential", "Flexible"}:
        return supplied_type

    name = str(category.categoryName or "").strip().lower()

    if any(keyword in name for keyword in ESSENTIAL_KEYWORDS):
        return "Essential"

    return "Flexible"


def _category_average(category: CategorySummary, months: int) -> float:
    avg = _to_float(category.averageMonthlyAmount)

    if avg > 0:
        return avg

    total = _to_float(category.totalAmount)

    if total > 0 and months > 0:
        return total / months

    return 0.0


def _category_percentage(
    category: CategorySummary,
    category_average: float,
    average_expenses: float,
) -> float:
    percentage = _to_float(category.percentageOfExpenses)

    if percentage > 0:
        return percentage

    return _safe_div(category_average, average_expenses) * 100.0


def _category_reduction_ratio(
    category_type: str,
    plan_type: str,
    trend: str,
    original_type: str,
) -> float:
    rule = PLAN_RULES[plan_type]

    if category_type == "Essential":
        base = rule["essential_base"]
        max_ratio = rule["essential_max"]
    else:
        base = rule["flexible_base"]
        max_ratio = rule["flexible_max"]

    if original_type == "Unknown":
        base *= 0.60

    if category_type == "Flexible":
        if trend == "Increasing":
            base += 0.03
        elif trend == "Decreasing":
            base -= 0.02

    return max(0.0, min(max_ratio, base))


def _build_candidates(
    categories: List[CategorySummary],
    plan_type: str,
    months: int,
    average_expenses: float,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    for category in categories:
        avg_amount = _category_average(category, months)

        if avg_amount <= 0:
            continue

        category_type = _resolve_category_type(category)
        percentage = _category_percentage(category, avg_amount, average_expenses)
        trend = str(category.trend or "Stable")

        reduction_ratio = _category_reduction_ratio(
            category_type=category_type,
            plan_type=plan_type,
            trend=trend,
            original_type=str(category.categoryType or "Unknown"),
        )

        max_saving = max(0.0, min(avg_amount, avg_amount * reduction_ratio))

        if max_saving <= 0:
            continue

        type_weight = 1.0 if category_type == "Flexible" else 0.15

        if trend == "Increasing":
            trend_weight = 1.25
        elif trend == "Decreasing":
            trend_weight = 0.80
        else:
            trend_weight = 1.00

        percentage_weight = 1.0 + min(1.0, percentage / 100.0)
        weight = max(1.0, avg_amount) * type_weight * trend_weight * percentage_weight

        candidates.append(
            {
                "categoryId": category.categoryId,
                "categoryName": category.categoryName,
                "categoryType": category_type,
                "currentAverage": round(avg_amount, 2),
                "percentageOfExpenses": round(percentage, 2),
                "trend": trend,
                "maxSaving": round(max_saving, 2),
                "maxReductionRatio": reduction_ratio,
                "weight": weight,
            }
        )

    candidates.sort(
        key=lambda item: (
            item["categoryType"] == "Flexible",
            item["maxSaving"],
            item["weight"],
        ),
        reverse=True,
    )

    return candidates


def _allocate_reductions(
    candidates: List[Dict[str, Any]],
    target_extra_saving: float,
) -> Dict[str, float]:
    target = max(0.0, _to_float(target_extra_saving))

    if target <= 0 or not candidates:
        return {}

    remaining = target
    active = [candidate for candidate in candidates if candidate["maxSaving"] > 0]
    allocations = {candidate["categoryName"]: 0.0 for candidate in active}

    while remaining > 0.01 and active:
        total_weight = sum(max(0.01, candidate["weight"]) for candidate in active)

        if total_weight <= 0:
            break

        progress = 0.0
        next_active = []

        for candidate in active:
            name = candidate["categoryName"]
            already_allocated = allocations[name]
            available = max(0.0, candidate["maxSaving"] - already_allocated)

            if available <= 0.01:
                continue

            share = remaining * (candidate["weight"] / total_weight)
            amount = min(available, share)

            if amount > 0:
                allocations[name] += amount
                progress += amount

            if candidate["maxSaving"] - allocations[name] > 0.01:
                next_active.append(candidate)

        if progress <= 0.01:
            break

        remaining -= progress
        active = next_active

    return {
        name: round(amount, 2)
        for name, amount in allocations.items()
        if amount > 0.01
    }


def _build_recommendation_reason(candidate: Dict[str, Any]) -> str:
    name = candidate["categoryName"]
    category_type = candidate["categoryType"]
    trend = candidate["trend"]

    if category_type == "Essential":
        return (
            f"{name} is treated as an essential category, so the suggested reduction "
            "is intentionally small and safe."
        )

    if trend == "Increasing":
        return (
            f"{name} is a flexible category and has been increasing recently, "
            "so it is a strong saving opportunity."
        )

    if candidate["percentageOfExpenses"] >= 20:
        return (
            f"{name} represents a high percentage of monthly expenses and can be "
            "reduced moderately without affecting essential needs."
        )

    return f"{name} is a flexible spending category that can be reduced safely."


def _build_recommendations(
    candidates: List[Dict[str, Any]],
    allocations: Dict[str, float],
) -> List[Dict[str, Any]]:
    recommendations: List[Dict[str, Any]] = []

    candidate_by_name = {
        candidate["categoryName"]: candidate
        for candidate in candidates
    }

    for category_name, expected_saving in allocations.items():
        candidate = candidate_by_name.get(category_name)

        if candidate is None:
            continue

        if (
            candidate["categoryType"] == "Essential"
            and expected_saving < MIN_ESSENTIAL_RECOMMENDATION_AMOUNT
        ):
            continue

        current_average = candidate["currentAverage"]
        expected_saving = min(expected_saving, current_average)
        recommended_budget = max(0.0, current_average - expected_saving)
        reduction_percentage = _safe_div(expected_saving, current_average) * 100.0

        recommendations.append(
            {
                "categoryId": candidate["categoryId"],
                "categoryName": candidate["categoryName"],
                "categoryType": candidate["categoryType"],
                "currentAverage": round(current_average, 2),
                "recommendedBudget": round(recommended_budget, 2),
                "reductionPercentage": round(reduction_percentage, 2),
                "expectedSaving": round(expected_saving, 2),
                "reason": _build_recommendation_reason(candidate),
            }
        )

    recommendations.sort(key=lambda item: item["expectedSaving"], reverse=True)

    return recommendations


def _difficulty_level(
    plan_type: str,
    plan_status: str,
    target_extra_saving: float,
    total_possible_saving: float,
) -> str:
    if plan_status in {"Critical", "Unrealistic", "Hard"}:
        return "High"

    if total_possible_saving <= 0:
        return PLAN_RULES[plan_type]["difficulty"]

    pressure_ratio = _safe_div(target_extra_saving, total_possible_saving)

    if pressure_ratio <= 0.35:
        return "Low"

    if pressure_ratio <= 0.75:
        return "Medium"

    return "High"


def _build_insights(
    currency: str,
    average_expenses: float,
    forecasted_expenses: float,
    categories: List[CategorySummary],
    months_count: int,
) -> List[str]:
    insights: List[str] = []

    insights.append(
        f"Based on previous spending behavior, Finexa expects next month expenses "
        f"to be around {_format_money(forecasted_expenses, currency)}."
    )

    if months_count < 3:
        insights.append(
            "Forecasting used an average-based fallback because the available history is limited."
        )

    category_stats = []

    for category in categories:
        avg_amount = _category_average(category, months_count)
        percentage = _category_percentage(category, avg_amount, average_expenses)

        if avg_amount > 0:
            category_stats.append(
                {
                    "name": category.categoryName,
                    "percentage": percentage,
                    "trend": category.trend,
                }
            )

    if category_stats:
        top_category = max(category_stats, key=lambda item: item["percentage"])

        insights.append(
            f"{top_category['name']} represents about "
            f"{round(top_category['percentage'], 1)}% of monthly expenses."
        )

    increasing_categories = [
        item["name"]
        for item in category_stats
        if str(item["trend"]) == "Increasing"
    ]

    if increasing_categories:
        insights.append(
            f"{increasing_categories[0]} increased recently and should be monitored."
        )

    subscription_categories = [
        item["name"]
        for item in category_stats
        if "subscription" in item["name"].lower()
    ]

    if subscription_categories:
        insights.append("Subscriptions should be reviewed for unused recurring payments.")

    return insights


def _build_summary_message(
    plan_status: str,
    plan_type: str,
    currency: str,
    extra_saving: float,
    recommended_monthly_saving: float,
    target_monthly_saving: Optional[float],
) -> str:
    if plan_status == "NotEnoughData":
        return "Add more transactions for at least one full month to generate an accurate saving plan."

    if plan_status == "MissingIncomeData":
        return "Income data is required to calculate your saving capacity."

    if plan_status == "Critical":
        return (
            "Your expenses are currently higher than your income. "
            "The plan will focus on reducing flexible spending first."
        )

    if plan_status == "Unrealistic":
        if target_monthly_saving is not None:
            return (
                f"Your target of {_format_money(target_monthly_saving, currency)}/month "
                "is higher than the safe saving opportunity currently available. "
                f"Finexa recommends a safer target around "
                f"{_format_money(recommended_monthly_saving, currency)}/month "
                f"using a {plan_type.lower()} plan."
            )

        return (
            "The requested saving target is too high based on your current income "
            "and spending structure."
        )

    if target_monthly_saving is not None and extra_saving <= 0:
        return (
            f"You already meet the target of {_format_money(target_monthly_saving, currency)} "
            "based on your current average saving."
        )

    if plan_status == "Hard":
        return (
            f"Your target is possible but requires strong spending control. "
            f"Recommended monthly saving is around "
            f"{_format_money(recommended_monthly_saving, currency)}."
        )

    return (
        f"You can increase your monthly saving by around "
        f"{_format_money(extra_saving, currency)} using a "
        f"{plan_type.lower()} plan focused on flexible spending categories."
    )


def build_saving_plan_ai(dto: SavingPlanRequest) -> Dict[str, Any]:
    months = int(dto.months)
    plan_type = dto.planType
    currency = dto.currency or "EGP"

    monthly_summary = sorted(dto.monthlySummary, key=lambda item: item.month)

    valid_monthly_summary = [
        item
        for item in monthly_summary
        if _to_float(item.income) > 0 or _to_float(item.expenses) > 0
    ]

    if not valid_monthly_summary:
        return _empty_response(
            dto=dto,
            plan_status="NotEnoughData",
            summary_message="Add more transactions for at least one full month to generate an accurate saving plan.",
            warnings=[
                "No usable monthly summary was provided by the backend."
            ],
        )

    incomes = [_to_float(item.income) for item in valid_monthly_summary]
    expenses = [_to_float(item.expenses) for item in valid_monthly_summary]

    average_income = _mean(incomes)
    average_expenses = _mean(expenses)

    if average_income <= 0:
        forecasted_expenses = _forecast_next_value(expenses)

        return _empty_response(
            dto=dto,
            plan_status="MissingIncomeData",
            summary_message="Income data is required to calculate your saving capacity.",
            average_income=0.0,
            average_expenses=average_expenses,
            current_average_saving=0.0,
            forecasted_income=0.0,
            forecasted_expenses=forecasted_expenses,
            forecasted_saving=0.0,
            difficulty="High",
            warnings=[
                "The backend did not provide income values in monthlySummary."
            ],
        )

    current_average_saving = average_income - average_expenses

    forecasted_income = _forecast_next_value(incomes)
    forecasted_expenses = _forecast_next_value(expenses)
    forecasted_saving = forecasted_income - forecasted_expenses

    if not dto.categorySummary and average_expenses > 0:
        return _empty_response(
            dto=dto,
            plan_status="NotEnoughData",
            summary_message="Category summary is required to generate saving recommendations.",
            average_income=average_income,
            average_expenses=average_expenses,
            current_average_saving=current_average_saving,
            forecasted_income=forecasted_income,
            forecasted_expenses=forecasted_expenses,
            forecasted_saving=forecasted_saving,
            warnings=[
                "No categorySummary was provided by the backend."
            ],
        )

    candidates = _build_candidates(
        categories=dto.categorySummary,
        plan_type=plan_type,
        months=months,
        average_expenses=average_expenses,
    )

    total_possible_saving = round(sum(item["maxSaving"] for item in candidates), 2)

    target_monthly_saving = dto.targetMonthlySaving
    target_extra_saving = 0.0

    if average_expenses > average_income:
        plan_status = "Critical"
        target_extra_saving = total_possible_saving

    elif target_monthly_saving is None:
        plan_status = "Realistic"
        target_extra_saving = total_possible_saving

    elif target_monthly_saving <= current_average_saving:
        plan_status = "Realistic"
        target_extra_saving = 0.0

    else:
        required_extra = target_monthly_saving - current_average_saving

        if total_possible_saving <= 0:
            plan_status = "Unrealistic"
            target_extra_saving = 0.0

        elif required_extra > total_possible_saving * 1.05:
            plan_status = "Unrealistic"
            target_extra_saving = total_possible_saving

        elif required_extra > total_possible_saving * 0.75:
            plan_status = "Hard"
            target_extra_saving = required_extra

        else:
            plan_status = "Realistic"
            target_extra_saving = required_extra

    allocations = _allocate_reductions(
        candidates=candidates,
        target_extra_saving=target_extra_saving,
    )

    recommendations = _build_recommendations(
        candidates=candidates,
        allocations=allocations,
    )

    achieved_extra_saving = round(
        sum(item["expectedSaving"] for item in recommendations),
        2,
    )

    recommended_monthly_saving = current_average_saving + achieved_extra_saving

    if target_monthly_saving is not None and target_monthly_saving <= current_average_saving:
        recommended_monthly_saving = current_average_saving

    recommended_monthly_saving = max(0.0, recommended_monthly_saving)
    extra_saving_opportunity = max(0.0, recommended_monthly_saving - current_average_saving)

    difficulty = _difficulty_level(
    plan_type=plan_type,
    plan_status=plan_status,
    target_extra_saving=target_extra_saving,
    total_possible_saving=total_possible_saving,
    )

    if target_monthly_saving is None and plan_status == "Realistic":
        difficulty = PLAN_RULES[plan_type]["difficulty"]

    warnings: List[str] = []

    if len(valid_monthly_summary) < 3:
        warnings.append(
            "Historical data is limited; the forecast uses a conservative average-based approach."
        )

    if any(item["categoryType"] == "Essential" for item in candidates):
        warnings.append(
            "Essential categories were not aggressively reduced."
        )

    if plan_status == "Unrealistic":
        warnings.append(
            "The requested target exceeds the safe saving opportunity calculated from current categories."
        )

    if plan_status == "Critical":
        warnings.append(
            "Expenses are higher than income; recommendations focus on flexible categories first."
        )

    insights = _build_insights(
        currency=currency,
        average_expenses=average_expenses,
        forecasted_expenses=forecasted_expenses,
        categories=dto.categorySummary,
        months_count=len(valid_monthly_summary),
    )

    summary_message = _build_summary_message(
        plan_status=plan_status,
        plan_type=plan_type,
        currency=currency,
        extra_saving=extra_saving_opportunity,
        recommended_monthly_saving=recommended_monthly_saving,
        target_monthly_saving=target_monthly_saving,
    )

    return {
        "analysisPeriodMonths": months,
        "currency": currency,

        "averageIncome": round(average_income, 2),
        "averageExpenses": round(average_expenses, 2),
        "currentAverageSaving": round(current_average_saving, 2),

        "forecastedIncome": round(forecasted_income, 2),
        "forecastedExpenses": round(forecasted_expenses, 2),
        "forecastedSaving": round(forecasted_saving, 2),

        "recommendedMonthlySaving": round(recommended_monthly_saving, 2),
        "extraSavingOpportunity": round(extra_saving_opportunity, 2),

        "difficulty": difficulty,
        "planStatus": plan_status,

        "summaryMessage": summary_message,
        "recommendations": recommendations,
        "insights": insights,
        "warnings": warnings,
    }