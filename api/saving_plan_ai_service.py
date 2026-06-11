from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from api.schemas import SavingPlanRequest, CategorySummary


MODEL_VERSION = "saving_plan_ai_v3_category_mapping"

PLAN_RULES: Dict[str, Dict[str, Any]] = {
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

# Backend categories are detailed.
# Saving Plan uses internal financial grouping and category protection rules.
SAVING_CATEGORY_MAPPING: Dict[str, Dict[str, str]] = {
    # Food group
    "food": {
        "canonical": "Food",
        "categoryType": "Flexible",
    },
    "foodd": {
        "canonical": "Food",
        "categoryType": "Flexible",
    },
    "foods": {
        "canonical": "Food",
        "categoryType": "Flexible",
    },
    "fod": {
        "canonical": "Food",
        "categoryType": "Flexible",
    },
    "drink": {
        "canonical": "Food",
        "categoryType": "Flexible",
    },
    "drinks": {
        "canonical": "Food",
        "categoryType": "Flexible",
    },
    "coffee": {
        "canonical": "Food",
        "categoryType": "Flexible",
    },
    "cafe": {
        "canonical": "Food",
        "categoryType": "Flexible",
    },
    "tea": {
        "canonical": "Food",
        "categoryType": "Flexible",
    },
    "juice": {
        "canonical": "Food",
        "categoryType": "Flexible",
    },
    "groceries": {
        "canonical": "Food",
        "categoryType": "Essential",
    },
    "grocery": {
        "canonical": "Food",
        "categoryType": "Essential",
    },
    "supermarket": {
        "canonical": "Food",
        "categoryType": "Essential",
    },
    "market": {
        "canonical": "Food",
        "categoryType": "Essential",
    },

    # Transport group
    "transport": {
        "canonical": "Transport",
        "categoryType": "Essential",
    },
    "transportation": {
        "canonical": "Transport",
        "categoryType": "Essential",
    },
    "uber": {
        "canonical": "Transport",
        "categoryType": "Flexible",
    },
    "taxi": {
        "canonical": "Transport",
        "categoryType": "Flexible",
    },
    "fuel": {
        "canonical": "Transport",
        "categoryType": "Essential",
    },
    "gasoline": {
        "canonical": "Transport",
        "categoryType": "Essential",
    },
    "bus": {
        "canonical": "Transport",
        "categoryType": "Essential",
    },
    "metro": {
        "canonical": "Transport",
        "categoryType": "Essential",
    },
    "travel": {
        "canonical": "Transport",
        "categoryType": "Flexible",
    },
    "traval": {
        "canonical": "Transport",
        "categoryType": "Flexible",
    },
    "travl": {
        "canonical": "Transport",
        "categoryType": "Flexible",
    },
    "trip": {
        "canonical": "Transport",
        "categoryType": "Flexible",
    },

    # Shopping group
    "shopping": {
        "canonical": "Shopping",
        "categoryType": "Flexible",
    },
    "shop": {
        "canonical": "Shopping",
        "categoryType": "Flexible",
    },
    "clothes": {
        "canonical": "Shopping",
        "categoryType": "Flexible",
    },
    "fashion": {
        "canonical": "Shopping",
        "categoryType": "Flexible",
    },
    "accessories": {
        "canonical": "Shopping",
        "categoryType": "Flexible",
    },
    "electronics": {
        "canonical": "Shopping",
        "categoryType": "Flexible",
    },
    "electronic": {
        "canonical": "Shopping",
        "categoryType": "Flexible",
    },
    "devices": {
        "canonical": "Shopping",
        "categoryType": "Flexible",
    },
    "device": {
        "canonical": "Shopping",
        "categoryType": "Flexible",
    },
    "mobile": {
        "canonical": "Shopping",
        "categoryType": "Flexible",
    },
    "laptop": {
        "canonical": "Shopping",
        "categoryType": "Flexible",
    },
    "computer": {
        "canonical": "Shopping",
        "categoryType": "Flexible",
    },

    # Bills group
    "bills": {
        "canonical": "Bills",
        "categoryType": "Essential",
    },
    "bill": {
        "canonical": "Bills",
        "categoryType": "Essential",
    },
    "rent": {
        "canonical": "Bills",
        "categoryType": "Essential",
    },
    "rentt": {
        "canonical": "Bills",
        "categoryType": "Essential",
    },
    "rnt": {
        "canonical": "Bills",
        "categoryType": "Essential",
    },
    "housing": {
        "canonical": "Bills",
        "categoryType": "Essential",
    },
    "house": {
        "canonical": "Bills",
        "categoryType": "Essential",
    },
    "home": {
        "canonical": "Bills",
        "categoryType": "Essential",
    },
    "utility": {
        "canonical": "Bills",
        "categoryType": "Essential",
    },
    "utilities": {
        "canonical": "Bills",
        "categoryType": "Essential",
    },
    "utilties": {
        "canonical": "Bills",
        "categoryType": "Essential",
    },
    "utlities": {
        "canonical": "Bills",
        "categoryType": "Essential",
    },
    "electricity": {
        "canonical": "Bills",
        "categoryType": "Essential",
    },
    "water": {
        "canonical": "Bills",
        "categoryType": "Essential",
    },
    "gas": {
        "canonical": "Bills",
        "categoryType": "Essential",
    },
    "internet": {
        "canonical": "Bills",
        "categoryType": "Essential",
    },
    "phone": {
        "canonical": "Bills",
        "categoryType": "Essential",
    },
    "subscription": {
        "canonical": "Bills",
        "categoryType": "Essential",
    },
    "subscriptions": {
        "canonical": "Bills",
        "categoryType": "Essential",
    },
    "netflix": {
        "canonical": "Bills",
        "categoryType": "Essential",
    },
    "spotify": {
        "canonical": "Bills",
        "categoryType": "Essential",
    },
    "youtube": {
        "canonical": "Bills",
        "categoryType": "Essential",
    },
    "software": {
        "canonical": "Bills",
        "categoryType": "Essential",
    },
    "saas": {
        "canonical": "Bills",
        "categoryType": "Essential",
    },
    "gym": {
        "canonical": "Bills",
        "categoryType": "Essential",
    },
    "fitness": {
        "canonical": "Bills",
        "categoryType": "Essential",
    },
    "receipt": {
        "canonical": "Bills",
        "categoryType": "Essential",
    },
    "receipts": {
        "canonical": "Bills",
        "categoryType": "Essential",
    },

    # Entertainment
    "entertainment": {
        "canonical": "Entertainment",
        "categoryType": "Flexible",
    },
    "entrtnmnt": {
        "canonical": "Entertainment",
        "categoryType": "Flexible",
    },
    "fun": {
        "canonical": "Entertainment",
        "categoryType": "Flexible",
    },
    "movies": {
        "canonical": "Entertainment",
        "categoryType": "Flexible",
    },
    "movie": {
        "canonical": "Entertainment",
        "categoryType": "Flexible",
    },
    "games": {
        "canonical": "Entertainment",
        "categoryType": "Flexible",
    },
    "gaming": {
        "canonical": "Entertainment",
        "categoryType": "Flexible",
    },

    # Health
    "health": {
        "canonical": "Health",
        "categoryType": "Essential",
    },
    "helth": {
        "canonical": "Health",
        "categoryType": "Essential",
    },
    "medical": {
        "canonical": "Health",
        "categoryType": "Essential",
    },
    "medicine": {
        "canonical": "Health",
        "categoryType": "Essential",
    },
    "pharmacy": {
        "canonical": "Health",
        "categoryType": "Essential",
    },
    "doctor": {
        "canonical": "Health",
        "categoryType": "Essential",
    },

    # Education
    "education": {
        "canonical": "Education",
        "categoryType": "Essential",
    },
    "educaton": {
        "canonical": "Education",
        "categoryType": "Essential",
    },
    "school": {
        "canonical": "Education",
        "categoryType": "Essential",
    },
    "course": {
        "canonical": "Education",
        "categoryType": "Essential",
    },
    "courses": {
        "canonical": "Education",
        "categoryType": "Essential",
    },
    "tuition": {
        "canonical": "Education",
        "categoryType": "Essential",
    },
    "university": {
        "canonical": "Education",
        "categoryType": "Essential",
    },

    # Other Expense
    "other": {
        "canonical": "Other Expense",
        "categoryType": "Flexible",
    },
    "others": {
        "canonical": "Other Expense",
        "categoryType": "Flexible",
    },
    "other expense": {
        "canonical": "Other Expense",
        "categoryType": "Flexible",
    },
    "misc": {
        "canonical": "Other Expense",
        "categoryType": "Flexible",
    },
    "miscellaneous": {
        "canonical": "Other Expense",
        "categoryType": "Flexible",
    },
    "unknown": {
        "canonical": "Other Expense",
        "categoryType": "Flexible",
    },
}

EXCLUDED_SAVING_PLAN_CATEGORIES = {
    "saving",
    "savings",
    "goal",
    "goals",
    "balance adjustment",
}

MIN_EXPECTED_SAVING_FOR_RECOMMENDATION = 50.0
MIN_ESSENTIAL_RECOMMENDATION_AMOUNT = 100.0


def get_saving_plan_model_version() -> str:
    return MODEL_VERSION


def _as_str(value: Any) -> str:
    if value is None:
        return ""

    if hasattr(value, "value"):
        return str(value.value)

    return str(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default

        result = float(value)

        if np.isnan(result) or np.isinf(result):
            return default

        return result

    except Exception:
        return default


def _round_money(value: float) -> float:
    return round(max(float(value), 0.0), 2)


def _format_money(value: float) -> str:
    return f"{round(value):,}"


def _get_plan_type(value: Any) -> str:
    plan_type = _as_str(value).strip().title()

    if plan_type not in PLAN_RULES:
        return "Balanced"

    return plan_type


def _get_trend(value: Any) -> str:
    trend = _as_str(value).strip().title()

    if trend not in {"Increasing", "Decreasing", "Stable"}:
        return "Stable"

    return trend


def _resolve_category_policy(category: CategorySummary) -> Dict[str, str]:
    category_name = _as_str(category.categoryName).strip()
    lower = category_name.lower()

    incoming_type = _as_str(category.categoryType).strip().title()

    if incoming_type not in {"Essential", "Flexible"}:
        incoming_type = "Flexible"

    if lower in EXCLUDED_SAVING_PLAN_CATEGORIES:
        return {
            "canonical": category_name,
            "categoryType": incoming_type,
            "isExcluded": "true",
        }

    mapped = SAVING_CATEGORY_MAPPING.get(lower)

    if mapped:
        return {
            "canonical": mapped["canonical"],
            "categoryType": mapped["categoryType"],
            "isExcluded": "false",
        }

    return {
        "canonical": category_name or "Other Expense",
        "categoryType": incoming_type,
        "isExcluded": "false",
    }


def _forecast_next_value(values: List[float]) -> float:
    clean_values = [float(v) for v in values if v is not None and float(v) >= 0]

    if not clean_values:
        return 0.0

    if len(clean_values) == 1:
        return round(clean_values[0], 2)

    average_value = float(np.mean(clean_values))
    last_value = float(clean_values[-1])
    first_value = float(clean_values[0])

    trend_step = (last_value - first_value) / max(len(clean_values) - 1, 1)
    trend_forecast = last_value + trend_step

    forecast = (0.70 * average_value) + (0.30 * trend_forecast)

    lower_bound = max(0.0, last_value * 0.80)
    upper_bound = max(lower_bound + 1e-6, last_value * 1.20)

    forecast = max(lower_bound, min(upper_bound, forecast))

    return round(float(forecast), 2)


def _build_empty_response(
    dto: SavingPlanRequest,
    status: str,
    message: str,
    average_income: float = 0.0,
    average_expenses: float = 0.0,
    current_average_saving: float = 0.0,
    forecasted_income: float = 0.0,
    forecasted_expenses: float = 0.0,
    forecasted_saving: float = 0.0,
) -> Dict[str, Any]:
    return {
        "analysisPeriodMonths": int(dto.months),
        "averageIncome": _round_money(average_income),
        "averageExpenses": _round_money(average_expenses),
        "currentAverageSaving": round(float(current_average_saving), 2),
        "forecastedIncome": _round_money(forecasted_income),
        "forecastedExpenses": _round_money(forecasted_expenses),
        "forecastedSaving": round(float(forecasted_saving), 2),
        "recommendedMonthlySaving": round(float(current_average_saving), 2),
        "extraSavingOpportunity": 0.0,
        "difficulty": "High" if status in {"Critical", "Unrealistic"} else "Low",
        "planStatus": status,
        "summaryMessage": message,
        "recommendations": [],
        "insights": [],
        "warnings": [message],
    }


def _calculate_reduction_percentage(
    category_name: str,
    category_type: str,
    trend: str,
    plan_type: str,
    percentage_of_expenses: float,
) -> float:
    rules = PLAN_RULES[plan_type]

    if category_type == "Essential":
        reduction = rules["essential_base"]

        if trend == "Increasing":
            reduction += 0.005

        if percentage_of_expenses >= 30:
            reduction += 0.005

        reduction = min(reduction, rules["essential_max"])

    else:
        reduction = rules["flexible_base"]

        if trend == "Increasing":
            reduction += 0.03

        elif trend == "Decreasing":
            reduction -= 0.02

        reduction = max(0.03, min(reduction, rules["flexible_max"]))

    return round(reduction, 4)


def _build_recommendation_reason(
    category_name: str,
    canonical_category: str,
    category_type: str,
    trend: str,
    percentage_of_expenses: float,
) -> str:
    if category_type == "Essential":
        if canonical_category == "Bills":
            return (
                f"{category_name} is treated as a recurring/essential commitment, "
                f"so Finexa recommends only a small safe reduction."
            )

        return (
            f"{category_name} is treated as an essential category, "
            f"so it was protected from aggressive reduction."
        )

    if trend == "Increasing":
        return (
            f"{category_name} is a flexible category and has been increasing recently, "
            f"so it is a strong saving opportunity."
        )

    if percentage_of_expenses >= 20:
        return (
            f"{category_name} represents a high percentage of monthly expenses "
            f"and can be reduced moderately without affecting essential needs."
        )

    return f"{category_name} is a flexible spending category that can be reduced safely."


def _build_recommendations(
    categories: List[CategorySummary],
    plan_type: str,
) -> List[Dict[str, Any]]:
    recommendations: List[Dict[str, Any]] = []

    for category in categories:
        category_name = _as_str(category.categoryName).strip() or "Unknown"
        category_id = _as_str(category.categoryId).strip()
        average_amount = _safe_float(category.averageMonthlyAmount)
        percentage_of_expenses = _safe_float(category.percentageOfExpenses)
        trend = _get_trend(category.trend)

        if average_amount <= 0:
            continue

        policy = _resolve_category_policy(category)

        if policy["isExcluded"] == "true":
            continue

        canonical_category = policy["canonical"]
        effective_category_type = policy["categoryType"]

        reduction_percentage = _calculate_reduction_percentage(
            category_name=category_name,
            category_type=effective_category_type,
            trend=trend,
            plan_type=plan_type,
            percentage_of_expenses=percentage_of_expenses,
        )

        expected_saving = average_amount * reduction_percentage

        if expected_saving < MIN_EXPECTED_SAVING_FOR_RECOMMENDATION:
            continue

        if (
            effective_category_type == "Essential"
            and expected_saving < MIN_ESSENTIAL_RECOMMENDATION_AMOUNT
        ):
            continue

        recommended_budget = max(0.0, average_amount - expected_saving)

        recommendations.append(
            {
                "categoryId": category_id,
                "categoryName": category_name,
                "categoryType": effective_category_type,
                "currentAverage": _round_money(average_amount),
                "recommendedBudget": _round_money(recommended_budget),
                "reductionPercentage": round(reduction_percentage * 100, 2),
                "expectedSaving": _round_money(expected_saving),
                "reason": _build_recommendation_reason(
                    category_name=category_name,
                    canonical_category=canonical_category,
                    category_type=effective_category_type,
                    trend=trend,
                    percentage_of_expenses=percentage_of_expenses,
                ),
            }
        )

    recommendations.sort(
        key=lambda item: (
            item["expectedSaving"],
            item["reductionPercentage"],
        ),
        reverse=True,
    )

    return recommendations


def _build_insights(
    average_expenses: float,
    forecasted_expenses: float,
    categories: List[CategorySummary],
    months_count: int,
) -> List[str]:
    insights: List[str] = []

    insights.append(
        f"Based on previous spending behavior, Finexa expects next month expenses "
        f"to be around {_format_money(forecasted_expenses)}."
    )

    if months_count < 3:
        insights.append(
            "The analysis is based on limited history, so recommendations may improve with more months of data."
        )

    valid_categories = [
        category for category in categories
        if _safe_float(category.averageMonthlyAmount) > 0
    ]

    if valid_categories:
        top_category = max(
            valid_categories,
            key=lambda c: _safe_float(c.percentageOfExpenses),
        )

        top_name = _as_str(top_category.categoryName)
        top_percentage = _safe_float(top_category.percentageOfExpenses)

        if top_percentage > 0:
            insights.append(
                f"{top_name} represents about {round(top_percentage, 1)}% of monthly expenses."
            )

    increasing_categories = [
        _as_str(category.categoryName)
        for category in valid_categories
        if _get_trend(category.trend) == "Increasing"
    ]

    if increasing_categories:
        insights.append(
            f"{increasing_categories[0]} increased recently and should be monitored."
        )

    subscription_like = [
        category for category in valid_categories
        if _resolve_category_policy(category)["canonical"] == "Bills"
        and _as_str(category.categoryName).lower() in {
            "subscriptions",
            "subscription",
            "netflix",
            "spotify",
            "youtube",
            "software",
            "saas",
        }
    ]

    if subscription_like:
        insights.append(
            "Subscriptions should be reviewed for unused recurring payments."
        )

    return insights[:5]


def _build_warnings(
    plan_status: str,
    target_monthly_saving: Optional[float],
    recommended_monthly_saving: float,
    has_essential_categories: bool,
) -> List[str]:
    warnings: List[str] = []

    if has_essential_categories:
        warnings.append("Essential categories were not aggressively reduced.")

    if (
        target_monthly_saving is not None
        and target_monthly_saving > recommended_monthly_saving
    ):
        warnings.append(
            "The requested target exceeds the safe saving opportunity calculated from current categories."
        )

    if plan_status == "Critical":
        warnings.append(
            "Expenses are higher than income, so the user should reduce spending before setting aggressive saving targets."
        )

    if not warnings:
        warnings.append(
            "The plan is based on current spending behavior and should be reviewed monthly."
        )

    return warnings


def _build_summary_message(
    plan_status: str,
    plan_type: str,
    extra_saving: float,
    recommended_monthly_saving: float,
    target_monthly_saving: Optional[float],
) -> str:
    if plan_status == "NotEnoughData":
        return "Not enough financial history to generate a reliable saving plan."

    if plan_status == "MissingIncomeData":
        return "Income data is missing, so Finexa cannot calculate a reliable saving plan."

    if plan_status == "Critical":
        return (
            "Current expenses are higher than income. Finexa recommends reducing spending "
            "before setting a monthly saving target."
        )

    if target_monthly_saving is not None and plan_status == "Unrealistic":
        return (
            f"Your target of {_format_money(target_monthly_saving)}/month is higher than "
            f"the safe saving opportunity currently available. Finexa recommends a safer "
            f"target around {_format_money(recommended_monthly_saving)}/month using a "
            f"{plan_type.lower()} plan."
        )

    if target_monthly_saving is not None and plan_status == "Hard":
        return (
            f"Your target of {_format_money(target_monthly_saving)}/month is possible, "
            f"but it requires strict control over flexible spending categories."
        )

    if target_monthly_saving is not None and plan_status == "Realistic":
        return (
            f"Your target of {_format_money(target_monthly_saving)}/month is realistic "
            f"based on your current spending behavior."
        )

    return (
        f"You can increase your monthly saving by around {_format_money(extra_saving)} "
        f"using a {plan_type.lower()} plan focused on flexible spending categories."
    )


def _determine_plan_status(
    average_income: float,
    average_expenses: float,
    current_average_saving: float,
    recommended_monthly_saving: float,
    target_monthly_saving: Optional[float],
) -> str:
    if average_income <= 0:
        return "MissingIncomeData"

    if average_expenses > average_income:
        return "Critical"

    if target_monthly_saving is None:
        return "Realistic"

    if target_monthly_saving <= current_average_saving:
        return "Realistic"

    if target_monthly_saving <= recommended_monthly_saving:
        gap = target_monthly_saving - current_average_saving
        safe_opportunity = max(recommended_monthly_saving - current_average_saving, 0.0)

        if safe_opportunity > 0 and gap >= safe_opportunity * 0.85:
            return "Hard"

        return "Realistic"

    return "Unrealistic"


def _determine_difficulty(
    plan_type: str,
    plan_status: str,
    target_monthly_saving: Optional[float],
    current_average_saving: float,
    recommended_monthly_saving: float,
) -> str:
    if plan_status in {"Critical", "Unrealistic", "Hard"}:
        return "High"

    if target_monthly_saving is not None:
        gap = target_monthly_saving - current_average_saving
        safe_opportunity = recommended_monthly_saving - current_average_saving

        if safe_opportunity > 0 and gap > safe_opportunity * 0.65:
            return "High"

    return PLAN_RULES[plan_type]["difficulty"]


def build_saving_plan_ai(dto: SavingPlanRequest) -> Dict[str, Any]:
    plan_type = _get_plan_type(dto.planType)
    target_monthly_saving = (
        _safe_float(dto.targetMonthlySaving)
        if dto.targetMonthlySaving is not None
        else None
    )

    valid_monthly_summary = [
        item for item in dto.monthlySummary
        if _safe_float(item.income) >= 0 and _safe_float(item.expenses) >= 0
    ]

    if not valid_monthly_summary:
        return _build_empty_response(
            dto=dto,
            status="NotEnoughData",
            message="Not enough monthly summary data to build a saving plan.",
        )

    incomes = [_safe_float(item.income) for item in valid_monthly_summary]
    expenses = [_safe_float(item.expenses) for item in valid_monthly_summary]
    savings = [_safe_float(item.saving) for item in valid_monthly_summary]

    average_income = float(np.mean(incomes)) if incomes else 0.0
    average_expenses = float(np.mean(expenses)) if expenses else 0.0

    if savings:
        current_average_saving = float(np.mean(savings))
    else:
        current_average_saving = average_income - average_expenses

    forecasted_income = _forecast_next_value(incomes)
    forecasted_expenses = _forecast_next_value(expenses)
    forecasted_saving = forecasted_income - forecasted_expenses

    if average_income <= 0:
        return _build_empty_response(
            dto=dto,
            status="MissingIncomeData",
            message="Income data is missing or zero.",
            average_income=average_income,
            average_expenses=average_expenses,
            current_average_saving=current_average_saving,
            forecasted_income=forecasted_income,
            forecasted_expenses=forecasted_expenses,
            forecasted_saving=forecasted_saving,
        )

    recommendations = _build_recommendations(
        categories=dto.categorySummary,
        plan_type=plan_type,
    )

    extra_saving_opportunity = sum(
        _safe_float(item["expectedSaving"])
        for item in recommendations
    )

    recommended_monthly_saving = current_average_saving + extra_saving_opportunity

    plan_status = _determine_plan_status(
        average_income=average_income,
        average_expenses=average_expenses,
        current_average_saving=current_average_saving,
        recommended_monthly_saving=recommended_monthly_saving,
        target_monthly_saving=target_monthly_saving,
    )

    difficulty = _determine_difficulty(
        plan_type=plan_type,
        plan_status=plan_status,
        target_monthly_saving=target_monthly_saving,
        current_average_saving=current_average_saving,
        recommended_monthly_saving=recommended_monthly_saving,
    )

    category_policies = [
        _resolve_category_policy(category)
        for category in dto.categorySummary
    ]

    has_essential_categories = any(
        policy["categoryType"] == "Essential"
        for policy in category_policies
        if policy["isExcluded"] == "false"
    )

    insights = _build_insights(
        average_expenses=average_expenses,
        forecasted_expenses=forecasted_expenses,
        categories=dto.categorySummary,
        months_count=len(valid_monthly_summary),
    )

    warnings = _build_warnings(
        plan_status=plan_status,
        target_monthly_saving=target_monthly_saving,
        recommended_monthly_saving=recommended_monthly_saving,
        has_essential_categories=has_essential_categories,
    )

    summary_message = _build_summary_message(
        plan_status=plan_status,
        plan_type=plan_type,
        extra_saving=extra_saving_opportunity,
        recommended_monthly_saving=recommended_monthly_saving,
        target_monthly_saving=target_monthly_saving,
    )

    return {
        "analysisPeriodMonths": int(dto.months),

        "averageIncome": _round_money(average_income),
        "averageExpenses": _round_money(average_expenses),
        "currentAverageSaving": round(float(current_average_saving), 2),

        "forecastedIncome": _round_money(forecasted_income),
        "forecastedExpenses": _round_money(forecasted_expenses),
        "forecastedSaving": round(float(forecasted_saving), 2),

        "recommendedMonthlySaving": round(float(recommended_monthly_saving), 2),
        "extraSavingOpportunity": _round_money(extra_saving_opportunity),

        "difficulty": difficulty,
        "planStatus": plan_status,

        "summaryMessage": summary_message,
        "recommendations": recommendations,
        "insights": insights,
        "warnings": warnings,
    }