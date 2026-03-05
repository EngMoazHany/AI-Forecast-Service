import math
import numpy as np
from typing import Dict

from schemas import (
    InsightsRequest,
    InsightsResponse,
    InsightItem,
    GoalStrategy
)

from api.forecasting_service import run_forecast, MODEL_VERSION
from api.optimization_engine import optimize_reductions


MAX_SAVE_RATIO = 0.7
MIN_BUFFER_RATIO = 0.2


def compute_goal_strategy(goal_amount, months, recommended_monthly_saving):

    if recommended_monthly_saving <= 0:
        return None

    max_goal = recommended_monthly_saving * months

    required_months = math.ceil(goal_amount / recommended_monthly_saving)

    return GoalStrategy(
        max_possible_goal_in_timeframe=round(max_goal, 2),
        recommended_timeline_months=required_months,
        recommended_monthly_saving=round(recommended_monthly_saving, 2)
    )


def generate_insights(dto: InsightsRequest):

    series = {
        k: [p.model_dump() for p in v]
        for k, v in dto.series.items()
    }

    forecast = run_forecast(series, dto.forecast_horizon)

    total_forecast = forecast["total_forecast"]
    forecast_by_cat = forecast["forecast"]

    predicted_expenses = [m["amount"] for m in total_forecast]

    avg_expense = float(np.mean(predicted_expenses))

    income = dto.income

    free_cash = income - avg_expense

    required_monthly = dto.goal_amount / dto.months

    feasible = free_cash >= required_monthly

    recommended = min(required_monthly, free_cash * MAX_SAVE_RATIO)

    ratio = free_cash / required_monthly if required_monthly > 0 else 0

    if ratio >= 1:
        risk = "low"
    elif ratio >= 0.7:
        risk = "medium"
    else:
        risk = "high"

    avg_cat: Dict[str, float] = {}

    for cat, arr in forecast_by_cat.items():
        avg_cat[cat] = float(np.mean([x["amount"] for x in arr]))

    required_cut = max(0, required_monthly - free_cash)

    optimization = optimize_reductions(avg_cat, required_cut)

    reductions = optimization.get("reductions", {})

    top_reductions = dict(
        sorted(reductions.items(), key=lambda x: x[1], reverse=True)[:3]
    )

    insights = []

    if feasible:

        insights.append(
            InsightItem(
                code="GOAL_FEASIBLE",
                severity="info",
                title="Goal is achievable",
                message=f"Saving about {round(recommended,2)} EGP per month keeps you on track.",
                impact_monthly_egp=round(recommended,2),
                recommendations=[
                    "Automate the monthly transfer to savings.",
                    "Track spending weekly."
                ],
                data={
                    "required_monthly_saving": required_monthly,
                    "recommended_monthly_saving": recommended
                }
            )
        )

    else:

        insights.append(
            InsightItem(
                code="GOAL_NOT_FEASIBLE",
                severity="critical",
                title="Goal is not achievable with current timeframe",
                message=f"Required saving {required_monthly} EGP/month but capacity is {round(recommended,2)}.",
                impact_monthly_egp=required_monthly - recommended,
                recommendations=[
                    "Extend goal timeline",
                    "Increase income",
                    "Reduce goal amount"
                ],
                data={
                    "required": required_monthly,
                    "capacity": recommended
                }
            )
        )

    total = sum(avg_cat.values())

    top_categories = sorted(avg_cat.items(), key=lambda x: x[1], reverse=True)[:3]

    share = sum(v for _, v in top_categories) / total if total > 0 else 0

    if share > 0.75:

        insights.append(
            InsightItem(
                code="SPENDING_CONCENTRATION_HIGH",
                severity="warning",
                title="Spending concentrated in few categories",
                message="Top categories dominate your expenses.",
                recommendations=[
                    "Reduce largest category by 5-10%"
                ],
                data={
                    "top_categories": [
                        {"category": c, "amount": round(v,2)}
                        for c, v in top_categories
                    ]
                }
            )
        )

    if top_reductions:

        insights.append(
            InsightItem(
                code="OPTIMIZATION_RECOMMENDATION",
                severity="info",
                title="Budget reduction suggestions",
                message="Consider reducing following categories",
                impact_monthly_egp=sum(top_reductions.values()),
                data={"top_reductions": top_reductions}
            )
        )

    goal_strategy = compute_goal_strategy(
        dto.goal_amount,
        dto.months,
        recommended
    )

    return InsightsResponse(

        model_version=MODEL_VERSION,

        feasible=feasible,

        required_monthly_saving=round(required_monthly,2),

        predicted_monthly_expenses_avg=round(avg_expense,2),

        predicted_free_cash_avg=round(free_cash,2),

        recommended_monthly_saving=round(recommended,2),

        risk_level=risk,

        insights=insights,

        optimization_status=optimization["status"],

        top_reductions=top_reductions,

        goal_strategy=goal_strategy
    )