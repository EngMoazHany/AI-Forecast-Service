from datetime import datetime
from typing import List
import numpy as np
import math

from schemas import (
    SavingPlanRequest,
    SavingPlanResponse,
    SavingPlanMonth,
    OptimizationResult,
    InsightItem,
    GoalStrategy
)

from api.forecasting_service import run_forecast, MODEL_VERSION
from api.optimization_engine import optimize_reductions


MAX_SAVE_RATIO = 0.7
MIN_BUFFER_RATIO = 0.2


def next_months(start: str, count: int) -> List[str]:

    year, month = map(int, start.split("-"))
    months = []

    for _ in range(count):

        month += 1

        if month == 13:
            year += 1
            month = 1

        months.append(f"{year:04d}-{month:02d}")

    return months


def compute_goal_strategy(goal_amount, months, recommended):

    if recommended <= 0:
        return None

    max_goal = recommended * months

    required_months = math.ceil(goal_amount / recommended)

    return GoalStrategy(
        max_possible_goal_in_timeframe=round(max_goal, 2),
        recommended_timeline_months=required_months,
        recommended_monthly_saving=round(recommended, 2)
    )


def build_saving_plan(dto: SavingPlanRequest):

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

    buffer_needed = free_cash * MIN_BUFFER_RATIO

    if free_cash - recommended < buffer_needed:
        recommended = max(0, free_cash - buffer_needed)

    required_cut = max(0, required_monthly - free_cash)

    avg_cat = {}

    for cat, arr in forecast_by_cat.items():
        avg_cat[cat] = float(np.mean([x["amount"] for x in arr]))

    optimization = optimize_reductions(avg_cat, required_cut)

    reductions = optimization.get("reductions", {})

    top_reductions = dict(
        sorted(reductions.items(), key=lambda x: x[1], reverse=True)[:3]
    )

    insights = []

    if not feasible:

        insights.append(
            InsightItem(
                code="GOAL_NOT_FEASIBLE",
                severity="critical",
                title="Goal is not achievable with current timeframe",
                message=f"Required saving {required_monthly} but capacity {round(recommended,2)}.",
                impact_monthly_egp=required_monthly - recommended,
                recommendations=[
                    "Extend timeline",
                    "Increase income"
                ]
            )
        )

    plan = []

    forecast_months = [m["month"] for m in total_forecast]

    months = forecast_months[: dto.months]

    for i, month in enumerate(months):

        expense = predicted_expenses[min(i, len(predicted_expenses) - 1)]

        free = income - expense

        save = min(recommended, free * MAX_SAVE_RATIO)

        plan.append(
            SavingPlanMonth(
                month=month,
                save=round(save, 2),
                expected_free_cash=round(free, 2)
            )
        )

    goal_strategy = compute_goal_strategy(
        dto.goal_amount,
        dto.months,
        recommended
    )

    return SavingPlanResponse(

        model_version=MODEL_VERSION,

        required_monthly_saving=round(required_monthly, 2),

        predicted_monthly_expenses_avg=round(avg_expense, 2),

        predicted_free_cash_avg=round(free_cash, 2),

        feasible=feasible,

        recommended_monthly_saving=round(recommended, 2),

        recommended_cut_target=round(required_cut, 2),

        risk_level="low" if feasible else "high",

        plan=plan,

        optimization=OptimizationResult(**optimization),

        insights=insights,

        goal_strategy=goal_strategy
    )