from datetime import datetime
from typing import List
import numpy as np

from schemas import (
    SavingPlanRequest,
    SavingPlanResponse,
    SavingPlanMonth
)

from api.forecasting_service import run_forecast, MODEL_VERSION


# ============================
# Configuration
# ============================

MAX_SAVE_RATIO = 0.7
MIN_BUFFER_RATIO = 0.2


# ============================
# Helpers
# ============================

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


# ============================
# Core Engine
# ============================

def build_saving_plan(dto: SavingPlanRequest) -> SavingPlanResponse:

    # --------------------------------
    # 1️⃣ Run Forecast
    # --------------------------------

    series = {
        k: [p.model_dump() for p in v]
        for k, v in dto.series.items()
    }

    forecast = run_forecast(series, dto.forecast_horizon)

    total_forecast = forecast.get("total_forecast", [])

    if not total_forecast:
        raise ValueError("Forecast returned empty result")

    predicted_expenses = [m["amount"] for m in total_forecast]

    avg_expense = float(np.mean(predicted_expenses))


    # --------------------------------
    # 2️⃣ Compute Disposable Income
    # --------------------------------

    income = dto.income

    free_cash = income - avg_expense


    # --------------------------------
    # 3️⃣ Required Saving
    # --------------------------------

    required_monthly = dto.goal_amount / dto.months

    feasible = free_cash >= required_monthly


    # --------------------------------
    # 4️⃣ Recommended Saving
    # --------------------------------

    max_possible_save = max(0, free_cash * MAX_SAVE_RATIO)

    if feasible:
        recommended = min(required_monthly, max_possible_save)
    else:
        recommended = max_possible_save


    # --------------------------------
    # 5️⃣ Buffer Protection
    # --------------------------------

    buffer_needed = free_cash * MIN_BUFFER_RATIO

    if free_cash - recommended < buffer_needed:
        recommended = max(0, free_cash - buffer_needed)


    # --------------------------------
    # 6️⃣ Expense Cut Suggestion
    # --------------------------------

    recommended_cut = 0

    if not feasible:
        recommended_cut = max(0, required_monthly - free_cash)


    # --------------------------------
    # 7️⃣ Risk Level
    # --------------------------------

    ratio = free_cash / required_monthly if required_monthly > 0 else 0

    if ratio >= 1:
        risk = "low"
    elif ratio >= 0.7:
        risk = "medium"
    else:
        risk = "high"


    # --------------------------------
    # 8️⃣ Monthly Plan
    # --------------------------------

    forecast_months = [m.get("month") for m in total_forecast]

    if not forecast_months:
        start = datetime.utcnow().strftime("%Y-%m")
        forecast_months = next_months(start, dto.months)

    months = forecast_months[: dto.months]

    plan = []

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


    # --------------------------------
    # 9️⃣ Response
    # --------------------------------

    return SavingPlanResponse(

        model_version=MODEL_VERSION,

        required_monthly_saving=round(required_monthly, 2),

        predicted_monthly_expenses_avg=round(avg_expense, 2),

        predicted_free_cash_avg=round(free_cash, 2),

        feasible=feasible,

        recommended_monthly_saving=round(recommended, 2),

        recommended_cut_target=round(recommended_cut, 2),

        risk_level=risk,

        plan=plan
    )