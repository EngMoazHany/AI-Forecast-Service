from datetime import datetime
from typing import List, Dict
import numpy as np

from schemas import (
    SavingPlanRequest,
    SavingPlanResponse,
    SavingPlanMonth,
    OptimizationResult
)

from api.forecasting_service import run_forecast, MODEL_VERSION
from api.optimization_engine import optimize_reductions


# ============================
# Configuration
# ============================

MAX_SAVE_RATIO = 0.7       # can't save more than 70% of free cash
MIN_BUFFER_RATIO = 0.2     # keep 20% buffer of free cash


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


def _avg_forecast_by_category(forecast_by_cat: Dict[str, List[dict]]) -> Dict[str, float]:
    """
    forecast['forecast'] = {cat: [{month, amount}, ...], ...}
    returns avg amount per category over horizon.
    """
    out: Dict[str, float] = {}
    for cat, arr in (forecast_by_cat or {}).items():
        if not arr:
            out[cat] = 0.0
            continue
        out[cat] = float(np.mean([float(x["amount"]) for x in arr]))
    return out


# ============================
# Core Engine
# ============================

def build_saving_plan(dto: SavingPlanRequest) -> SavingPlanResponse:

    # --------------------------------
    # 1) Run Forecast
    # --------------------------------
    series = {k: [p.model_dump() for p in v] for k, v in dto.series.items()}
    forecast = run_forecast(series, dto.forecast_horizon)

    total_forecast = forecast.get("total_forecast", [])
    forecast_by_cat = forecast.get("forecast", {})

    if not total_forecast:
        raise ValueError("Forecast returned empty result")

    predicted_expenses = [float(m["amount"]) for m in total_forecast]
    avg_expense = float(np.mean(predicted_expenses))

    # avg per category (monthly)
    avg_cat = _avg_forecast_by_category(forecast_by_cat)

    # --------------------------------
    # 2) Compute Disposable Income (before optimization)
    # --------------------------------
    income = float(dto.income)
    free_cash = float(income - avg_expense)

    # --------------------------------
    # 3) Required Saving (goal)
    # --------------------------------
    required_monthly = float(dto.goal_amount / dto.months)

    # --------------------------------
    # 4) Compute REQUIRED CUT (professional, constraint-aware)
    # We want free_cash' such that:
    #   (A) required_monthly <= MAX_SAVE_RATIO * free_cash'
    #   (B) free_cash' - required_monthly >= MIN_BUFFER_RATIO * free_cash'
    #
    # (B) => free_cash' * (1 - MIN_BUFFER_RATIO) >= required_monthly
    # => free_cash' >= required_monthly / (1 - MIN_BUFFER_RATIO)
    #
    # So threshold = max( required/MAX_SAVE_RATIO , required/(1-buffer) )
    # required_cut = max(0, threshold - free_cash)
    # --------------------------------
    threshold_a = required_monthly / MAX_SAVE_RATIO if MAX_SAVE_RATIO > 0 else float("inf")
    threshold_b = required_monthly / (1.0 - MIN_BUFFER_RATIO) if (1.0 - MIN_BUFFER_RATIO) > 0 else float("inf")
    required_free_cash_target = max(threshold_a, threshold_b)

    required_cut = max(0.0, required_free_cash_target - free_cash)

    # --------------------------------
    # 5) Always run Optimization (even if required_cut = 0)
    # --------------------------------
    opt = optimize_reductions(
        forecast_by_category=avg_cat,
        required_cut=required_cut,
    )

    achieved_cut = float(opt.get("achieved_cut", 0.0))

    # after optimization, expenses go down
    optimized_avg_expense = max(0.0, avg_expense - achieved_cut)
    optimized_free_cash = float(income - optimized_avg_expense)

    # --------------------------------
    # 6) Feasibility / Recommendation after optimization
    # --------------------------------
    feasible = (optimized_free_cash >= required_monthly)

    # recommended saving respects max-save-ratio
    recommended = min(required_monthly, max(0.0, optimized_free_cash * MAX_SAVE_RATIO))

    # buffer protection (keep buffer)
    buffer_needed = optimized_free_cash * MIN_BUFFER_RATIO
    if optimized_free_cash - recommended < buffer_needed:
        recommended = max(0.0, optimized_free_cash - buffer_needed)

    # if still not meeting required (rare if optimizer infeasible/no_solver)
    # recommended_cut_target can show remaining gap
    recommended_cut = max(0.0, required_free_cash_target - optimized_free_cash)

    # --------------------------------
    # 7) Risk Level (based on optimized free cash)
    # --------------------------------
    ratio = (optimized_free_cash / required_monthly) if required_monthly > 0 else 0.0
    if ratio >= 1.0:
        risk = "low"
    elif ratio >= 0.7:
        risk = "medium"
    else:
        risk = "high"

    # --------------------------------
    # 8) Monthly Plan (apply achieved_cut as monthly reduction on totals)
    # achieved_cut is monthly (because forecast numbers are monthly)
    # --------------------------------
    forecast_months = [m.get("month") for m in total_forecast if m.get("month")]

    if not forecast_months:
        start = datetime.utcnow().strftime("%Y-%m")
        forecast_months = next_months(start, dto.months)

    months = forecast_months[: dto.months]

    plan: List[SavingPlanMonth] = []
    for i, month in enumerate(months):
        base_exp = predicted_expenses[min(i, len(predicted_expenses) - 1)]
        optimized_month_exp = max(0.0, base_exp - achieved_cut)

        free = float(income - optimized_month_exp)

        save = min(recommended, max(0.0, free * MAX_SAVE_RATIO))

        # ensure buffer monthly too
        buf = free * MIN_BUFFER_RATIO
        if free - save < buf:
            save = max(0.0, free - buf)

        plan.append(
            SavingPlanMonth(
                month=month,
                save=round(save, 2),
                expected_free_cash=round(free, 2)
            )
        )

    # --------------------------------
    # 9) Response + Optimization Result
    # --------------------------------
    optimization = OptimizationResult(
        status=opt.get("status", "ok"),
        required_cut=float(opt.get("required_cut", 0.0)),
        achieved_cut=float(opt.get("achieved_cut", 0.0)),
        reductions=opt.get("reductions", {}) or {},
        new_budgets=opt.get("new_budgets", {}) or {},
        meta=opt.get("meta"),
        note=opt.get("note"),
    )

    return SavingPlanResponse(
        model_version=MODEL_VERSION,

        required_monthly_saving=round(required_monthly, 2),

        predicted_monthly_expenses_avg=round(optimized_avg_expense, 2),
        predicted_free_cash_avg=round(optimized_free_cash, 2),

        feasible=feasible,

        recommended_monthly_saving=round(recommended, 2),

        recommended_cut_target=round(recommended_cut, 2),

        risk_level=risk,

        plan=plan,

        optimization=optimization
    )