from __future__ import annotations

from typing import Dict, List, Any, Tuple
import numpy as np

from schemas import (
    InsightsRequest,
    InsightsResponse,
    InsightItem,
)

from api.forecasting_service import run_forecast, MODEL_VERSION
from api.optimization_engine import optimize_reductions

# same safety knobs used elsewhere
MAX_SAVE_RATIO = 0.7
MIN_BUFFER_RATIO = 0.2


def _avg_forecast_by_category(forecast_by_cat: Dict[str, List[dict]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for cat, arr in (forecast_by_cat or {}).items():
        if not arr:
            out[cat] = 0.0
        else:
            out[cat] = float(np.mean([float(x["amount"]) for x in arr]))
    return out


def _top_k(d: Dict[str, float], k: int = 3) -> List[Tuple[str, float]]:
    return sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]


def generate_insights(dto: InsightsRequest) -> InsightsResponse:
    # ---------------------------
    # 1) Forecast
    # ---------------------------
    series = {k: [p.model_dump() for p in v] for k, v in dto.series.items()}
    forecast = run_forecast(series, dto.forecast_horizon)

    total_forecast = forecast.get("total_forecast", [])
    forecast_by_cat = forecast.get("forecast", {})

    if not total_forecast:
        raise ValueError("Forecast returned empty result")

    predicted_expenses = [float(m["amount"]) for m in total_forecast]
    avg_expense = float(np.mean(predicted_expenses))

    avg_cat = _avg_forecast_by_category(forecast_by_cat)

    income = float(dto.income)
    free_cash = float(income - avg_expense)

    required_monthly = float(dto.goal_amount / dto.months)

    # feasibility after constraints
    threshold_a = required_monthly / MAX_SAVE_RATIO if MAX_SAVE_RATIO > 0 else float("inf")
    threshold_b = required_monthly / (1.0 - MIN_BUFFER_RATIO) if (1.0 - MIN_BUFFER_RATIO) > 0 else float("inf")
    required_free_cash_target = max(threshold_a, threshold_b)
    required_cut = max(0.0, required_free_cash_target - free_cash)

    # ---------------------------
    # 2) Optimization (always)
    # ---------------------------
    opt = optimize_reductions(avg_cat, required_cut)
    achieved_cut = float(opt.get("achieved_cut", 0.0))
    opt_status = opt.get("status", "ok")

    optimized_avg_expense = max(0.0, avg_expense - achieved_cut)
    optimized_free_cash = float(income - optimized_avg_expense)

    feasible = optimized_free_cash >= required_monthly

    # recommended saving with guardrails
    recommended = min(required_monthly, max(0.0, optimized_free_cash * MAX_SAVE_RATIO))
    buf = optimized_free_cash * MIN_BUFFER_RATIO
    if optimized_free_cash - recommended < buf:
        recommended = max(0.0, optimized_free_cash - buf)

    # risk
    ratio = (optimized_free_cash / required_monthly) if required_monthly > 0 else 0.0
    if ratio >= 1.0:
        risk = "low"
    elif ratio >= 0.7:
        risk = "medium"
    else:
        risk = "high"

    # ---------------------------
    # 3) Insights (Rules + Metrics)
    # ---------------------------
    insights: List[InsightItem] = []

    # A) Goal feasibility insight
    if feasible:
        insights.append(
            InsightItem(
                code="GOAL_FEASIBLE",
                severity="info",
                title="Goal is achievable",
                message=f"Based on your forecasted spending, saving about {round(recommended,2)} EGP per month should keep you on track.",
                impact_monthly_egp=round(recommended, 2),
                recommendations=[
                    "Automate the monthly transfer to a savings account.",
                    "Review progress monthly and adjust if expenses change."
                ],
                data={
                    "required_monthly_saving": round(required_monthly, 2),
                    "recommended_monthly_saving": round(recommended, 2),
                }
            )
        )
    else:
        gap = max(0.0, required_monthly - optimized_free_cash)
        insights.append(
            InsightItem(
                code="GOAL_NOT_FEASIBLE",
                severity="critical",
                title="Goal is not achievable with current timeframe",
                message=(
                    f"Your required saving is {round(required_monthly,2)} EGP/month, "
                    f"but your realistic capacity is about {round(recommended,2)} EGP/month."
                ),
                impact_monthly_egp=round(gap, 2),
                recommendations=[
                    "Extend the goal timeline (increase months).",
                    "Reduce the goal amount.",
                    "Increase income or create an additional income stream."
                ],
                data={
                    "required": round(required_monthly, 2),
                    "capacity": round(recommended, 2),
                    "optimization_status": opt_status
                }
            )
        )

    # B) Spending concentration insight (top categories share)
    total = float(sum(avg_cat.values())) if avg_cat else 0.0
    if total > 0:
        top3 = _top_k(avg_cat, 3)
        top3_sum = sum(v for _, v in top3)
        share = top3_sum / total

        if share >= 0.75:
            insights.append(
                InsightItem(
                    code="SPENDING_CONCENTRATION_HIGH",
                    severity="warning",
                    title="Spending is concentrated in a few categories",
                    message=(
                        f"Top categories represent {round(share*100,1)}% of your forecasted monthly expenses. "
                        f"Reducing 1–2 of them can significantly improve savings."
                    ),
                    recommendations=[
                        "Focus on reducing the biggest category by 5–10%.",
                        "Set per-category monthly caps (budgets)."
                    ],
                    data={"top_categories": [{"category": c, "amount": round(v,2)} for c, v in top3]}
                )
            )

    # C) Optimization recommendations insight (if reductions exist)
    reductions = opt.get("reductions", {}) or {}
    if reductions:
        top_red = dict(sorted(reductions.items(), key=lambda x: x[1], reverse=True)[:3])
        insights.append(
            InsightItem(
                code="OPTIMIZATION_RECOMMENDATION",
                severity="info" if feasible else "warning",
                title="Budget adjustment suggestion",
                message=(
                    "To improve your plan, consider reducing the following categories "
                    f"(monthly): {', '.join([f'{k}: {round(v,2)}' for k,v in top_red.items()])} EGP."
                ),
                impact_monthly_egp=float(round(sum(top_red.values()), 2)),
                recommendations=[
                    "Start with the largest reduction first for fastest impact.",
                    "Track actual spending weekly to maintain the new budgets."
                ],
                data={"top_reductions": {k: round(v,2) for k, v in top_red.items()}}
            )
        )

    # D) If optimizer infeasible (hard limit reached)
    if opt_status == "infeasible":
        max_possible_cut = opt.get("achieved_cut", 0.0)
        insights.append(
            InsightItem(
                code="OPTIMIZATION_INFEASIBLE",
                severity="critical",
                title="Even maximum reductions cannot meet the target",
                message=(
                    f"Even after maximum feasible reductions (~{round(max_possible_cut,2)} EGP/month), "
                    "the goal still cannot be met under current constraints."
                ),
                recommendations=[
                    "Increase months substantially.",
                    "Increase income or lower the goal amount.",
                    "Re-evaluate fixed costs (Bills) if possible."
                ],
                data={"required_cut": round(float(opt.get('required_cut',0.0)),2), "achieved_cut": round(float(max_possible_cut),2)}
            )
        )

    top_reductions = dict(sorted(reductions.items(), key=lambda x: x[1], reverse=True)[:3]) if reductions else {}

    return InsightsResponse(
        model_version=MODEL_VERSION,
        feasible=feasible,
        required_monthly_saving=round(required_monthly, 2),
        predicted_monthly_expenses_avg=round(optimized_avg_expense, 2),
        predicted_free_cash_avg=round(optimized_free_cash, 2),
        recommended_monthly_saving=round(recommended, 2),
        risk_level=risk,  # type: ignore
        insights=insights,
        optimization_status=opt_status,
        top_reductions={k: round(float(v), 2) for k, v in top_reductions.items()},
    )