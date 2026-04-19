from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ===============================
# Shared Data Point
# ===============================

class DataPoint(BaseModel):
    month: str = Field(..., description="Month in YYYY-MM format")
    amount: float = Field(..., ge=0, description="Amount for the month")


# ===============================
# Forecast Schemas
# ===============================

class ForecastRequest(BaseModel):
    series: Dict[str, List[DataPoint]] = Field(
        ...,
        description="Historical monthly amounts grouped by category"
    )
    forecast_horizon: int = Field(
        ...,
        gt=0,
        le=24,
        description="Number of future months to forecast"
    )


# ===============================
# Saving Plan Request
# ===============================

class SavingPlanRequest(BaseModel):
    income: float = Field(..., gt=0, description="User monthly income")
    goal_amount: float = Field(..., gt=0, description="Target savings goal amount")
    months: int = Field(..., gt=0, le=120, description="Target timeline in months")

    series: Dict[str, List[DataPoint]] = Field(
        ...,
        description="Historical monthly amounts grouped by category"
    )
    forecast_horizon: int = Field(
        3,
        gt=0,
        le=24,
        description="Number of future months to forecast"
    )


# ===============================
# Saving Plan Month
# ===============================

class SavingPlanMonth(BaseModel):
    month: str = Field(..., description="Future month in YYYY-MM format")
    save: float = Field(..., ge=0, description="Recommended saving for that month")
    expected_free_cash: float = Field(..., description="Expected remaining free cash")


# ===============================
# Optimization Result
# ===============================

class OptimizationResult(BaseModel):
    status: Literal["ok", "infeasible", "no_solver"] = Field(
        ...,
        description="Optimization execution result"
    )

    required_cut: float = Field(..., ge=0, description="Required monthly cut to hit goal")
    achieved_cut: float = Field(..., ge=0, description="Achievable monthly cut based on constraints")

    reductions: Dict[str, float] = Field(
        default_factory=dict,
        description="Suggested reductions by category"
    )

    new_budgets: Dict[str, float] = Field(
        default_factory=dict,
        description="Projected new budgets after reductions"
    )

    meta: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional optimizer metadata"
    )

    note: Optional[str] = Field(
        default=None,
        description="Optional explanatory note"
    )


# ===============================
# Insights
# ===============================

class InsightItem(BaseModel):
    code: str = Field(..., description="Machine-readable insight code")

    severity: Literal["info", "warning", "critical"] = Field(
        ...,
        description="Insight severity level"
    )

    title: str = Field(..., description="Insight title")
    message: str = Field(..., description="Human-readable message")

    impact_monthly_egp: float = Field(
        default=0,
        description="Estimated monthly impact in EGP"
    )

    recommendations: List[str] = Field(
        default_factory=list,
        description="Suggested user actions"
    )

    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Extra structured data for the insight"
    )


# ===============================
# Goal Strategy
# ===============================

class GoalStrategy(BaseModel):
    max_possible_goal_in_timeframe: float = Field(
        ...,
        ge=0,
        description="Maximum achievable goal amount within the selected timeframe"
    )

    recommended_timeline_months: int = Field(
        ...,
        gt=0,
        description="Suggested number of months to achieve the goal"
    )

    recommended_monthly_saving: float = Field(
        ...,
        ge=0,
        description="Recommended monthly saving amount"
    )


# ===============================
# Final Response
# ===============================

class SavingPlanResponse(BaseModel):
    model_version: str = Field(..., description="Saving plan AI model version")

    required_monthly_saving: float = Field(
        ...,
        ge=0,
        description="Exact monthly saving needed to hit the goal in the selected timeframe"
    )

    predicted_monthly_expenses_avg: float = Field(
        ...,
        ge=0,
        description="Average predicted monthly expenses"
    )

    predicted_free_cash_avg: float = Field(
        ...,
        description="Average predicted free cash after expenses"
    )

    feasible: bool = Field(..., description="Whether the goal is achievable")

    recommended_monthly_saving: float = Field(
        ...,
        ge=0,
        description="AI-recommended monthly saving amount"
    )

    recommended_cut_target: float = Field(
        ...,
        ge=0,
        description="Suggested monthly budget cut target"
    )

    risk_level: Literal["low", "medium", "high"] = Field(
        ...,
        description="Overall risk assessment for achieving the goal"
    )

    plan: List[SavingPlanMonth] = Field(
        default_factory=list,
        description="Month-by-month saving plan"
    )

    optimization: OptimizationResult = Field(
        ...,
        description="Optimization result for category reductions"
    )

    insights: List[InsightItem] = Field(
        default_factory=list,
        description="Generated AI insights and recommendations"
    )

    goal_strategy: Optional[GoalStrategy] = Field(
        default=None,
        description="Suggested alternative strategy to reach the goal"
    )