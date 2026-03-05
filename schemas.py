from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional, Any


# ===============================
# Forecast Schemas
# ===============================

class DataPoint(BaseModel):
    month: str
    amount: float


class ForecastRequest(BaseModel):
    series: Dict[str, List[DataPoint]]
    forecast_horizon: int = Field(..., gt=0, le=24)


# ===============================
# Saving Plan Schemas
# ===============================

class SavingPlanRequest(BaseModel):
    income: float = Field(..., gt=0)
    goal_amount: float = Field(..., gt=0)
    months: int = Field(..., gt=0, le=120)

    series: Dict[str, List[DataPoint]]
    forecast_horizon: int = Field(3, gt=0, le=24)


class SavingPlanMonth(BaseModel):
    month: str
    save: float
    expected_free_cash: float


class OptimizationResult(BaseModel):

    status: Literal["ok", "infeasible", "no_solver"]

    required_cut: float
    achieved_cut: float

    reductions: Dict[str, float] = {}

    new_budgets: Dict[str, float] = {}

    meta: Optional[Dict[str, Any]] = None
    note: Optional[str] = None


class SavingPlanResponse(BaseModel):

    model_version: str

    required_monthly_saving: float

    predicted_monthly_expenses_avg: float
    predicted_free_cash_avg: float

    feasible: bool

    recommended_monthly_saving: float

    recommended_cut_target: float

    risk_level: Literal["low", "medium", "high"]

    plan: List[SavingPlanMonth]

    optimization: OptimizationResult


# ===============================
# AI Insights Schemas (NEW)
# ===============================

class InsightItem(BaseModel):

    code: str

    severity: Literal["info", "warning", "critical"]

    title: str

    message: str

    impact_monthly_egp: float = 0

    recommendations: List[str] = []

    data: Optional[Dict[str, Any]] = None


class InsightsRequest(BaseModel):

    income: float = Field(..., gt=0)

    goal_amount: float = Field(..., gt=0)

    months: int = Field(..., gt=0, le=120)

    series: Dict[str, List[DataPoint]]

    forecast_horizon: int = Field(3, gt=0, le=24)


class InsightsResponse(BaseModel):

    model_version: str

    feasible: bool

    required_monthly_saving: float

    predicted_monthly_expenses_avg: float

    predicted_free_cash_avg: float

    recommended_monthly_saving: float

    risk_level: Literal["low", "medium", "high"]

    insights: List[InsightItem]

    optimization_status: Literal["ok", "infeasible", "no_solver"]

    top_reductions: Dict[str, float] = {}