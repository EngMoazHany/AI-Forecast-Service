from pydantic import BaseModel, Field
from typing import Dict, List, Literal


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

    # historical spending per category
    series: Dict[str, List[DataPoint]]

    # how many months to forecast
    forecast_horizon: int = Field(3, gt=0, le=24)


class SavingPlanMonth(BaseModel):
    month: str
    save: float
    expected_free_cash: float


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