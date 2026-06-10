from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PlanType = Literal["Easy", "Balanced", "Aggressive"]
CategoryType = Literal["Essential", "Flexible", "Unknown"]
TrendType = Literal["Increasing", "Decreasing", "Stable", "Unknown"]
PlanStatus = Literal[
    "Realistic",
    "Hard",
    "Unrealistic",
    "NotEnoughData",
    "MissingIncomeData",
    "Critical",
]
Difficulty = Literal["Low", "Medium", "High"]


def _normalize_enum(value: Any, mapping: Dict[str, str], default: str) -> str:
    if value is None:
        return default

    key = str(value).strip().lower()
    return mapping.get(key, default)


class ForecastPoint(BaseModel):
    model_config = ConfigDict(extra="ignore")

    month: str = Field(..., description="Month in YYYY-MM format")
    amount: float = Field(..., ge=0)


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    series: Dict[str, List[ForecastPoint]]
    forecast_horizon: int = Field(default=3, ge=1, le=24)


class MonthlySummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    month: str = Field(..., description="Month in YYYY-MM format")
    income: float = Field(default=0, ge=0)
    expenses: float = Field(default=0, ge=0)
    saving: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def accept_expense_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)

            if "expense" in data and "expenses" not in data:
                data["expenses"] = data["expense"]

            if "monthlyIncome" in data and "income" not in data:
                data["income"] = data["monthlyIncome"]

            if "monthlyExpenses" in data and "expenses" not in data:
                data["expenses"] = data["monthlyExpenses"]

            if "monthlySaving" in data and "saving" not in data:
                data["saving"] = data["monthlySaving"]

        return data


class CategorySummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    categoryId: Optional[str] = None
    categoryName: str
    categoryType: CategoryType = "Flexible"
    averageMonthlyAmount: float = Field(default=0, ge=0)
    totalAmount: float = Field(default=0, ge=0)
    percentageOfExpenses: float = Field(default=0, ge=0)
    trend: TrendType = "Stable"

    @model_validator(mode="before")
    @classmethod
    def accept_backend_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)

            aliases = {
                "id": "categoryId",
                "category_id": "categoryId",
                "name": "categoryName",
                "category_name": "categoryName",
                "type": "categoryType",
                "category_type": "categoryType",
                "average_monthly_amount": "averageMonthlyAmount",
                "averageAmount": "averageMonthlyAmount",
                "total_amount": "totalAmount",
                "percentage": "percentageOfExpenses",
                "percentage_of_expenses": "percentageOfExpenses",
            }

            for old_key, new_key in aliases.items():
                if old_key in data and new_key not in data:
                    data[new_key] = data[old_key]

        return data

    @field_validator("categoryType", mode="before")
    @classmethod
    def normalize_category_type(cls, value: Any) -> str:
        return _normalize_enum(
            value,
            {
                "essential": "Essential",
                "fixed": "Essential",
                "need": "Essential",
                "needs": "Essential",
                "flexible": "Flexible",
                "variable": "Flexible",
                "want": "Flexible",
                "wants": "Flexible",
                "unknown": "Unknown",
            },
            "Flexible",
        )

    @field_validator("trend", mode="before")
    @classmethod
    def normalize_trend(cls, value: Any) -> str:
        return _normalize_enum(
            value,
            {
                "increasing": "Increasing",
                "increase": "Increasing",
                "up": "Increasing",
                "decreasing": "Decreasing",
                "decrease": "Decreasing",
                "down": "Decreasing",
                "stable": "Stable",
                "flat": "Stable",
                "unknown": "Unknown",
            },
            "Stable",
        )


class SavingPlanRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    months: int = Field(default=6)
    planType: PlanType = "Balanced"
    targetMonthlySaving: Optional[float] = Field(default=None, ge=0)
    currency: str = "EGP"

    monthlySummary: List[MonthlySummary] = Field(default_factory=list)
    categorySummary: List[CategorySummary] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def accept_backend_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)

            aliases = {
                "plan_type": "planType",
                "target_monthly_saving": "targetMonthlySaving",
                "targetSaving": "targetMonthlySaving",
                "monthly_summary": "monthlySummary",
                "category_summary": "categorySummary",
            }

            for old_key, new_key in aliases.items():
                if old_key in data and new_key not in data:
                    data[new_key] = data[old_key]

        return data

    @field_validator("months")
    @classmethod
    def validate_months(cls, value: int) -> int:
        if value not in (3, 6):
            raise ValueError("months must be either 3 or 6.")
        return value

    @field_validator("planType", mode="before")
    @classmethod
    def normalize_plan_type(cls, value: Any) -> str:
        return _normalize_enum(
            value,
            {
                "easy": "Easy",
                "balanced": "Balanced",
                "balance": "Balanced",
                "medium": "Balanced",
                "aggressive": "Aggressive",
                "hard": "Aggressive",
            },
            "Balanced",
        )


class SavingRecommendation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    categoryId: Optional[str] = None
    categoryName: str
    categoryType: CategoryType
    currentAverage: float
    recommendedBudget: float
    reductionPercentage: float
    expectedSaving: float
    reason: str


class SavingPlanResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    analysisPeriodMonths: int
    currency: str

    averageIncome: float
    averageExpenses: float
    currentAverageSaving: float

    forecastedIncome: float
    forecastedExpenses: float
    forecastedSaving: float

    recommendedMonthlySaving: float
    extraSavingOpportunity: float

    difficulty: Difficulty
    planStatus: PlanStatus

    summaryMessage: str
    recommendations: List[SavingRecommendation]
    insights: List[str]
    warnings: List[str]