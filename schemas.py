from pydantic import BaseModel
from typing import Dict, List


class DataPoint(BaseModel):
    month: str
    amount: float


class ForecastRequest(BaseModel):
    series: Dict[str, List[DataPoint]]
    forecast_horizon: int