from forecasting_service import run_forecast
from pydantic import BaseModel
from typing import Dict, List
from fastapi import FastAPI

app = FastAPI()

class MonthPoint(BaseModel):
    month: str
    amount: float

class ForecastRequest(BaseModel):
    series: Dict[str, List[MonthPoint]]
    forecast_horizon: int = 1

@app.post("/")
async def forecast(dto: ForecastRequest):
    return run_forecast(dto.series, dto.forecast_horizon)