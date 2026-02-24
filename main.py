from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from forecasting_service import run_forecast

app = FastAPI(
    title="Finexa AI Forecast Service",
    version="1.0.0"
)

# =============================
# DTOs
# =============================

class MonthPoint(BaseModel):
    month: str
    amount: float

class ForecastRequest(BaseModel):
    series: Dict[str, List[MonthPoint]]
    forecast_horizon: int = 1

# =============================
# Routes
# =============================

@app.post("/forecast")
def forecast(dto: ForecastRequest):

    try:
        result = run_forecast(dto.series, dto.forecast_horizon)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}