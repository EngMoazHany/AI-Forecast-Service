from fastapi import FastAPI, HTTPException
from typing import Dict, Any

from api.forecasting_service import run_forecast, MODEL_VERSION

app = FastAPI(
    title="Finexa AI Forecast Service",
    version="1.0.0"
)

@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}

@app.post("/forecast")
async def forecast(dto: Dict[str, Any]):
    try:
        series = dto.get("series", {})
        horizon = int(dto.get("forecast_horizon", 1))
        return run_forecast(series, horizon)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))