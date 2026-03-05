from fastapi import FastAPI, HTTPException

from schemas import (
    ForecastRequest,
    SavingPlanRequest,
    SavingPlanResponse
)

from api.forecasting_service import run_forecast, MODEL_VERSION
from api.saving_plan_service import build_saving_plan


app = FastAPI(
    title="Finexa AI Forecast Service",
    version="1.1.0"
)


# ===============================
# Health Check
# ===============================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION
    }


# ===============================
# Forecast Endpoint
# ===============================

@app.post("/forecast")
async def forecast(dto: ForecastRequest):

    try:

        # convert schema to dict structure expected by service
        series = {
            k: [p.model_dump() for p in v]
            for k, v in dto.series.items()
        }

        horizon = dto.forecast_horizon

        return run_forecast(series, horizon)

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# ===============================
# Saving Plan Endpoint
# ===============================

@app.post("/saving-plan", response_model=SavingPlanResponse)
async def saving_plan(dto: SavingPlanRequest):

    try:

        return build_saving_plan(dto)

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail={
                "code": "SAVING_PLAN_ERROR",
                "message": str(e)
            }
        )