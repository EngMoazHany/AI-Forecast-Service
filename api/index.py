from fastapi import FastAPI, HTTPException

from api.schemas import (
    ForecastRequest,
    SavingPlanRequest,
    SavingPlanResponse,
)

from api.forecasting_service import run_forecast, MODEL_VERSION as FORECAST_MODEL_VERSION
from api.saving_plan_ai_service import (
    build_saving_plan_ai,
    get_saving_plan_model_version,
)

app = FastAPI(
    title="Finexa AI Forecast Service",
    version="1.5.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Finexa AI Forecast Service",
        "version": "1.5.0",
        "forecast_model_version": FORECAST_MODEL_VERSION,
        "saving_plan_model_version": get_saving_plan_model_version(),
    }


@app.post("/forecast")
async def forecast(dto: ForecastRequest):
    try:
        series = {
            category: [point.model_dump() for point in points]
            for category, points in dto.series.items()
        }

        return run_forecast(series, dto.forecast_horizon)

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "FORECAST_ERROR",
                "message": str(e),
            },
        )


@app.post("/api/saving-plan", response_model=SavingPlanResponse)
@app.post("/saving-plan", response_model=SavingPlanResponse, include_in_schema=False)
async def saving_plan(dto: SavingPlanRequest):
    try:
        return build_saving_plan_ai(dto)

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "SAVING_PLAN_AI_ERROR",
                "message": str(e),
            },
        )