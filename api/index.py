from fastapi import FastAPI
from api.forecasting_service import run_forecast

app = FastAPI()

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/forecast")
async def forecast(dto: dict):
    return run_forecast(dto["series"], dto.get("forecast_horizon", 1))


# 👇 مهم جدًا للسيرفرلس
def handler(request):
    return app(request.scope, request.receive, request.send)