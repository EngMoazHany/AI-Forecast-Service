import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any

                           
                    
                           
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "global_expense_model.pkl")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
category_mapping = bundle["category_mapping"]

MODEL_VERSION = "rf_guardrails_v1"

                           
                     
                           
MAX_MOM_CHANGE = 0.25                                                           
SMOOTHING_LAMBDA = 0.6                                                 


def _next_months(last_month: str, horizon: int):
    last = datetime.strptime(last_month, "%Y-%m").replace(day=1)
    return [
        (last + pd.DateOffset(months=i)).strftime("%Y-%m")
        for i in range(1, horizon + 1)
    ]


def _get_month(x):
    return x["month"] if isinstance(x, dict) else x.month


def _get_amount(x):
    return float(x["amount"]) if isinstance(x, dict) else float(x.amount)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def run_forecast(series_dict: Dict[str, List[Any]], horizon: int):
    forecast_out = {}
    total = None

    for category, data in (series_dict or {}).items():

        if category not in category_mapping:
            raise ValueError(f"Unknown category: {category}")

        if not data or len(data) < 3:
            continue

        data_sorted = sorted(data, key=_get_month)
        values = [_get_amount(p) for p in data_sorted]

        last_month = _get_month(data_sorted[-1])
        months = _next_months(last_month, horizon)
        category_code = category_mapping[category]

        preds = []
        temp_values = values.copy()

                                  
        last_val = float(temp_values[-1])
        mean3 = float(np.mean(temp_values[-3:]))
        mean6 = float(np.mean(temp_values[-6:])) if len(temp_values) >= 6 else mean3

        for i in range(horizon):
            lag1 = float(temp_values[-1])
            lag2 = float(temp_values[-2])
            lag3 = float(temp_values[-3])
            rolling_mean = float(np.mean(temp_values[-3:]))
            month_num = datetime.strptime(months[i], "%Y-%m").month

            X = np.array([[lag1, lag2, lag3, rolling_mean, month_num, category_code]], dtype=float)
            raw_pred = float(model.predict(X)[0])

                                       
                                       
                                       
            lo_mom = last_val * (1.0 - MAX_MOM_CHANGE)
            hi_mom = last_val * (1.0 + MAX_MOM_CHANGE)

                                       
                                                     
                                                  
                                       
            volatility = float(np.std(temp_values[-6:])) if len(temp_values) >= 6 else float(np.std(temp_values))
            band = max(0.15 * mean6, 1.5 * volatility)                
            lo_mean = mean6 - band
            hi_mean = mean6 + band

                            
            lo = max(0.0, lo_mom, lo_mean)
            hi = max(lo + 1e-6, hi_mom, hi_mean)                  

            clamped = _clamp(raw_pred, lo, hi)

                                       
                                               
                                                      
                                       
            final_pred = (SMOOTHING_LAMBDA * last_val) + ((1.0 - SMOOTHING_LAMBDA) * clamped)
            final_pred = max(final_pred, 0.0)

            preds.append(final_pred)

                                    
            temp_values.append(final_pred)
            last_val = final_pred
            mean3 = float(np.mean(temp_values[-3:]))
            mean6 = float(np.mean(temp_values[-6:])) if len(temp_values) >= 6 else mean3

        forecast_out[category] = [
            {"month": months[i], "amount": round(preds[i], 2)}
            for i in range(horizon)
        ]

        if total is None:
            total = preds.copy()
        else:
            total = [total[i] + preds[i] for i in range(horizon)]

    total_output = []
    if total and forecast_out:
        any_cat = next(iter(forecast_out.values()))
        for i in range(horizon):
            total_output.append({
                "month": any_cat[i]["month"],
                "amount": round(total[i], 2)
            })

    return {
        "forecast": forecast_out,
        "total_forecast": total_output,
        "model_version": MODEL_VERSION,
        "meta": {
            "guardrails": {
                "max_mom_change": MAX_MOM_CHANGE,
                "smoothing_lambda": SMOOTHING_LAMBDA
            }
        }
    }