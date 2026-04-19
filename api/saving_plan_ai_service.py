from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from api.forecasting_service import run_forecast, MODEL_VERSION as FORECAST_MODEL_VERSION
from api.optimization_engine import optimize_reductions
from api.planning_features import (
    FEATURE_COLUMNS,
    build_goal_strategy,
    build_monthly_plan,
    build_planning_features,
    compute_safe_monthly_capacity,
    extract_average_forecast_by_category,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "saving_plan_model.pkl"

DEFAULT_MODEL_VERSION = "saving_plan_ai_fallback_v1"
FEASIBLE_TOLERANCE_RATIO = 0.98


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


@lru_cache(maxsize=1)
def _load_model_bundle() -> Any:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def get_saving_plan_model_version() -> str:
    bundle = _load_model_bundle()

    if isinstance(bundle, dict):
        return str(bundle.get("model_version", DEFAULT_MODEL_VERSION))

    if bundle is not None:
        return "saving_plan_ai_single_model_v1"

    return DEFAULT_MODEL_VERSION


def _get_bundle_item(bundle: Any, *keys: str) -> Any:
    if not isinstance(bundle, dict):
        return None

    for key in keys:
        if key in bundle:
            return bundle[key]

    return None


def _resolve_primary_regressor(bundle: Any) -> Any:
    if bundle is None:
        return None

    if isinstance(bundle, dict):
        return (
            bundle.get("saving_regressor")
            or bundle.get("recommended_saving_model")
            or bundle.get("regressor")
            or bundle.get("model")
        )

    return bundle


def _resolve_feasible_classifier(bundle: Any) -> Any:
    return _get_bundle_item(
        bundle,
        "feasible_classifier",
        "feasibility_classifier",
        "feasible_model",
    )


def _resolve_cut_regressor(bundle: Any) -> Any:
    return _get_bundle_item(
        bundle,
        "cut_regressor",
        "recommended_cut_model",
        "cut_model",
    )


def _resolve_risk_classifier(bundle: Any) -> Any:
    return _get_bundle_item(
        bundle,
        "risk_classifier",
        "risk_model",
    )


def _resolve_risk_inverse_mapping(bundle: Any) -> Optional[dict]:
    value = _get_bundle_item(bundle, "risk_inverse_mapping")
    if isinstance(value, dict):
        return value
    return None


def _align_features_for_estimator(
    features_df: pd.DataFrame,
    bundle: Any,
    estimator: Any = None
) -> pd.DataFrame:
    df = features_df.copy()

    bundle_feature_names = None
    if isinstance(bundle, dict):
        bundle_feature_names = bundle.get("feature_names")

    if bundle_feature_names:
        columns = list(bundle_feature_names)
        for col in columns:
            if col not in df.columns:
                df[col] = 0.0
        return df[columns]

    if estimator is not None and hasattr(estimator, "feature_names_in_"):
        columns = list(estimator.feature_names_in_)
        for col in columns:
            if col not in df.columns:
                df[col] = 0.0
        return df[columns]

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    return df[FEATURE_COLUMNS]


def _predict_regression(estimator: Any, X: pd.DataFrame, default: float) -> float:
    if estimator is None or not hasattr(estimator, "predict"):
        return float(default)

    pred = estimator.predict(X)
    value = pred[0] if isinstance(pred, (list, tuple, np.ndarray)) else pred
    return float(value)


def _predict_probability(estimator: Any, X: pd.DataFrame, default: float) -> float:
    if estimator is None:
        return float(default)

    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
            return float(proba[0][1])

    if hasattr(estimator, "decision_function"):
        score = estimator.decision_function(X)
        raw = score[0] if isinstance(score, (list, tuple, np.ndarray)) else score
        return float(1.0 / (1.0 + np.exp(-float(raw))))

    if hasattr(estimator, "predict"):
        pred = estimator.predict(X)
        raw = pred[0] if isinstance(pred, (list, tuple, np.ndarray)) else pred
        return float(max(0.0, min(1.0, float(raw))))

    return float(default)


def _predict_risk_label(
    estimator: Any,
    X: pd.DataFrame,
    safe_capacity: float,
    required_monthly_saving: float,
    feasible_probability: float,
    risk_inverse_mapping: Optional[dict] = None,
) -> str:
    if estimator is not None and hasattr(estimator, "predict"):
        pred = estimator.predict(X)
        label = pred[0] if isinstance(pred, (list, tuple, np.ndarray)) else pred

        if isinstance(label, str):
            label = label.strip().lower()
            if label in {"low", "medium", "high"}:
                return label

        if isinstance(label, (int, float)):
            index = int(label)

            if risk_inverse_mapping and index in risk_inverse_mapping:
                mapped = str(risk_inverse_mapping[index]).strip().lower()
                if mapped in {"low", "medium", "high"}:
                    return mapped

            fallback_mapping = {0: "low", 1: "medium", 2: "high"}
            return fallback_mapping.get(index, "medium")

    ratio = (safe_capacity / required_monthly_saving) if required_monthly_saving > 0 else 1.0

    if feasible_probability >= 0.75 and ratio >= 1.0:
        return "low"
    if feasible_probability >= 0.45 or ratio >= 0.7:
        return "medium"
    return "high"


def _fallback_recommended_saving(
    safe_capacity: float,
    required_monthly_saving: float
) -> float:
    if safe_capacity <= 0:
        return 0.0
    return min(safe_capacity, required_monthly_saving)


def _apply_guardrails(
    raw_recommended: float,
    safe_capacity: float,
    required_monthly_saving: float
) -> float:
    value = max(0.0, _to_float(raw_recommended))
    value = min(value, safe_capacity)

    if required_monthly_saving > 0:
        value = min(value, required_monthly_saving)

    return round(value, 2)


def build_saving_plan_ai(dto: Any) -> Dict[str, Any]:
    if int(dto.months) <= 0:
        raise ValueError("months must be greater than 0.")

    if int(dto.forecast_horizon) <= 0:
        raise ValueError("forecast_horizon must be greater than 0.")

    if int(dto.forecast_horizon) < int(dto.months):
        raise ValueError("forecast_horizon must be greater than or equal to months.")

    income = _to_float(dto.income)
    goal_amount = _to_float(dto.goal_amount)
    months = int(dto.months)
    forecast_horizon = int(dto.forecast_horizon)

    series = {
        k: [p.model_dump() for p in v]
        for k, v in dto.series.items()
    }

    forecast_result = run_forecast(series, forecast_horizon)
    total_forecast = forecast_result.get("total_forecast", []) or []

    if not total_forecast:
        raise ValueError("Forecast result is empty.")

    planning_window = total_forecast[:months]
    predicted_expenses = [_to_float(item.get("amount")) for item in planning_window]

    avg_expense = float(np.mean(predicted_expenses)) if predicted_expenses else 0.0
    avg_free_cash = income - avg_expense
    safe_capacity = compute_safe_monthly_capacity(avg_free_cash)
    required_monthly_saving = goal_amount / months if months > 0 else 0.0

    features_df = build_planning_features(
        series=series,
        forecast_result=forecast_result,
        income=income,
        goal_amount=goal_amount,
        months=months,
        forecast_horizon=forecast_horizon,
    )

    bundle = _load_model_bundle()
    primary_regressor = _resolve_primary_regressor(bundle)
    feasible_classifier = _resolve_feasible_classifier(bundle)
    cut_regressor = _resolve_cut_regressor(bundle)
    risk_classifier = _resolve_risk_classifier(bundle)
    risk_inverse_mapping = _resolve_risk_inverse_mapping(bundle)

    X_primary = _align_features_for_estimator(features_df, bundle, primary_regressor)
    raw_recommended = _predict_regression(
        primary_regressor,
        X_primary,
        default=_fallback_recommended_saving(safe_capacity, required_monthly_saving)
    )

    recommended_monthly_saving = _apply_guardrails(
        raw_recommended=raw_recommended,
        safe_capacity=safe_capacity,
        required_monthly_saving=required_monthly_saving,
    )

    default_feasible_probability = 1.0 if safe_capacity >= required_monthly_saving else 0.0
    X_feasible = _align_features_for_estimator(features_df, bundle, feasible_classifier)
    raw_feasible_probability = _predict_probability(
        feasible_classifier,
        X_feasible,
        default=default_feasible_probability
    )

    capacity_ratio = (safe_capacity / required_monthly_saving) if required_monthly_saving > 0 else 1.0
    capacity_ratio = max(0.0, min(1.0, capacity_ratio))

    adjusted_feasible_probability = round(
        max(0.0, min(1.0, raw_feasible_probability * capacity_ratio)),
        4
    )

    if bundle is None:
        feasible = recommended_monthly_saving >= (required_monthly_saving * FEASIBLE_TOLERANCE_RATIO)
    else:
        feasible = (
            adjusted_feasible_probability >= 0.60 and
            recommended_monthly_saving >= (required_monthly_saving * FEASIBLE_TOLERANCE_RATIO)
        )

    X_cut = _align_features_for_estimator(features_df, bundle, cut_regressor)
    raw_cut_target = _predict_regression(
        cut_regressor,
        X_cut,
        default=max(0.0, required_monthly_saving - recommended_monthly_saving)
    )

    deterministic_gap = max(0.0, required_monthly_saving - recommended_monthly_saving)
    recommended_cut_target = max(
        0.0,
        round((0.60 * deterministic_gap) + (0.40 * max(0.0, raw_cut_target)), 2)
    )

    X_risk = _align_features_for_estimator(features_df, bundle, risk_classifier)
    risk_level = _predict_risk_label(
        estimator=risk_classifier,
        X=X_risk,
        safe_capacity=safe_capacity,
        required_monthly_saving=required_monthly_saving,
        feasible_probability=adjusted_feasible_probability,
        risk_inverse_mapping=risk_inverse_mapping,
    )

    gap_ratio = 0.0
    if required_monthly_saving > 0:
        gap_ratio = max(
            0.0,
            (required_monthly_saving - recommended_monthly_saving) / required_monthly_saving
        )

    if not feasible:
        if gap_ratio <= 0.05:
            risk_level = "medium"
        else:
            risk_level = "high"

    avg_by_category = extract_average_forecast_by_category(forecast_result)
    optimization = optimize_reductions(avg_by_category, recommended_cut_target)

    reductions = optimization.get("reductions", {}) or {}
    top_reductions = dict(
        sorted(reductions.items(), key=lambda x: x[1], reverse=True)[:3]
    )

    plan = build_monthly_plan(
        total_forecast=planning_window,
        income=income,
        recommended_monthly_saving=recommended_monthly_saving,
        months=months,
    )

    goal_strategy = build_goal_strategy(
        goal_amount=goal_amount,
        months=months,
        recommended_monthly_saving=recommended_monthly_saving,
    )

    if feasible and goal_strategy is not None:
        goal_strategy["recommended_timeline_months"] = months

    planner_model_version = get_saving_plan_model_version()

    insights = [
        {
            "code": "AI_SAVING_PLAN_ENGINE",
            "severity": "info" if bundle is not None else "warning",
            "title": "AI saving plan engine status",
            "message": (
                "AI planner model is loaded and used for inference."
                if bundle is not None
                else "AI planner model file is missing, so fallback planning logic was used."
            ),
            "data": {
                "planner_model_version": planner_model_version,
                "forecast_model_version": FORECAST_MODEL_VERSION,
                "raw_feasible_probability": round(raw_feasible_probability, 4),
                "adjusted_feasible_probability": round(adjusted_feasible_probability, 4),
                "capacity_ratio": round(capacity_ratio, 4),
            },
        }
    ]

    if feasible:
        insights.append(
            {
                "code": "GOAL_FEASIBLE",
                "severity": "info",
                "title": "Goal is achievable",
                "message": (
                    f"Recommended saving is {round(recommended_monthly_saving, 2)} EGP/month "
                    f"with adjusted feasibility probability {round(adjusted_feasible_probability, 2)}."
                ),
                "impact_monthly_egp": round(recommended_monthly_saving, 2),
                "recommendations": [
                    "Automate monthly saving transfer.",
                    "Track spending weekly.",
                ],
                "data": {
                    "required_monthly_saving": round(required_monthly_saving, 2),
                    "safe_monthly_saving_capacity": round(safe_capacity, 2),
                },
            }
        )
    else:
        insights.append(
            {
                "code": "GOAL_NOT_FEASIBLE",
                "severity": "critical",
                "title": "Goal is not achievable with current constraints",
                "message": (
                    f"Required saving is {round(required_monthly_saving, 2)} EGP/month, "
                    f"but recommended safe saving is {round(recommended_monthly_saving, 2)} EGP/month."
                ),
                "impact_monthly_egp": round(required_monthly_saving - recommended_monthly_saving, 2),
                "recommendations": [
                    "Extend timeline.",
                    "Increase income.",
                    "Reduce flexible categories first.",
                ],
                "data": {
                    "required_monthly_saving": round(required_monthly_saving, 2),
                    "recommended_cut_target": round(recommended_cut_target, 2),
                },
            }
        )

    if top_reductions:
        insights.append(
            {
                "code": "TOP_REDUCTION_CATEGORIES",
                "severity": "info",
                "title": "Top budget reduction categories",
                "message": "These categories are the best candidates for reduction based on current forecast.",
                "impact_monthly_egp": round(sum(top_reductions.values()), 2),
                "data": {
                    "top_reductions": {
                        k: round(v, 2)
                        for k, v in top_reductions.items()
                    }
                },
            }
        )

    return {
        "model_version": planner_model_version,
        "required_monthly_saving": round(required_monthly_saving, 2),
        "predicted_monthly_expenses_avg": round(avg_expense, 2),
        "predicted_free_cash_avg": round(avg_free_cash, 2),
        "feasible": feasible,
        "recommended_monthly_saving": round(recommended_monthly_saving, 2),
        "recommended_cut_target": round(recommended_cut_target, 2),
        "risk_level": risk_level,
        "plan": plan,
        "optimization": optimization,
        "insights": insights,
        "goal_strategy": goal_strategy,
    }