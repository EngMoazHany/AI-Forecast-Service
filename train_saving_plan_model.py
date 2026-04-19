from __future__ import annotations

from pathlib import Path
import random
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from api.forecasting_service import run_forecast, category_mapping
from api.optimization_engine import optimize_reductions
from api.planning_features import (
    FEATURE_COLUMNS,
    build_planning_features,
    compute_safe_monthly_capacity,
    extract_average_forecast_by_category,
)

warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`"
)

RANDOM_SEED = 42
N_SCENARIOS = 240
HISTORY_MONTHS = 6

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

OUTPUT_MODEL_PATH = MODELS_DIR / "saving_plan_model.pkl"

ALL_CATEGORY_BASES = {
    "Food": (2500, 9000),
    "Transport": (600, 3500),
    "Shopping": (500, 7000),
    "Entertainment": (300, 5000),
    "Bills": (1500, 10000),
    "Others": (300, 2500),
}

CATEGORY_BASES = {
    k: v for k, v in ALL_CATEGORY_BASES.items()
    if k in category_mapping
}

if len(CATEGORY_BASES) < 3:
    raise RuntimeError(
        f"Too few supported categories from forecast model. Supported: {list(category_mapping.keys())}"
    )


def month_sequence(count: int, start_year: int = 2024, start_month: int = 1) -> list[str]:
    months = []
    year = start_year
    month = start_month

    for _ in range(count):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            month = 1
            year += 1

    return months


def generate_category_points(months: list[str], low: float, high: float) -> list[dict]:
    base = random.uniform(low, high)
    trend = random.uniform(-0.02, 0.05)
    seasonality_amp = random.uniform(0.00, 0.08)
    noise_amp = random.uniform(0.03, 0.10)

    values = []
    current = base

    for i, month in enumerate(months):
        seasonal_factor = 1.0 + seasonality_amp * np.sin((2 * np.pi * i) / 6.0)
        noisy = current * seasonal_factor * (1.0 + random.uniform(-noise_amp, noise_amp))
        amount = max(50.0, noisy)

        values.append({
            "month": month,
            "amount": round(float(amount), 2)
        })

        current = max(50.0, current * (1.0 + trend))

    return values


def generate_series(history_months: int = HISTORY_MONTHS) -> dict[str, list[dict]]:
    months = month_sequence(history_months)
    series = {}

    for category, (low, high) in CATEGORY_BASES.items():
        series[category] = generate_category_points(months, low, high)

    return series


def estimate_baseline_monthly_expense(series: dict[str, list[dict]]) -> float:
    total = 0.0
    for points in series.values():
        total += float(np.mean([float(p["amount"]) for p in points]))
    return total


def derive_teacher_labels(
    forecast_result: dict,
    income: float,
    goal_amount: float,
    months: int
) -> dict:
    planning_window = forecast_result["total_forecast"][:months]
    predicted_expenses = [float(item["amount"]) for item in planning_window]

    avg_expense = float(np.mean(predicted_expenses)) if predicted_expenses else 0.0
    avg_free_cash = float(income) - avg_expense
    safe_capacity = compute_safe_monthly_capacity(avg_free_cash)

    required_monthly_saving = float(goal_amount) / months if months > 0 else 0.0
    recommended_monthly_saving = max(0.0, min(required_monthly_saving, safe_capacity))

    feasible = 1 if safe_capacity >= required_monthly_saving else 0

    avg_by_category = extract_average_forecast_by_category(forecast_result)
    gap = max(0.0, required_monthly_saving - recommended_monthly_saving)

    optimization = optimize_reductions(avg_by_category, gap)
    max_possible_cut = (
        optimization.get("meta", {}).get("max_possible_cut", gap)
        if isinstance(optimization, dict)
        else gap
    )

    recommended_cut_target = round(min(gap, max_possible_cut), 2)

    ratio = (safe_capacity / required_monthly_saving) if required_monthly_saving > 0 else 1.0
    if ratio >= 1.0:
        risk = "low"
    elif ratio >= 0.7:
        risk = "medium"
    else:
        risk = "high"

    return {
        "recommended_monthly_saving": round(recommended_monthly_saving, 2),
        "feasible": feasible,
        "recommended_cut_target": round(recommended_cut_target, 2),
        "risk_level": risk,
    }


def generate_income_and_goal(
    baseline_expense: float,
    months: int,
    target_feasible: bool
) -> tuple[float, float]:
    """
    Generate synthetic income/goal pairs in a controlled way
    so the final dataset is balanced between feasible and infeasible cases.
    """
    if target_feasible:
        income_multiplier = random.uniform(1.20, 2.10)
        income = baseline_expense * income_multiplier

        monthly_capacity_guess = max(500.0, (income - baseline_expense) * random.uniform(0.45, 0.70))
        goal_monthly_ratio = random.uniform(0.45, 0.90)
        required_monthly_saving = monthly_capacity_guess * goal_monthly_ratio
    else:
        income_multiplier = random.uniform(0.90, 1.35)
        income = baseline_expense * income_multiplier

        monthly_capacity_guess = max(300.0, (income - baseline_expense) * random.uniform(0.20, 0.55))
        required_monthly_saving = max(
            monthly_capacity_guess * random.uniform(1.20, 2.20),
            baseline_expense * random.uniform(0.20, 0.55)
        )

    goal_amount = required_monthly_saving * months
    return float(income), float(goal_amount)


def build_training_dataset(n_scenarios: int = N_SCENARIOS) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_rows = []
    y_rows = []

    target_half = n_scenarios // 2
    feasible_count = 0
    infeasible_count = 0

    attempts = 0
    max_attempts = n_scenarios * 10

    while (feasible_count + infeasible_count) < n_scenarios and attempts < max_attempts:
        attempts += 1

        target_feasible = feasible_count < target_half
        if infeasible_count < target_half and feasible_count >= target_half:
            target_feasible = False

        series = generate_series(history_months=HISTORY_MONTHS)
        baseline_expense = estimate_baseline_monthly_expense(series)

        months = random.randint(2, 8)
        forecast_horizon = months

        income, goal_amount = generate_income_and_goal(
            baseline_expense=baseline_expense,
            months=months,
            target_feasible=target_feasible
        )

        forecast_result = run_forecast(series, forecast_horizon)

        features_df = build_planning_features(
            series=series,
            forecast_result=forecast_result,
            income=income,
            goal_amount=goal_amount,
            months=months,
            forecast_horizon=forecast_horizon,
        )

        labels = derive_teacher_labels(
            forecast_result=forecast_result,
            income=income,
            goal_amount=goal_amount,
            months=months,
        )

        actual_feasible = int(labels["feasible"])

        if target_feasible and actual_feasible != 1:
            continue

        if (not target_feasible) and actual_feasible != 0:
            continue

        X_rows.append(features_df.iloc[0].to_dict())
        y_rows.append(labels)

        if actual_feasible == 1:
            feasible_count += 1
        else:
            infeasible_count += 1

        total_done = feasible_count + infeasible_count
        if total_done % 20 == 0:
            print(
                f"Generated {total_done}/{n_scenarios} scenarios..."
                f" (feasible={feasible_count}, infeasible={infeasible_count})"
            )

    if not X_rows:
        raise RuntimeError("Failed to generate any training samples.")

    X = pd.DataFrame(X_rows)
    y = pd.DataFrame(y_rows)

    for col in FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = 0.0

    X = X[FEATURE_COLUMNS]
    return X, y


def train_and_save():
    print("Generating balanced training dataset...")
    X, y = build_training_dataset()

    print(f"Dataset shape: {X.shape}")
    print("Feasible distribution:")
    print(y["feasible"].value_counts(dropna=False))
    print()

    saving_regressor = RandomForestRegressor(
        n_estimators=120,
        max_depth=8,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )

    feasible_classifier = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=1,
        class_weight="balanced",
    )

    cut_regressor = RandomForestRegressor(
        n_estimators=100,
        max_depth=7,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )

    risk_mapping = {"low": 0, "medium": 1, "high": 2}
    risk_inverse_mapping = {0: "low", 1: "medium", 2: "high"}
    y_risk = y["risk_level"].map(risk_mapping).astype(int)

    risk_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=1,
        class_weight="balanced",
    )

    print("Training saving regressor...")
    saving_regressor.fit(X, y["recommended_monthly_saving"])

    print("Training feasibility classifier...")
    feasible_classifier.fit(X, y["feasible"])

    print("Training cut regressor...")
    cut_regressor.fit(X, y["recommended_cut_target"])

    print("Training risk classifier...")
    risk_classifier.fit(X, y_risk)

    bundle = {
        "model_version": "saving_plan_ai_v2",
        "feature_names": FEATURE_COLUMNS,
        "saving_regressor": saving_regressor,
        "feasible_classifier": feasible_classifier,
        "cut_regressor": cut_regressor,
        "risk_classifier": risk_classifier,
        "risk_inverse_mapping": risk_inverse_mapping,
        "supported_categories_for_training": list(CATEGORY_BASES.keys()),
        "training_meta": {
            "n_scenarios": int(len(X)),
            "feasible_distribution": y["feasible"].value_counts(dropna=False).to_dict(),
        },
    }

    joblib.dump(bundle, OUTPUT_MODEL_PATH)
    print(f"Model saved to: {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    train_and_save()