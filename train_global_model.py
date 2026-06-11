from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import json
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT_DIR = Path(__file__).resolve().parent

DATA_PATH = ROOT_DIR / "data" / "processed" / "finexa_monthly_category_amount.csv"

MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "global_expense_model.pkl"

ARTIFACTS_DIR = ROOT_DIR / "artifacts" / "presentation"
CHARTS_DIR = ARTIFACTS_DIR / "charts"
DIAGRAMS_DIR = ARTIFACTS_DIR / "diagrams"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_VERSION = "rf_global_finexa_user_category_v5_log_target"

# Canonical categories seen by the AI model.
# Backend may have more detailed categories, but the model groups them into stronger training groups.
SUPPORTED_CATEGORIES = [
    "Food",
    "Transport",
    "Shopping",
    "Bills",
    "Entertainment",
    "Health",
    "Education",
    "Other Expense",
]

CATEGORY_ALIASES: Dict[str, str] = {
    # Food group
    "food": "Food",
    "foodd": "Food",
    "foods": "Food",
    "fod": "Food",
    "drink": "Food",
    "drinks": "Food",
    "coffee": "Food",
    "cafe": "Food",
    "tea": "Food",
    "juice": "Food",
    "grocery": "Food",
    "groceries": "Food",
    "supermarket": "Food",
    "market": "Food",
    "restaurant": "Food",
    "restaurants": "Food",
    "dining": "Food",
    "meal": "Food",
    "meals": "Food",

    # Transport group
    "transport": "Transport",
    "transportation": "Transport",
    "uber": "Transport",
    "taxi": "Transport",
    "fuel": "Transport",
    "gasoline": "Transport",
    "bus": "Transport",
    "metro": "Transport",
    "travel": "Transport",
    "traval": "Transport",
    "travl": "Transport",
    "trip": "Transport",
    "flight": "Transport",
    "hotel": "Transport",

    # Shopping group
    "shopping": "Shopping",
    "shop": "Shopping",
    "clothes": "Shopping",
    "fashion": "Shopping",
    "accessories": "Shopping",
    "electronics": "Shopping",
    "electronic": "Shopping",
    "devices": "Shopping",
    "device": "Shopping",
    "mobile": "Shopping",
    "phone device": "Shopping",
    "laptop": "Shopping",
    "computer": "Shopping",

    # Bills group
    "bill": "Bills",
    "bills": "Bills",
    "utility": "Bills",
    "utilities": "Bills",
    "utilties": "Bills",
    "utlities": "Bills",
    "electricity": "Bills",
    "water": "Bills",
    "gas": "Bills",
    "internet": "Bills",
    "phone": "Bills",
    "rent": "Bills",
    "rentt": "Bills",
    "rnt": "Bills",
    "housing": "Bills",
    "house": "Bills",
    "home": "Bills",
    "subscription": "Bills",
    "subscriptions": "Bills",
    "netflix": "Bills",
    "spotify": "Bills",
    "youtube": "Bills",
    "software": "Bills",
    "saas": "Bills",
    "gym": "Bills",
    "fitness": "Bills",
    "receipt": "Bills",
    "receipts": "Bills",

    # Entertainment
    "entertainment": "Entertainment",
    "entrtnmnt": "Entertainment",
    "fun": "Entertainment",
    "movies": "Entertainment",
    "movie": "Entertainment",
    "games": "Entertainment",
    "gaming": "Entertainment",

    # Health
    "health": "Health",
    "helth": "Health",
    "medical": "Health",
    "medicine": "Health",
    "pharmacy": "Health",
    "doctor": "Health",

    # Education
    "education": "Education",
    "educaton": "Education",
    "school": "Education",
    "course": "Education",
    "courses": "Education",
    "tuition": "Education",
    "university": "Education",

    # Other Expense
    "other": "Other Expense",
    "others": "Other Expense",
    "other expense": "Other Expense",
    "misc": "Other Expense",
    "miscellaneous": "Other Expense",
    "unknown": "Other Expense",
}

# System/internal categories that should not be used in expense forecasting.
EXCLUDED_CATEGORIES = {
    "saving",
    "savings",
    "goal",
    "goals",
    "balance adjustment",
}

FEATURE_COLUMNS = [
    "lag1",
    "lag2",
    "lag3",
    "rolling_mean",
    "month_num",
    "category_code",
]

MIN_NON_ZERO_MONTHS_PER_USER_CATEGORY = 4

MAX_MOM_CHANGE = 0.25
SMOOTHING_LAMBDA = 0.6


def normalize_category(value: str) -> Optional[str]:
    raw = str(value).strip()

    if not raw:
        return "Other Expense"

    lower = raw.lower().strip()

    if lower in EXCLUDED_CATEGORIES:
        return None

    if lower in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[lower]

    for category in SUPPORTED_CATEGORIES:
        if category.lower() == lower:
            return category

    return "Other Expense"


def load_raw_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = [col.strip() for col in df.columns]

    required_cols = {"user_id", "category", "month", "amount"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset missing required columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()

    df["user_id"] = df["user_id"].astype(str).str.strip()
    df["category_original"] = df["category"].astype(str)
    df["category"] = df["category"].apply(normalize_category)

    parsed_month = pd.to_datetime(df["month"], errors="coerce")
    df["month"] = parsed_month.dt.to_period("M")

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    df = df.dropna(subset=["user_id", "category", "month", "amount"])
    df = df[df["user_id"] != ""]
    df = df[df["amount"] >= 0].copy()

    df["month"] = df["month"].astype(str)

    df = (
        df.groupby(["user_id", "category", "month"], as_index=False)["amount"]
        .sum()
        .sort_values(["user_id", "category", "month"])
        .reset_index(drop=True)
    )

    if df.empty:
        raise RuntimeError("No usable data after preprocessing.")

    return df


def cap_outliers_per_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parts: List[pd.DataFrame] = []

    for category, group in df.groupby("category"):
        group = group.copy()

        non_zero = group[group["amount"] > 0]["amount"]

        if len(non_zero) >= 10:
            q95 = non_zero.quantile(0.95)
            median = non_zero.median()
            cap = max(q95, median * 4)
            group["amount"] = group["amount"].clip(upper=cap)

        parts.append(group)

    return pd.concat(parts, ignore_index=True)


def fill_missing_months_per_user_category(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []

    for (user_id, category), group in df.groupby(["user_id", "category"], sort=False):
        group = group.sort_values("month").reset_index(drop=True)

        non_zero_months = int((group["amount"] > 0).sum())

        if non_zero_months < MIN_NON_ZERO_MONTHS_PER_USER_CATEGORY:
            continue

        start = pd.Period(group["month"].min(), freq="M")
        end = pd.Period(group["month"].max(), freq="M")

        amount_by_month = {
            str(row["month"]): float(row["amount"])
            for _, row in group.iterrows()
        }

        for period in pd.period_range(start=start, end=end, freq="M"):
            month = str(period)

            rows.append(
                {
                    "user_id": user_id,
                    "category": category,
                    "month": month,
                    "amount": amount_by_month.get(month, 0.0),
                }
            )

    result = pd.DataFrame(rows)

    if result.empty:
        raise RuntimeError(
            "No usable user-category series after filtering. "
            "Each user/category needs at least 4 non-zero monthly records."
        )

    return result.sort_values(["user_id", "category", "month"]).reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["user_id", "category", "month"]).reset_index(drop=True)

    group_cols = ["user_id", "category"]

    df["lag1"] = df.groupby(group_cols)["amount"].shift(1)
    df["lag2"] = df.groupby(group_cols)["amount"].shift(2)
    df["lag3"] = df.groupby(group_cols)["amount"].shift(3)

    # Leak-free rolling mean: previous 3 months only, not the current target month.
    df["rolling_mean"] = (
        df.groupby(group_cols)["amount"]
        .shift(1)
        .groupby([df["user_id"], df["category"]])
        .rolling(3)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    df["month_num"] = pd.to_datetime(df["month"]).dt.month

    df = df.dropna().reset_index(drop=True)

    if df.empty:
        raise RuntimeError("No rows left after lag/rolling feature generation.")

    return df


def build_category_mapping() -> Dict[str, int]:
    return {
        category: index
        for index, category in enumerate(SUPPORTED_CATEGORIES)
    }


def time_based_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    for _, group in df.groupby(["user_id", "category"], sort=False):
        group = group.sort_values("month").reset_index(drop=True)

        if len(group) < 5:
            train_parts.append(group)
            continue

        split_idx = max(3, int(len(group) * 0.8))
        split_idx = min(split_idx, len(group) - 1)

        train_parts.append(group.iloc[:split_idx])
        test_parts.append(group.iloc[split_idx:])

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()

    if train_df.empty:
        raise RuntimeError("Training set is empty.")

    return train_df, test_df


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def create_dataset_artifacts(
    raw_df: pd.DataFrame,
    capped_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> None:
    category_counts = (
        capped_df["category"]
        .value_counts()
        .rename_axis("category")
        .reset_index(name="records")
    )

    category_counts.to_csv(
        ARTIFACTS_DIR / "category_counts_after_normalization.csv",
        index=False,
    )

    dataset_summary = pd.DataFrame(
        [
            {"item": "Final dataset file", "value": str(DATA_PATH)},
            {"item": "Final format", "value": "user_id, category, month, amount"},
            {"item": "Raw rows after preprocessing", "value": len(raw_df)},
            {"item": "Rows after outlier capping", "value": len(capped_df)},
            {"item": "Monthly rows after filling missing months", "value": len(monthly_df)},
            {"item": "Feature rows used for ML", "value": len(feature_df)},
            {
                "item": "User-category series count",
                "value": monthly_df.groupby(["user_id", "category"]).ngroups,
            },
            {"item": "Canonical AI categories", "value": ", ".join(SUPPORTED_CATEGORIES)},
            {
                "item": "Backend category grouping",
                "value": (
                    "Drinks/Groceries -> Food, Electronics -> Shopping, "
                    "Subscriptions/Gym/Receipt/Rent -> Bills, Travel -> Transport"
                ),
            },
        ]
    )

    dataset_summary.to_csv(ARTIFACTS_DIR / "dataset_summary.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(category_counts["category"], category_counts["records"])
    plt.title("Category Distribution After Normalization")
    plt.xlabel("Category")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "category_distribution.png", dpi=200)
    plt.close()


def create_metrics_artifacts(metrics: Dict[str, Any], train_rows: int, test_rows: int) -> None:
    metrics_table = pd.DataFrame(
        [
            {
                "Metric": "MAE",
                "Value": metrics.get("mae"),
                "Meaning": "Average absolute prediction error",
            },
            {
                "Metric": "RMSE",
                "Value": metrics.get("rmse"),
                "Meaning": "Penalizes larger prediction errors",
            },
            {
                "Metric": "R2 Score",
                "Value": metrics.get("r2"),
                "Meaning": "Explains variance in spending behavior",
            },
            {
                "Metric": "Train Rows",
                "Value": train_rows,
                "Meaning": "Number of training samples",
            },
            {
                "Metric": "Test Rows",
                "Value": test_rows,
                "Meaning": "Number of testing samples",
            },
            {
                "Metric": "Model Version",
                "Value": MODEL_VERSION,
                "Meaning": "Final deployed forecasting model",
            },
        ]
    )

    metrics_table.to_csv(ARTIFACTS_DIR / "forecasting_metrics_table.csv", index=False)

    save_json(
        {
            "model_version": MODEL_VERSION,
            "metrics": metrics,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "task_type": "Regression",
            "why_no_confusion_matrix": (
                "Forecasting predicts numeric spending amounts, so it is a regression task. "
                "Confusion matrix is for classification tasks, therefore MAE, RMSE, and R2 are used."
            ),
            "category_grouping": {
                "Drinks": "Food",
                "Groceries": "Food",
                "Electronics": "Shopping",
                "Subscriptions": "Bills",
                "Gym": "Bills",
                "Receipt": "Bills",
                "Rent": "Bills",
                "Travel": "Transport",
            },
        },
        ARTIFACTS_DIR / "forecasting_metrics.json",
    )


def create_actual_vs_predicted_artifacts(
    test_df: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
) -> None:
    eval_df = test_df[["user_id", "category", "month"]].copy()
    eval_df["actual"] = y_test.values
    eval_df["predicted"] = predictions
    eval_df["abs_error"] = (eval_df["actual"] - eval_df["predicted"]).abs()

    eval_df.to_csv(ARTIFACTS_DIR / "actual_vs_predicted_full.csv", index=False)

    sample_df = eval_df.sort_values("abs_error").head(40).copy()
    sample_df = sample_df.reset_index(drop=True)
    sample_df["sample_index"] = sample_df.index + 1

    sample_df.to_csv(ARTIFACTS_DIR / "actual_vs_predicted_sample.csv", index=False)

    plt.figure(figsize=(11, 5))
    plt.plot(sample_df["sample_index"], sample_df["actual"], marker="o", label="Actual")
    plt.plot(sample_df["sample_index"], sample_df["predicted"], marker="o", label="Predicted")
    plt.title("Actual vs Predicted Monthly Spending")
    plt.xlabel("Sample Index")
    plt.ylabel("Amount")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "actual_vs_predicted_line.png", dpi=200)
    plt.close()

    scatter_df = eval_df.sample(min(500, len(eval_df)), random_state=42)

    plt.figure(figsize=(6, 6))
    plt.scatter(scatter_df["actual"], scatter_df["predicted"], alpha=0.6)

    max_value = max(scatter_df["actual"].max(), scatter_df["predicted"].max())
    plt.plot([0, max_value], [0, max_value], linestyle="--", label="Perfect Prediction")

    plt.title("Actual vs Predicted Scatter")
    plt.xlabel("Actual Amount")
    plt.ylabel("Predicted Amount")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "actual_vs_predicted_scatter.png", dpi=200)
    plt.close()


def next_months(last_month: str, horizon: int) -> List[str]:
    start = pd.Period(last_month, freq="M")
    return [str(start + i) for i in range(1, horizon + 1)]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def forecast_sample(
    model: Any,
    category_mapping: Dict[str, int],
    series: Dict[str, List[Dict[str, Any]]],
    horizon: int = 3,
) -> Dict[str, Any]:
    forecast_out: Dict[str, List[Dict[str, Any]]] = {}
    total_values: Optional[List[float]] = None

    for category, points in series.items():
        mapped_category = normalize_category(category)

        if mapped_category is None:
            continue

        if mapped_category not in category_mapping:
            continue

        data_sorted = sorted(points, key=lambda p: p["month"])

        if len(data_sorted) < 3:
            continue

        values = [float(item["amount"]) for item in data_sorted]
        last_month = data_sorted[-1]["month"]
        months = next_months(last_month, horizon)
        category_code = category_mapping[mapped_category]

        preds: List[float] = []
        temp_values = values.copy()

        last_value = float(temp_values[-1])

        for i in range(horizon):
            lag1 = float(temp_values[-1])
            lag2 = float(temp_values[-2])
            lag3 = float(temp_values[-3])
            rolling_mean = float(np.mean(temp_values[-3:]))
            month_num = pd.Period(months[i], freq="M").month

            X = pd.DataFrame(
                [
                    {
                        "lag1": lag1,
                        "lag2": lag2,
                        "lag3": lag3,
                        "rolling_mean": rolling_mean,
                        "month_num": month_num,
                        "category_code": category_code,
                    }
                ],
                columns=FEATURE_COLUMNS,
            )

            raw_pred = float(model.predict(X)[0])
            raw_pred = max(raw_pred, 0.0)

            mean3 = float(np.mean(temp_values[-3:]))
            mean6 = float(np.mean(temp_values[-6:])) if len(temp_values) >= 6 else mean3

            lo_mom = last_value * (1.0 - MAX_MOM_CHANGE)
            hi_mom = last_value * (1.0 + MAX_MOM_CHANGE)

            volatility = float(np.std(temp_values[-6:])) if len(temp_values) >= 6 else float(np.std(temp_values))
            band = max(0.15 * mean6, 1.5 * volatility)

            lo_mean = mean6 - band
            hi_mean = mean6 + band

            lo = max(0.0, lo_mom, lo_mean)
            hi = max(lo + 1e-6, hi_mom, hi_mean)

            clamped = clamp(raw_pred, lo, hi)

            final_pred = (SMOOTHING_LAMBDA * last_value) + ((1.0 - SMOOTHING_LAMBDA) * clamped)
            final_pred = max(final_pred, 0.0)

            preds.append(final_pred)
            temp_values.append(final_pred)
            last_value = final_pred

        if mapped_category not in forecast_out:
            forecast_out[mapped_category] = [
                {"month": months[i], "amount": round(preds[i], 2)}
                for i in range(horizon)
            ]
        else:
            for i in range(horizon):
                forecast_out[mapped_category][i]["amount"] = round(
                    forecast_out[mapped_category][i]["amount"] + preds[i],
                    2,
                )

        if total_values is None:
            total_values = preds.copy()
        else:
            total_values = [total_values[i] + preds[i] for i in range(horizon)]

    total_forecast = []

    if total_values is not None and forecast_out:
        first_category = next(iter(forecast_out.values()))

        for i in range(horizon):
            total_forecast.append(
                {
                    "month": first_category[i]["month"],
                    "amount": round(total_values[i], 2),
                }
            )

    return {
        "forecast": forecast_out,
        "total_forecast": total_forecast,
        "model_version": MODEL_VERSION,
        "meta": {
            "category_grouping": {
                "Drinks": "Food",
                "Groceries": "Food",
                "Electronics": "Shopping",
                "Subscriptions": "Bills",
                "Gym": "Bills",
                "Receipt": "Bills",
                "Rent": "Bills",
                "Travel": "Transport",
            },
            "guardrails": {
                "max_mom_change": MAX_MOM_CHANGE,
                "smoothing_lambda": SMOOTHING_LAMBDA,
            },
        },
    }


def create_forecast_sample_artifacts(model: Any, category_mapping: Dict[str, int]) -> None:
    sample_series = {
        "Food": [
            {"month": "2026-01", "amount": 4000},
            {"month": "2026-02", "amount": 4200},
            {"month": "2026-03", "amount": 3900},
            {"month": "2026-04", "amount": 4100},
        ],
        "Drinks": [
            {"month": "2026-01", "amount": 800},
            {"month": "2026-02", "amount": 900},
            {"month": "2026-03", "amount": 850},
            {"month": "2026-04", "amount": 950},
        ],
        "Electronics": [
            {"month": "2026-01", "amount": 4000},
            {"month": "2026-02", "amount": 4200},
            {"month": "2026-03", "amount": 3900},
            {"month": "2026-04", "amount": 4100},
        ],
        "Subscriptions": [
            {"month": "2026-01", "amount": 300},
            {"month": "2026-02", "amount": 300},
            {"month": "2026-03", "amount": 300},
            {"month": "2026-04", "amount": 300},
        ],
    }

    result = forecast_sample(model, category_mapping, sample_series, horizon=3)

    save_json(result, ARTIFACTS_DIR / "forecast_sample_response.json")

    rows = []

    for item in result["total_forecast"]:
        month = item["month"]
        row = {
            "month": month,
            "total": item["amount"],
        }

        for category, values in result["forecast"].items():
            match = next((x for x in values if x["month"] == month), None)
            row[category] = match["amount"] if match else 0.0

        rows.append(row)

    forecast_df = pd.DataFrame(rows)
    forecast_df.to_csv(ARTIFACTS_DIR / "forecast_sample_table.csv", index=False)

    plt.figure(figsize=(10, 5))

    for column in forecast_df.columns:
        if column != "month":
            plt.plot(forecast_df["month"], forecast_df[column], marker="o", label=column)

    plt.title("Forecast Sample Output")
    plt.xlabel("Month")
    plt.ylabel("Forecasted Amount")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "forecast_sample_chart.png", dpi=200)
    plt.close()


def create_saving_plan_artifacts() -> None:
    plan_type_matrix = pd.DataFrame(
        [
            {
                "Plan Type": "Easy",
                "Flexible Categories Reduction": "8% - 10%",
                "Essential Categories Reduction": "1% - 3%",
                "Difficulty": "Low",
            },
            {
                "Plan Type": "Balanced",
                "Flexible Categories Reduction": "15% - 20%",
                "Essential Categories Reduction": "2.5% - 5%",
                "Difficulty": "Medium",
            },
            {
                "Plan Type": "Aggressive",
                "Flexible Categories Reduction": "27% - 35%",
                "Essential Categories Reduction": "4% - 8%",
                "Difficulty": "High",
            },
        ]
    )

    status_matrix = pd.DataFrame(
        [
            {"Condition": "No valid monthly summary", "Output Status": "NotEnoughData"},
            {"Condition": "Income missing or zero", "Output Status": "MissingIncomeData"},
            {"Condition": "Expenses higher than income", "Output Status": "Critical"},
            {"Condition": "Target saving is within safe capacity", "Output Status": "Realistic"},
            {"Condition": "Target saving is close to maximum possible reduction", "Output Status": "Hard"},
            {"Condition": "Target saving exceeds safe opportunity", "Output Status": "Unrealistic"},
        ]
    )

    why_not_ml = pd.DataFrame(
        [
            {
                "Reason": "No labeled dataset",
                "Explanation": "There are no real labels for ideal recommended budgets.",
            },
            {
                "Reason": "Financial explainability required",
                "Explanation": "Recommendations must be clear and explainable.",
            },
            {
                "Reason": "Safety constraints are mandatory",
                "Explanation": "Essential categories should not be aggressively reduced.",
            },
            {
                "Reason": "Business rules are clearer",
                "Explanation": "Realistic, Hard, Critical, and Unrealistic states must be deterministic.",
            },
            {
                "Reason": "Forecasting already uses ML",
                "Explanation": "Saving Plan uses ML forecast output plus financial rules and guardrails.",
            },
        ]
    )

    plan_type_matrix.to_csv(ARTIFACTS_DIR / "saving_plan_type_matrix.csv", index=False)
    status_matrix.to_csv(ARTIFACTS_DIR / "saving_plan_status_matrix.csv", index=False)
    why_not_ml.to_csv(ARTIFACTS_DIR / "why_saving_plan_not_ml.csv", index=False)


def create_mermaid_diagrams() -> None:
    forecasting_pipeline = """
flowchart LR
    A[Raw Monthly Dataset] --> B[Normalize Backend Categories]
    B --> C[Group Detailed Categories into AI Categories]
    C --> D[Cap Extreme Outliers]
    D --> E[Fill Missing Months per User-Category]
    E --> F[Create Lag Features]
    F --> G[Train RandomForest Regression Model]
    G --> H[Evaluate with MAE RMSE R2]
    H --> I[Save global_expense_model.pkl]
    I --> J[Forecast API]
"""

    saving_plan_flow = """
flowchart TD
    A[Frontend Request] --> B[Backend API]
    B --> C[Get UserId from JWT]
    C --> D[Load Transactions from Database]
    D --> E[Build monthlySummary]
    D --> F[Build categorySummary]
    E --> G[AI Saving Plan API]
    F --> G
    G --> H[Spending Analysis]
    H --> I[Forecast Next Month]
    I --> J[Apply Saving Plan Rules]
    J --> K[Generate Recommendations]
    K --> L[Backend Validates Response]
    L --> M[Frontend Displays Cards Charts Insights]
"""

    ai_architecture = """
flowchart LR
    FE[Frontend] --> BE[Backend]
    BE --> DB[(Database)]
    BE --> AI[Finexa AI Service]
    AI --> FC[Forecasting Engine]
    AI --> SP[Saving Plan Advisor]
    FC --> ML[(Trained Forecast Model)]
    SP --> Rules[Financial Rules and Guardrails]
    AI --> BE
    BE --> FE
"""

    category_grouping = """
flowchart TD
    A[Backend Categories] --> B[AI Canonical Categories]
    A1[Drinks] --> F[Food]
    A2[Groceries] --> F
    A3[Food] --> F

    A4[Electronics] --> S[Shopping]
    A5[Shopping] --> S

    A6[Subscriptions] --> BL[Bills]
    A7[Gym] --> BL
    A8[Receipt] --> BL
    A9[Rent] --> BL
    A10[Bills] --> BL

    A11[Travel] --> T[Transport]
    A12[Transport] --> T

    A13[Other Expense] --> O[Other Expense]
"""

    (DIAGRAMS_DIR / "forecasting_pipeline.mmd").write_text(forecasting_pipeline.strip(), encoding="utf-8")
    (DIAGRAMS_DIR / "saving_plan_flow.mmd").write_text(saving_plan_flow.strip(), encoding="utf-8")
    (DIAGRAMS_DIR / "ai_service_architecture.mmd").write_text(ai_architecture.strip(), encoding="utf-8")
    (DIAGRAMS_DIR / "category_grouping.mmd").write_text(category_grouping.strip(), encoding="utf-8")


def train_model() -> None:
    print("Loading dataset...")
    print(f"Dataset path: {DATA_PATH}")
    print()

    raw_df = load_raw_data()

    print("Raw category counts after normalization:")
    print(raw_df["category"].value_counts().sort_values(ascending=False))
    print()

    print("Capping extreme outliers...")
    capped_df = cap_outliers_per_category(raw_df)

    print("Amount statistics after capping:")
    print(capped_df.groupby("category")["amount"].describe()[["count", "mean", "50%", "max"]])
    print()

    print("Filling missing months per user/category...")
    monthly_df = fill_missing_months_per_user_category(capped_df)

    print("Monthly rows after user-category preparation:")
    print(len(monthly_df))
    print()

    print("User-category series count:")
    print(monthly_df.groupby(["user_id", "category"]).ngroups)
    print()

    print("Category rows after preparation:")
    print(monthly_df["category"].value_counts().sort_values(ascending=False))
    print()

    print("Building lag features...")
    feature_df = build_features(monthly_df)

    category_mapping = build_category_mapping()
    feature_df["category_code"] = feature_df["category"].map(category_mapping)

    missing_codes = feature_df[feature_df["category_code"].isna()]["category"].unique().tolist()

    if missing_codes:
        raise RuntimeError(f"Categories missing from mapping: {missing_codes}")

    train_df, test_df = time_based_split(feature_df)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["amount"]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["amount"]

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Supported categories in model: {list(category_mapping.keys())}")
    print()

    missing_training_categories = [
        category
        for category in SUPPORTED_CATEGORIES
        if category not in feature_df["category"].unique().tolist()
    ]

    if missing_training_categories:
        print("Warning: these supported categories have no training samples:")
        print(missing_training_categories)
        print()

    base_model = RandomForestRegressor(
        n_estimators=350,
        max_depth=14,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model = TransformedTargetRegressor(
        regressor=base_model,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )

    print("Training RandomForest model with log1p target transform...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    predictions = np.maximum(predictions, 0.0)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, predictions) if len(y_test) > 1 else 0.0

    metrics = {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
        "test_rows": int(len(X_test)),
    }

    print("Evaluation metrics:")
    print(metrics)
    print()

    bundle = {
        "model": model,
        "category_mapping": category_mapping,
        "feature_names": FEATURE_COLUMNS,
        "model_version": MODEL_VERSION,
        "supported_categories": list(category_mapping.keys()),
        "category_grouping": {
            "Drinks": "Food",
            "Groceries": "Food",
            "Electronics": "Shopping",
            "Subscriptions": "Bills",
            "Gym": "Bills",
            "Receipt": "Bills",
            "Rent": "Bills",
            "Travel": "Transport",
            "Other": "Other Expense",
            "Others": "Other Expense",
        },
        "training_meta": {
            "dataset_path": str(DATA_PATH),
            "raw_rows_after_preprocessing": int(len(raw_df)),
            "raw_rows_after_capping": int(len(capped_df)),
            "monthly_rows_after_filling": int(len(monthly_df)),
            "feature_rows": int(len(feature_df)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "user_category_series": int(monthly_df.groupby(["user_id", "category"]).ngroups),
            "target_transform": "log1p/expm1",
            "outlier_cap": "per-category max(q95, median*4)",
            "categories_after_normalization": (
                capped_df["category"].value_counts().sort_values(ascending=False).to_dict()
            ),
            "missing_training_categories": missing_training_categories,
        },
        "metrics": metrics,
    }

    joblib.dump(bundle, MODEL_PATH)

    print(f"Model trained and saved successfully to: {MODEL_PATH}")
    print(f"Model version: {MODEL_VERSION}")
    print()

    print("Creating presentation artifacts...")

    create_dataset_artifacts(
        raw_df=raw_df,
        capped_df=capped_df,
        monthly_df=monthly_df,
        feature_df=feature_df,
    )

    create_metrics_artifacts(
        metrics=metrics,
        train_rows=len(X_train),
        test_rows=len(X_test),
    )

    create_actual_vs_predicted_artifacts(
        test_df=test_df,
        y_test=y_test,
        predictions=predictions,
    )

    create_forecast_sample_artifacts(
        model=model,
        category_mapping=category_mapping,
    )

    create_saving_plan_artifacts()
    create_mermaid_diagrams()

    print(f"Presentation artifacts saved to: {ARTIFACTS_DIR}")
    print()
    print("Generated files:")
    for path in sorted(ARTIFACTS_DIR.rglob("*")):
        if path.is_file():
            print(f"- {path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    train_model()