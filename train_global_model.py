from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "processed" / "finexa_monthly_category_amount.csv"

MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "global_expense_model.pkl"

MODEL_VERSION = "rf_global_finexa_user_category_v5_log_target"

SUPPORTED_CATEGORIES = [
    "Bills",
    "Entertainment",
    "Food",
    "Shopping",
    "Transport",
    "Health",
    "Education",
    "Others",
]

CATEGORY_ALIASES: Dict[str, str] = {
    "bill": "Bills",
    "bills": "Bills",
    "utility": "Bills",
    "utilities": "Bills",
    "utilties": "Bills",
    "utlities": "Bills",
    "electricity": "Bills",
    "water": "Bills",
    "gas": "Bills",
    "rent": "Bills",
    "rentt": "Bills",
    "rnt": "Bills",
    "housing": "Bills",
    "house": "Bills",
    "home": "Bills",

    "entertainment": "Entertainment",
    "entrtnmnt": "Entertainment",
    "fun": "Entertainment",
    "movies": "Entertainment",
    "movie": "Entertainment",
    "games": "Entertainment",
    "gaming": "Entertainment",

    "food": "Food",
    "foodd": "Food",
    "foods": "Food",
    "fod": "Food",
    "restaurant": "Food",
    "restaurants": "Food",
    "dining": "Food",
    "groceries": "Food",
    "grocery": "Food",

    "shopping": "Shopping",
    "shop": "Shopping",
    "clothes": "Shopping",
    "fashion": "Shopping",
    "accessories": "Shopping",

    "transport": "Transport",
    "transportation": "Transport",
    "uber": "Transport",
    "taxi": "Transport",
    "fuel": "Transport",
    "gasoline": "Transport",
    "travel": "Transport",
    "traval": "Transport",
    "travl": "Transport",

    "health": "Health",
    "helth": "Health",
    "medical": "Health",
    "medicine": "Health",
    "pharmacy": "Health",
    "doctor": "Health",

    "education": "Education",
    "educaton": "Education",
    "school": "Education",
    "course": "Education",
    "courses": "Education",
    "tuition": "Education",

    "other": "Others",
    "others": "Others",
    "misc": "Others",
    "miscellaneous": "Others",
}

EXCLUDED_CATEGORIES = {
    "saving",
    "savings",
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


def normalize_category(value: str) -> Optional[str]:
    raw = str(value).strip()

    if not raw:
        return "Others"

    lower = raw.lower()

    if lower in EXCLUDED_CATEGORIES:
        return None

    if lower in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[lower]

    for category in SUPPORTED_CATEGORIES:
        if category.lower() == lower:
            return category

    return "Others"


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

    df["rolling_mean"] = (
        df.groupby(group_cols)["amount"]
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


def train_model() -> None:
    print("Loading dataset...")
    print(f"Dataset path: {DATA_PATH}")
    print()

    raw_df = load_raw_data()

    print("Raw category counts after normalization:")
    print(raw_df["category"].value_counts().sort_values(ascending=False))
    print()

    print("Capping extreme outliers...")
    raw_df = cap_outliers_per_category(raw_df)

    print("Amount statistics after capping:")
    print(raw_df.groupby("category")["amount"].describe()[["count", "mean", "50%", "max"]])
    print()

    print("Filling missing months per user/category...")
    monthly_df = fill_missing_months_per_user_category(raw_df)

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

    X_test = (
        test_df[FEATURE_COLUMNS]
        if not test_df.empty
        else pd.DataFrame(columns=FEATURE_COLUMNS)
    )

    y_test = (
        test_df["amount"]
        if not test_df.empty
        else pd.Series(dtype=float)
    )

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

    metrics = {}

    if not X_test.empty:
        predictions = model.predict(X_test)
        predictions = np.maximum(predictions, 0.0)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_test, predictions) if len(y_test) > 1 else 0.0

        metrics = {
            "mae": round(float(mae), 4),
            "rmse": round(rmse, 4),
            "r2": round(float(r2), 4),
            "test_rows": int(len(X_test)),
        }

        print("Evaluation metrics:")
        print(metrics)
        print()
    else:
        print("No test set generated. Model trained on all available data.")
        print()

    bundle = {
        "model": model,
        "category_mapping": category_mapping,
        "feature_names": FEATURE_COLUMNS,
        "model_version": MODEL_VERSION,
        "supported_categories": list(category_mapping.keys()),
        "training_meta": {
            "dataset_path": str(DATA_PATH),
            "raw_rows_after_capping": int(len(raw_df)),
            "monthly_rows_after_filling": int(len(monthly_df)),
            "feature_rows": int(len(feature_df)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "user_category_series": int(monthly_df.groupby(["user_id", "category"]).ngroups),
            "target_transform": "log1p/expm1",
            "outlier_cap": "per-category max(q95, median*4)",
            "categories_after_normalization": (
                raw_df["category"].value_counts().sort_values(ascending=False).to_dict()
            ),
            "missing_training_categories": missing_training_categories,
        },
        "metrics": metrics,
    }

    joblib.dump(bundle, MODEL_PATH)

    print(f"✅ Model trained and saved successfully to: {MODEL_PATH}")
    print(f"✅ Model version: {MODEL_VERSION}")


if __name__ == "__main__":
    train_model()