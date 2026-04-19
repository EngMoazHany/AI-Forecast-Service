from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "all_users_monthly_data_with_others.csv"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "global_expense_model.pkl"

MODEL_VERSION = "rf_global_v2"

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
    "rent": "Bills",
    "electricity": "Bills",
    "water": "Bills",
    "gas": "Bills",

    "entertainment": "Entertainment",
    "fun": "Entertainment",
    "movies": "Entertainment",
    "games": "Entertainment",
    "gaming": "Entertainment",

    "food": "Food",
    "restaurant": "Food",
    "restaurants": "Food",
    "dining": "Food",
    "groceries": "Food",
    "grocery": "Food",

    "shopping": "Shopping",
    "clothes": "Shopping",
    "fashion": "Shopping",

    "transport": "Transport",
    "transportation": "Transport",
    "uber": "Transport",
    "taxi": "Transport",
    "fuel": "Transport",
    "gasoline": "Transport",

    "health": "Health",
    "medical": "Health",
    "medicine": "Health",
    "pharmacy": "Health",
    "doctor": "Health",

    "education": "Education",
    "school": "Education",
    "course": "Education",
    "courses": "Education",
    "tuition": "Education",

    "other": "Others",
    "others": "Others",
    "misc": "Others",
    "miscellaneous": "Others",
}

MIN_HISTORY_PER_CATEGORY = 4


def normalize_category(value: str) -> str:
    raw = str(value).strip()
    if not raw:
        return "Others"

    lower = raw.lower()

    if lower in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[lower]

    for cat in SUPPORTED_CATEGORIES:
        if cat.lower() == lower:
            return cat

    return "Others"


def load_and_prepare_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_cols = {"month", "category", "amount"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    df = df.copy()

    df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.to_period("M").astype(str)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["category"] = df["category"].apply(normalize_category)

    df = df.dropna(subset=["month", "amount", "category"])
    df = df[df["amount"] >= 0].copy()

    df = (
        df.groupby(["category", "month"], as_index=False)["amount"]
        .sum()
        .sort_values(["category", "month"])
        .reset_index(drop=True)
    )

    counts = df.groupby("category")["month"].count()
    valid_categories = counts[counts >= MIN_HISTORY_PER_CATEGORY].index.tolist()

    df = df[df["category"].isin(valid_categories)].copy()

    if df.empty:
        raise RuntimeError("No usable training data after preprocessing.")

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["category", "month"]).reset_index(drop=True)

    df["lag1"] = df.groupby("category")["amount"].shift(1)
    df["lag2"] = df.groupby("category")["amount"].shift(2)
    df["lag3"] = df.groupby("category")["amount"].shift(3)

    df["rolling_mean"] = (
        df.groupby("category")["amount"]
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["month_num"] = pd.to_datetime(df["month"]).dt.month

    df = df.dropna().reset_index(drop=True)

    if df.empty:
        raise RuntimeError("No rows left after lag/rolling feature generation.")

    return df


def build_category_mapping(df: pd.DataFrame) -> Dict[str, int]:
    categories = sorted(df["category"].unique().tolist())
    ordered = [cat for cat in SUPPORTED_CATEGORIES if cat in categories]
    return {cat: idx for idx, cat in enumerate(ordered)}


def time_based_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    for _, grp in df.groupby("category", sort=False):
        grp = grp.sort_values("month").reset_index(drop=True)

        if len(grp) < 5:
            train_parts.append(grp)
            continue

        split_idx = max(3, int(len(grp) * 0.8))
        split_idx = min(split_idx, len(grp) - 1)

        train_parts.append(grp.iloc[:split_idx])
        test_parts.append(grp.iloc[split_idx:])

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()

    return train_df, test_df


def train_model():
    print("Loading dataset...")
    df = load_and_prepare_data()

    print("Category counts after normalization:")
    print(df.groupby("category")["month"].count().sort_values(ascending=False))
    print()

    print("Building lag features...")
    df = build_features(df)

    category_mapping = build_category_mapping(df)
    df["category_code"] = df["category"].map(category_mapping)

    feature_columns = ["lag1", "lag2", "lag3", "rolling_mean", "month_num", "category_code"]

    train_df, test_df = time_based_split(df)

    X_train = train_df[feature_columns]
    y_train = train_df["amount"]

    X_test = test_df[feature_columns] if not test_df.empty else pd.DataFrame(columns=feature_columns)
    y_test = test_df["amount"] if not test_df.empty else pd.Series(dtype=float)

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Supported categories in model: {list(category_mapping.keys())}")
    print()

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    print("Training RandomForest model...")
    model.fit(X_train, y_train)

    metrics = {}
    if not X_test.empty:
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)

        metrics = {
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
            "test_rows": int(len(X_test)),
        }

        print("Evaluation metrics:")
        print(metrics)
        print()

    bundle = {
        "model": model,
        "category_mapping": category_mapping,
        "feature_names": feature_columns,
        "model_version": MODEL_VERSION,
        "supported_categories": list(category_mapping.keys()),
        "metrics": metrics,
    }

    joblib.dump(bundle, MODEL_PATH)
    print(f"✅ Model trained and saved successfully to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()