from __future__ import annotations

from pathlib import Path
import random

import pandas as pd
import numpy as np

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "all_users_monthly_data.csv"
OUTPUT_PATH = ROOT_DIR / "all_users_monthly_data_with_others.csv"

FLEX_CATEGORIES = {"Food", "Shopping", "Entertainment", "Transport"}

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_cols = {"month", "category", "amount"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.to_period("M").astype(str)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["month", "category", "amount"])
    df = df[df["amount"] >= 0]

    # لو Others موجودة أصلاً، نشيلها ونعيد توليدها بشكل منظم
    df = df[df["category"].astype(str).str.strip().str.lower() != "others"].copy()

    monthly = (
        df[df["category"].isin(FLEX_CATEGORIES)]
        .groupby("month", as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "flex_total"})
    )

    others_rows = []
    for _, row in monthly.iterrows():
        month = row["month"]
        flex_total = float(row["flex_total"])

        # Others = 4% -> 12% من الإنفاق المرن تقريبًا
        ratio = random.uniform(0.04, 0.12)
        amount = round(max(50.0, flex_total * ratio), 2)

        others_rows.append({
            "month": month,
            "category": "Others",
            "amount": amount
        })

    others_df = pd.DataFrame(others_rows)

    final_df = pd.concat([df, others_df], ignore_index=True)
    final_df = final_df.sort_values(["category", "month"]).reset_index(drop=True)

    final_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved augmented dataset to: {OUTPUT_PATH}")
    print("Category counts:")
    print(final_df.groupby("category")["month"].count().sort_values(ascending=False))

if __name__ == "__main__":
    main()