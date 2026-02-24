import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# =============================
# Load dataset
# =============================
df = pd.read_csv("all_users_monthly_data.csv")

# =============================
# Feature Engineering
# =============================
df = df.sort_values(["category", "month"])

df["lag1"] = df.groupby("category")["amount"].shift(1)
df["lag2"] = df.groupby("category")["amount"].shift(2)
df["lag3"] = df.groupby("category")["amount"].shift(3)

df["rolling_mean"] = df.groupby("category")["amount"].rolling(3).mean().reset_index(0, drop=True)
df["month_num"] = pd.to_datetime(df["month"]).dt.month

df = df.dropna()

# =============================
# Encode category
# =============================
category_mapping = {
    cat: idx for idx, cat in enumerate(df["category"].unique())
}
df["category_code"] = df["category"].map(category_mapping)

# =============================
# Features / Target
# =============================
features = ["lag1", "lag2", "lag3", "rolling_mean", "month_num", "category_code"]

X = df[features]
y = df["amount"]

# =============================
# Train model
# =============================
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

model.fit(X, y)

# =============================
# Save bundle
# =============================
bundle = {
    "model": model,
    "category_mapping": category_mapping
}

joblib.dump(bundle, "global_expense_model.pkl")

print("✅ RandomForest model trained and saved successfully.")