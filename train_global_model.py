import pandas as pd
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ==============================
# 1️⃣ Load Dataset
# ==============================

df = pd.read_csv("all_users_monthly_data.csv")
df = df.sort_values(["user_id", "category", "month"]).reset_index(drop=True)

# ==============================
# 2️⃣ Feature Engineering
# ==============================

df["lag1"] = df.groupby(["user_id", "category"])["amount"].shift(1)
df["lag2"] = df.groupby(["user_id", "category"])["amount"].shift(2)
df["lag3"] = df.groupby(["user_id", "category"])["amount"].shift(3)

df["rolling_mean"] = (
    df.groupby(["user_id", "category"])["amount"]
    .transform(lambda x: x.rolling(3).mean())
)

df["month_num"] = pd.to_datetime(df["month"]).dt.month

# ==============================
# 3️⃣ Fixed Category Encoding
# ==============================

categories = sorted(df["category"].unique())
category_mapping = {cat: idx for idx, cat in enumerate(categories)}

df["category_code"] = df["category"].map(category_mapping)

df = df.dropna().reset_index(drop=True)

FEATURES = [
    "lag1",
    "lag2",
    "lag3",
    "rolling_mean",
    "month_num",
    "category_code"
]

X = df[FEATURES]
y = df["amount"]

# ==============================
# 4️⃣ Train / Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ==============================
# 5️⃣ Model Training
# ==============================

model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# 6️⃣ Evaluation
# ==============================

pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)

print("===================================")
print("MAE:", mae)
print("===================================")

# ==============================
# 7️⃣ Feature Importance
# ==============================

print("\nFeature Importance:")
for feature, importance in zip(FEATURES, model.feature_importances_):
    print(f"{feature}: {importance}")

plt.figure(figsize=(8, 5))
lgb.plot_importance(model, max_num_features=10)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# ==============================
# 8️⃣ Save Model
# ==============================

joblib.dump({
    "model": model,
    "category_mapping": category_mapping,
    "features": FEATURES
}, "global_expense_model.pkl")

print("\nModel saved successfully.")