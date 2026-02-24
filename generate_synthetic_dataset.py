import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import random

np.random.seed(42)
random.seed(42)

NUM_USERS = 120
NUM_MONTHS = 24

CATEGORIES = {
    "Food": {"base": (2500, 6000), "trend": (10, 50)},
    "Transport": {"base": (400, 1200), "trend": (-5, 10)},
    "Entertainment": {"base": (300, 2000), "trend": (0, 25)},
    "Bills": {"base": (800, 3500), "trend": (5, 20)},
    "Shopping": {"base": (500, 4000), "trend": (-10, 30)},
}

START_DATE = datetime(2023, 1, 1)

rows = []

for user_id in range(1, NUM_USERS + 1):

    for category, config in CATEGORIES.items():

        base = random.uniform(*config["base"])
        trend = random.uniform(*config["trend"])
        season_amp = 0.15 * base

        for m in range(NUM_MONTHS):

            date = START_DATE + relativedelta(months=m)
            month = date.strftime("%Y-%m")

            seasonality = season_amp * np.sin(2 * np.pi * m / 12)
            noise = np.random.normal(0, 0.05 * base)

            amount = base + trend * m + seasonality + noise
            amount = max(amount, 0)

            rows.append({
                "user_id": user_id,
                "category": category,
                "month": month,
                "amount": round(amount, 2)
            })

df = pd.DataFrame(rows)
df.to_csv("all_users_monthly_data.csv", index=False)

print("Dataset generated:", len(df), "rows")