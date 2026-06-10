from pathlib import Path
import pandas as pd


INPUT_PATH = Path(r"U:\finexa_forecasting_ai\budgetwise_finance_dataset.csv")
OUTPUT_PATH = Path(r"U:\finexa_forecasting_ai\data\processed\finexa_monthly_category_amount.csv")


def prepare_monthly_forecasting_dataset(input_path: Path, output_path: Path):
    df = pd.read_csv(input_path)

    df.columns = df.columns.str.strip().str.lower()

    required_columns = [
        "transaction_id",
        "user_id",
        "date",
        "transaction_type",
        "category",
        "amount",
        "payment_mode",
        "location",
        "notes"
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    df = df.dropna(
        subset=["user_id", "date", "transaction_type", "category", "amount"]
    )

    df["transaction_type"] = (
        df["transaction_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["category"] = (
        df["category"]
        .astype(str)
        .str.strip()
        .str.title()
    )
    CATEGORY_MAPPING = {
        "Edu": "Education",
        "Education": "Education",

        "Entertain": "Entertainment",
        "Entertainment": "Entertainment",

        "Food": "Food",
        "Groceries": "Food",
        "Dining": "Food",
        "Restaurant": "Food",

        "Transport": "Transportation",
        "Transportation": "Transportation",
        "Travel": "Transportation",

        "Health": "Health",
        "Healthcare": "Health",
        "Medical": "Health",

        "Shopping": "Shopping",
        "Clothing": "Shopping",

        "Bills": "Bills",
        "Utilities": "Bills",

        "Rent": "Housing",
        "Housing": "Housing",

        "Other": "Other",
        "Misc": "Other",
        "Miscellaneous": "Other",
    }

    df["category"] = df["category"].replace(CATEGORY_MAPPING)

    print("Transaction types:")
    print(df["transaction_type"].value_counts())

    expense_values = ["expense", "debit", "spending", "spent"]
    df = df[df["transaction_type"].isin(expense_values)]

    df["amount"] = df["amount"].abs()
    df["month"] = df["date"].dt.to_period("M").astype(str)

    monthly_df = (
        df.groupby(["user_id", "category", "month"], as_index=False)["amount"]
        .sum()
        .sort_values(["user_id", "category", "month"])
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    monthly_df.to_csv(output_path, index=False)

    return monthly_df


if __name__ == "__main__":
    monthly_df = prepare_monthly_forecasting_dataset(INPUT_PATH, OUTPUT_PATH)

    print("Dataset prepared successfully")
    print(monthly_df.head())
    print(f"Saved to: {OUTPUT_PATH}")