"""Load the raw Flipkart CSV and print a data-quality report.

Run from the project root:
    python -m src.inspect_data
"""
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "flipkart_smartphones.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    print("=== Shape ===")
    print(f"{df.shape[0]} rows, {df.shape[1]} columns\n")

    print("=== Column types ===")
    print(df.dtypes, "\n")

    print("=== Missing values per column ===")
    print(df.isna().sum(), "\n")

    print(f"=== Exact duplicate rows: {df.duplicated().sum()} ===\n")
    print(f"=== Duplicate brand+model pairs: {df.duplicated(subset=['brand', 'model']).sum()} ===\n")

    print("=== Price summary (INR) ===")
    print(df[["original_price", "discounted_price"]].describe().round(0), "\n")

    print("=== Sample values from text columns ===")
    for col in ["processor", "rear_camera", "front_camera", "battery_type"]:
        print(f"\n{col}:")
        print(df[col].dropna().sample(5, random_state=42).to_string(index=False))


if __name__ == "__main__":
    main()