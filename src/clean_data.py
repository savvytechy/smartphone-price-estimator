"""Clean the raw Flipkart CSV and write data/flipkart_clean.csv.

Run from the project root:
    python -m src.clean_data

Every drop is printed with its row count so the cleaning is auditable.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "flipkart_smartphones.csv"
CLEAN_PATH = ROOT / "data" / "flipkart_clean.csv"

# Order matters only if a processor name contains two keywords (none do in this data).
VENDOR_KEYWORDS = [
    ("snapdragon", "qualcomm"),
    ("helio", "mediatek"),
    ("dimensity", "mediatek"),
    ("mediatek", "mediatek"),
    ("mtk", "mediatek"),
    ("unisoc", "unisoc"),
    ("spreadtrum", "unisoc"),
    ("sc9", "unisoc"),
    ("bionic", "apple"),
    ("exynos", "samsung"),
    ("tensor", "google"),
    ("kirin", "huawei"),
]

# Columns dropped from the model's inputs, with the reason for each.
DROP_COLUMNS = {
    "ratings": "not a spec; unknown for an unlisted phone",
    "rating_count": "not a spec; unknown for an unlisted phone",
    "reviews": "not a spec; unknown for an unlisted phone",
    "colour": "no spec value",
    "battery_type": "43% missing; not worth imputing",
    "processor": "replaced by processor_vendor",
    "rear_camera": "replaced by rear_main_mp and rear_lens_count",
    "front_camera": "replaced by front_mp",
}


def processor_vendor(text):
    if pd.isna(text):
        return "unknown"
    lower = text.lower()
    for keyword, vendor in VENDOR_KEYWORDS:
        if keyword in lower:
            return vendor
    return "other"


def parse_rear_camera(text):
    """Return (main megapixels, number of lenses) from strings like '50MP + 2MP'."""
    if pd.isna(text):
        return np.nan, np.nan
    parts = [p.strip() for p in text.split("+") if p.strip()]
    match = re.search(r"(\d+(?:\.\d+)?)\s*MP", parts[0], re.IGNORECASE)
    main_mp = float(match.group(1)) if match else np.nan
    return main_mp, float(len(parts))


def parse_front_camera(text):
    """Return the front camera's megapixels from strings like ' 16MP' or '60 MP with OIS'."""
    if pd.isna(text):
        return np.nan
    match = re.search(r"(\d+(?:\.\d+)?)\s*MP", text, re.IGNORECASE)
    return float(match.group(1)) if match else np.nan


def main() -> None:
    raw = pd.read_csv(RAW_PATH)
    df = raw.copy()
    print(f"Raw rows:                                  {len(df)}")

    # 1. Exact duplicate rows (scraping repeats).
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped exact duplicate rows:              {n_before - len(df)}")

    # 2. Discount larger than original price: impossible, so the scrape is wrong.
    n_before = len(df)
    df = df[df["discounted_price"] <= df["original_price"]]
    print(f"Dropped discounted > original price:       {n_before - len(df)}")

    # 3. Parse text columns into numbers or categories.
    df["processor_vendor"] = df["processor"].apply(processor_vendor)
    rear = df["rear_camera"].apply(parse_rear_camera)
    df["rear_main_mp"] = [r[0] for r in rear]
    df["rear_lens_count"] = [r[1] for r in rear]
    df["front_mp"] = df["front_camera"].apply(parse_front_camera)

    # Front camera: keep the phone, record that the value was filled, use the median.
    df["front_camera_missing"] = df["front_mp"].isna().astype(int)
    front_median = df["front_mp"].median()
    df["front_mp"] = df["front_mp"].fillna(front_median)
    print(f"Front camera filled with median ({front_median} MP) for "
          f"{df['front_camera_missing'].sum()} phones")

    # 4. Drop rows missing a core spec. Imputing these would invent specs for real phones.
    core = ["memory", "storage", "battery_capacity", "rear_main_mp"]
    n_before = len(df)
    df = df.dropna(subset=core)
    print(f"Dropped rows missing a core spec ({', '.join(core)}): {n_before - len(df)}")

    # 5. Strip stray whitespace from text identifiers.
    for col in ["brand", "model"]:
        df[col] = df[col].str.strip()

    # 6. Drop the columns the app will not use.
    df = df.drop(columns=list(DROP_COLUMNS))

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)

    print(f"\nClean rows:                                {len(df)}")
    print(f"Distinct models:                           {df['model'].nunique()}")
    print(f"Saved to: {CLEAN_PATH}")


if __name__ == "__main__":
    main()
