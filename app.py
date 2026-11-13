"""Streamlit app: estimate a smartphone's price in INR from its specs.

Run from the project root:
    streamlit run app.py

Prices come from Flipkart listings scraped on 26 February 2023.
They will not match today's market.
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "price_model.joblib"

CATEGORICAL = ["brand", "processor_vendor"]
NUMERIC = [
    "memory", "storage", "display_size", "battery_capacity",
    "rear_main_mp", "rear_lens_count", "front_mp", "front_camera_missing",
]
# Median front camera (MP) from the cleaning step, used when the user doesn't know it.
MEDIAN_FRONT_MP = 12.0
# Price band for the value verdict: within this share of the estimate counts as fair.
FAIR_BAND = 0.10


@st.cache_resource
def load_model():
    """Load the trained model and read its category options from the encoder."""
    model = joblib.load(MODEL_PATH)
    encoder = model.regressor_.named_steps["prep"].named_transformers_["cat"]
    brands, vendors = encoder.categories_
    return model, list(brands), list(vendors)


def verdict(listed_price: float, estimate: float) -> str:
    ratio = listed_price / estimate
    gap = abs(ratio - 1) * 100
    if ratio < 1 - FAIR_BAND:
        return f"Looks **good value**: the listed price is about {gap:.0f}% below the estimate."
    if ratio > 1 + FAIR_BAND:
        return f"Looks **poor value**: the listed price is about {gap:.0f}% above the estimate."
    return f"Looks **fair**: the listed price is within {FAIR_BAND:.0%} of the estimate."


def main() -> None:
    st.set_page_config(page_title="Smartphone Price Estimator", page_icon="📱")
    st.title("Smartphone Price Estimator")
    st.caption(
        "Estimates a phone's price in INR from its specs. Trained on Flipkart listings "
        "scraped in February 2023, so it will not match today's prices."
    )

    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}. Run `python -m src.train` first.")
        st.stop()

    model, brands, vendors = load_model()

    st.subheader("Phone specs")
    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Brand", brands)
        processor_vendor = st.selectbox("Processor vendor", vendors)
        memory = st.selectbox("RAM (GB)", [2, 3, 4, 6, 8, 12, 16], index=2)
        storage = st.selectbox("Storage (GB)", [16, 32, 64, 128, 256, 512], index=2)
        display_size = st.number_input("Display size (inches)", 4.0, 8.0, 6.5, step=0.1)
    with col2:
        battery_capacity = st.number_input("Battery (mAh)", 1000.0, 10000.0, 5000.0, step=100.0)
        rear_main_mp = st.number_input("Main rear camera (MP)", 2.0, 200.0, 50.0, step=1.0)
        rear_lens_count = st.number_input("Number of rear lenses", 1, 4, 2)
        unknown_front = st.checkbox("I don't know the front camera")
        front_mp = st.number_input("Front camera (MP)", 2.0, 60.0, 16.0, step=1.0,
                                   disabled=unknown_front)

    st.subheader("Optional: check a listing")
    listed_price = st.number_input(
        "Listed price in INR (leave at 0 to skip)", 0.0, 1_000_000.0, 0.0, step=500.0
    )

    if st.button("Estimate price", type="primary"):
        row = {
            "brand": brand,
            "processor_vendor": processor_vendor,
            "memory": float(memory),
            "storage": float(storage),
            "display_size": float(display_size),
            "battery_capacity": float(battery_capacity),
            "rear_main_mp": float(rear_main_mp),
            "rear_lens_count": float(rear_lens_count),
            "front_mp": MEDIAN_FRONT_MP if unknown_front else float(front_mp),
            "front_camera_missing": 1 if unknown_front else 0,
        }
        X = pd.DataFrame([row])[CATEGORICAL + NUMERIC]
        estimate = float(model.predict(X)[0])

        st.metric("Estimated price", f"₹{estimate:,.0f}")
        if listed_price > 0:
            st.write(verdict(listed_price, estimate))
        st.caption(
            "The estimate is a model's guess from specs alone. It is not a valuation. "
            "The value verdict is a rough comparison, not financial advice."
        )


if __name__ == "__main__":
    main()
