"""Train a baseline and two stronger price models, compare them, and save the best.

Run from the project root:
    python -m src.train
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "flipkart_clean.csv"
MODEL_DIR = ROOT / "models"
SEED = 42

TARGET = "original_price"
CATEGORICAL = ["brand", "processor_vendor"]
NUMERIC = [
    "memory", "storage", "display_size", "battery_capacity",
    "rear_main_mp", "rear_lens_count", "front_mp", "front_camera_missing",
]


def build_model(regressor, scale_numeric: bool):
    """One-hot encode categories, optionally scale numbers, and model log(price)."""
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", numeric_step, NUMERIC),
    ])
    pipeline = Pipeline([("prep", preprocess), ("model", regressor)])
    # Prices are right-skewed, so learn log(price) and convert back to rupees.
    return TransformedTargetRegressor(
        regressor=pipeline, func=np.log1p, inverse_func=np.expm1
    )


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df[CATEGORICAL + NUMERIC]
    y = df[TARGET]
    groups = df["model"]

    # Split by phone model so the same phone is never in both train and test.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    g_train = groups.iloc[train_idx]

    print(f"Train: {len(X_train)} rows, {g_train.nunique()} models")
    print(f"Test:  {len(X_test)} rows, {groups.iloc[test_idx].nunique()} models\n")

    candidates = {
        "Linear regression (baseline)": build_model(LinearRegression(), scale_numeric=True),
        "Random forest": build_model(
            RandomForestRegressor(n_estimators=300, random_state=SEED), scale_numeric=False
        ),
        "Gradient boosting": build_model(
            GradientBoostingRegressor(random_state=SEED), scale_numeric=False
        ),
    }

    # Choose the model with cross-validation on the training set only.
    # The test set is touched once, at the end.
    cv = GroupKFold(n_splits=5)
    rows, fitted = [], {}
    for name, model in candidates.items():
        cv_rmse = -cross_val_score(
            model, X_train, y_train, groups=g_train, cv=cv,
            scoring="neg_root_mean_squared_error",
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "model": name,
            "CV RMSE (INR)": cv_rmse.mean(),
            "CV RMSE std": cv_rmse.std(),
            "Test RMSE (INR)": np.sqrt(mean_squared_error(y_test, pred)),
            "Test MAE (INR)": mean_absolute_error(y_test, pred),
            "Test R2": r2_score(y_test, pred),
        })
        fitted[name] = model

    results = pd.DataFrame(rows).round(2)
    print(results.to_string(index=False))

    best_name = results.sort_values("CV RMSE (INR)").iloc[0]["model"]
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(fitted[best_name], MODEL_DIR / "price_model.joblib")
    results.to_csv(MODEL_DIR / "metrics.csv", index=False)
    print(f"\nBest by cross-validation: {best_name}")
    print(f"Saved model to {MODEL_DIR / 'price_model.joblib'}")
    print(f"Saved metrics to {MODEL_DIR / 'metrics.csv'}")


if __name__ == "__main__":
    main()
