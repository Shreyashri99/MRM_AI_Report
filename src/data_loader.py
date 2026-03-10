"""
Phase 1 — Data Loader
Loads the UCI Adult Income dataset, a realistic credit/income scoring dataset
with demographic attributes (gender, race) — ideal for bias testing.
"""
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


def load_adult_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads the Adult Income dataset.
    Target: income (0 = <=50K, 1 = >50K)
    Returns: X_train, X_test, y_train, y_test + full raw df
    """
    print("📥  Loading Adult Income dataset (UCI)...")
    adult = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = adult.frame.copy()

    # ── Clean column names ─────────────────────────────────────────────────
    df.columns = df.columns.str.strip().str.lower().str.replace("-", "_")

    # ── Drop rows with missing values ──────────────────────────────────────
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Encode target ──────────────────────────────────────────────────────
    df["income"] = (df["class"].str.strip() == ">50K").astype(int)
    df.drop(columns=["class"], inplace=True)

    # ── Encode protected attributes for readability ────────────────────────
    df["gender"] = df["sex"].str.strip()
    df["race"]   = df["race"].str.strip()

    # ── Select features ────────────────────────────────────────────────────
    feature_cols = [
        "age", "education_num", "hours_per_week",
        "capital_gain", "capital_loss",
        "workclass", "marital_status", "occupation",
        "relationship", "native_country",
    ]

    # Label-encode categoricals for the model
    df_model = df[feature_cols + ["income"]].copy()
    # Ensure all string/category columns are encoded
    for col in df_model.columns:
        if df_model[col].dtype == "object" or str(df_model[col].dtype) == "category":
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))

    X = df_model.drop("income", axis=1)
    y = df_model["income"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"✅  Dataset loaded — Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"    Positive rate (>50K): {y.mean():.1%}")
    return X_train, X_test, y_train, y_test, df


def simulate_production_drift(X_test: pd.DataFrame, drift_intensity: float = 0.3) -> pd.DataFrame:
    """
    Simulates production data drift by shifting feature distributions.
    Mimics real-world scenarios: economic changes, population shift, etc.
    """
    X_drift = X_test.copy()
    rng     = np.random.default_rng(99)

    numeric_cols = X_drift.select_dtypes(include=np.number).columns.tolist()
    n_shift      = max(1, int(len(numeric_cols) * drift_intensity))
    cols_to_shift = rng.choice(numeric_cols, size=n_shift, replace=False)

    for col in cols_to_shift:
        std   = X_drift[col].std()
        shift = rng.uniform(0.5, 1.5) * std          # Random shift
        noise = rng.normal(0, std * 0.1, len(X_drift))
        X_drift[col] = X_drift[col] + shift + noise

    print(f"⚡  Drift simulated on {n_shift} features: {list(cols_to_shift)}")
    return X_drift
