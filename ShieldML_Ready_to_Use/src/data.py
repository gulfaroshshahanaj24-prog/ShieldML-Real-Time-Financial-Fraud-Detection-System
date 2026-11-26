from pathlib import Path
import numpy as np
import pandas as pd

DATA_PATH = Path("data/transactions.csv")

def make_demo_data(n=12000, seed=42):
    """Create deterministic, artificial transactions for a software smoke test."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01")
    seconds = np.sort(rng.integers(0, 60 * 60 * 24 * 30, size=n))
    amount = np.clip(rng.lognormal(mean=3.0, sigma=1.0, size=n), 1, 5000)
    merchant = rng.choice([f"merchant_{i}" for i in range(30)], n)
    device = rng.choice([f"device_{i}" for i in range(250)], n)
    customer = rng.choice([f"customer_{i}" for i in range(1000)], n)
    # Artificial pattern: unusually large transactions and repeated activity are riskier.
    high_amount = amount > np.quantile(amount, .96)
    risk = 0.008 + high_amount * 0.16 + (rng.random(n) < 0.018) * 0.12
    fraud = (rng.random(n) < risk).astype(int)
    return pd.DataFrame({
        "timestamp": start + pd.to_timedelta(seconds, unit="s"),
        "amount": amount,
        "merchant": merchant,
        "device_id": device,
        "customer_id": customer,
        "is_fraud": fraud,
    })

def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        print("data/transactions.csv not found; generating synthetic DEMO data.")
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df = make_demo_data()
        df.to_csv(DATA_PATH, index=False)

    if "Class" in df.columns and "is_fraud" not in df.columns:
        df = df.rename(columns={"Class": "is_fraud"})
    if "Time" in df.columns and "timestamp" not in df.columns:
        # Kaggle creditcard.csv Time is seconds elapsed from dataset start.
        df["timestamp"] = pd.Timestamp("2023-01-01") + pd.to_timedelta(df["Time"], unit="s")
    if "timestamp" not in df.columns:
        raise ValueError("Dataset must contain timestamp or Kaggle-style Time column.")
    if "is_fraud" not in df.columns:
        raise ValueError("Dataset must contain is_fraud or Kaggle-style Class target.")
    if "amount" not in df.columns and "Amount" in df.columns:
        df = df.rename(columns={"Amount": "amount"})
    if "amount" not in df.columns:
        raise ValueError("Dataset must contain amount or Kaggle-style Amount column.")

    for col, default in [("merchant", "unknown_merchant"),
                         ("device_id", "unknown_device"),
                         ("customer_id", "unknown_customer")]:
        if col not in df.columns:
            df[col] = default

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce")
    df = df.dropna(subset=["timestamp", "amount", "is_fraud"]).copy()
    df["is_fraud"] = df["is_fraud"].astype(int)
    df = df[df["is_fraud"].isin([0, 1])].sort_values("timestamp").reset_index(drop=True)
    return df
