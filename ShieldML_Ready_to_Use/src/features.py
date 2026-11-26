import numpy as np
import pandas as pd

NUMERIC_BASE = ["amount", "hour_sin", "hour_cos", "day_of_week", "merchant_amount_z",
                "customer_tx_count_1h", "customer_tx_count_24h", "customer_tx_count_7d",
                "device_frequency"]

def add_features(df):
    """Past-only rolling features. Input must be in chronological order."""
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"])
    x = x.sort_values("timestamp").reset_index(drop=True)
    hour = x["timestamp"].dt.hour + x["timestamp"].dt.minute / 60
    x["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    x["day_of_week"] = x["timestamp"].dt.dayofweek.astype(float)

    # Merchant amount z-score uses expanding past-only statistics (shifted by one row).
    grouped = x.groupby("merchant", sort=False)["amount"]
    prior_mean = grouped.transform(lambda s: s.shift(1).expanding().mean())
    prior_std = grouped.transform(lambda s: s.shift(1).expanding().std())
    x["merchant_amount_z"] = ((x["amount"] - prior_mean) / prior_std.replace(0, np.nan)).fillna(0)

    # For each transaction, count only earlier events for the same customer.
    x["customer_tx_count_1h"] = 0.0
    x["customer_tx_count_24h"] = 0.0
    x["customer_tx_count_7d"] = 0.0
    x["device_frequency"] = 0.0
    for _, idx in x.groupby("customer_id", sort=False).groups.items():
        times = x.loc[idx, "timestamp"].astype("int64").to_numpy() / 1e9
        counts = []
        for window in (3600, 86400, 604800):
            left = np.searchsorted(times, times - window, side="left")
            pos = np.arange(len(times))
            counts.append((pos - left).astype(float))
        x.loc[idx, "customer_tx_count_1h"] = counts[0]
        x.loc[idx, "customer_tx_count_24h"] = counts[1]
        x.loc[idx, "customer_tx_count_7d"] = counts[2]
    # Past device usage count.
    for _, idx in x.groupby("device_id", sort=False).groups.items():
        x.loc[idx, "device_frequency"] = np.arange(len(idx), dtype=float)

    return x

def model_matrix(featured):
    return featured[NUMERIC_BASE].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
