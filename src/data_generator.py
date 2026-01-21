"""Generate synthetic trading data with predictive signals."""
import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_data(
    n_days: int = 1000, n_signals: int = 5, seed: int = 42,
    output_path: str = "data/synthetic_data.csv"
) -> pd.DataFrame:
    """Generate synthetic price data with noisy predictive signals."""
    np.random.seed(seed)
    trend = np.cumsum(np.random.randn(n_days + 10) * 0.02)

    # Price using GBM with trend influence (trend leads by ~10 days)
    returns = 0.0005 + 0.3 * trend[10:] - 0.3 * trend[:-10] + np.random.randn(n_days) * 0.02
    prices = 100.0 * np.exp(np.cumsum(returns))

    # Generate signals with varying quality
    signals = []
    for quality in np.linspace(0.3, 0.7, n_signals):
        signal = np.tanh(trend[5:-5] + np.random.randn(n_days) * (1 - quality) * 2)
        signals.append(signal)

    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n_days, freq="B"), "price": prices})
    for i, signal in enumerate(signals):
        df[f"signal_{i+1}"] = signal

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {n_days} days with {n_signals} signals -> {output_path}")
    return df


if __name__ == "__main__":
    generate_synthetic_data()
