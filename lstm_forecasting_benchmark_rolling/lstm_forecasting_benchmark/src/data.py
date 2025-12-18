\
import numpy as np
import pandas as pd

def make_time_features(dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Time features derived ONLY from the date (known in the future).
    Returns (N, 5): [dow_sin, dow_cos, doy_sin, doy_cos, trend]
    """
    dow = dates.dayofweek.values.astype(np.float32)
    doy = dates.dayofyear.values.astype(np.float32)
    n = len(dates)

    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)

    doy_sin = np.sin(2 * np.pi * (doy - 1.0) / 365.25)
    doy_cos = np.cos(2 * np.pi * (doy - 1.0) / 365.25)

    trend = (np.arange(n, dtype=np.float32) / max(n - 1, 1)).astype(np.float32)

    return np.column_stack([dow_sin, dow_cos, doy_sin, doy_cos, trend]).astype(np.float32)


def generate_synthetic_series(
    start_date: str,
    history_days: int,
    test_days: int,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Synthetic daily multivariate series with:
      - weekly + annual seasonality
      - mild trend
      - correlated AR(1) noise
      - cross-series coupling
    """
    rng = np.random.default_rng(random_seed)

    total_days = history_days + test_days
    dates = pd.date_range(start=start_date, periods=total_days, freq="D")
    t = np.arange(total_days, dtype=np.float32)

    year = np.sin(2 * np.pi * t / 365.25).astype(np.float32)
    week = np.sin(2 * np.pi * t / 7.0).astype(np.float32)
    trend = (t / max(total_days - 1, 1)).astype(np.float32)

    # base levels
    Yd_mean, Yd_std = 600.0, 60.0
    TSE_mean, TSE_std = 600.0, 60.0
    TSEC_mean, TSEC_std = 40.0, 4.0

    # correlated shocks + AR(1)
    cov = np.array([
        [1.0, 0.6, 0.3],
        [0.6, 1.0, 0.4],
        [0.3, 0.4, 1.0],
    ], dtype=np.float32) * 0.6

    eps = rng.multivariate_normal(mean=[0, 0, 0], cov=cov, size=total_days).astype(np.float32)

    phi = 0.7
    e = np.zeros_like(eps, dtype=np.float32)
    for i in range(1, total_days):
        e[i] = phi * e[i - 1] + eps[i]

    Yd = (Yd_mean
          + 0.30 * Yd_std * year
          + 0.15 * Yd_std * week
          + 0.10 * Yd_std * trend
          + 0.35 * Yd_std * e[:, 0])

    TSE = (TSE_mean
           + 0.25 * TSE_std * year
           - 0.10 * TSE_std * week
           + 0.08 * TSE_std * trend
           + 0.35 * TSE_std * e[:, 1]
           + 0.08 * (Yd - Yd_mean))

    TSEC = (TSEC_mean
            + 0.20 * TSEC_std * year
            + 0.10 * TSEC_std * week
            + 0.05 * TSEC_std * trend
            + 0.35 * TSEC_std * e[:, 2]
            + 0.05 * (TSE - TSE_mean))

    Yd = np.maximum(Yd, 0.0)
    TSE = np.maximum(TSE, 0.0)
    TSEC = np.maximum(TSEC, 0.0)

    split = np.array(["history"] * history_days + ["test"] * test_days)
    return pd.DataFrame({
        "date": dates,
        "Yd": Yd.astype(np.float32),
        "TSE": TSE.astype(np.float32),
        "TSEC": TSEC.astype(np.float32),
        "split": split
    })
