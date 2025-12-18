import numpy as np
import pandas as pd


def generate_synthetic_series(
    start_date: str,
    history_days: int,
    test_days: int,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Keep your legacy synthetic generator (seasonality + AR + coupling).
    """
    rng = np.random.default_rng(random_seed)

    total_days = history_days + test_days
    dates = pd.date_range(start=start_date, periods=total_days, freq="D")
    t = np.arange(total_days, dtype=np.float32)

    year = np.sin(2 * np.pi * t / 365.25).astype(np.float32)
    week = np.sin(2 * np.pi * t / 7.0).astype(np.float32)
    trend = (t / max(total_days - 1, 1)).astype(np.float32)

    # base levels (legacy)
    Yd_mean, Yd_std = 600.0, 60.0
    TSE_mean, TSE_std = 600.0, 60.0
    TSEC_mean, TSEC_std = 40.0, 4.0

    cov = np.array([
        [1.0, 0.6, 0.3],
        [0.6, 1.0, 0.4],
        [0.3, 0.4, 1.0],
    ], dtype=np.float32)

    eps = rng.multivariate_normal(mean=[0, 0, 0], cov=cov, size=total_days).astype(np.float32)
    phi = 0.7
    e = np.zeros_like(eps, dtype=np.float32)
    for i in range(total_days):
        e[i] = eps[i] if i == 0 else (phi * e[i - 1] + eps[i])

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


def generate_case_series(
    start_date: str,
    history_days: int,
    test_days: int,
    M: int,
    yd_level: str,
    solar_level: str,
    random_seed: int = 42,
    clip_nonnegative: bool = True,
) -> pd.DataFrame:
    """
    28-case generator: 7 (M) × 2 (Yd volatility) × 2 (Solar volatility)

    - Yd volatility controlled by yd_level:
        HUR:  Yd ~ N(150*M, 15*M)
        LUR:  Yd ~ N(150*M, 1.5*M)

    - Solar volatility controlled by solar_level:
        HUR:  TSE  ~ N(150*M, 15*M),   TSEC ~ N(10*M, 1*M)
        LUR:  TSE  ~ N(150*M, 1.5*M),  TSEC ~ N(10*M, 0.1*M)
    """
    yd_level = yd_level.upper().strip()
    solar_level = solar_level.upper().strip()
    if yd_level not in {"HUR", "LUR"}:
        raise ValueError("yd_level must be 'HUR' or 'LUR'")
    if solar_level not in {"HUR", "LUR"}:
        raise ValueError("solar_level must be 'HUR' or 'LUR'")

    rng = np.random.default_rng(random_seed)
    total_days = history_days + test_days
    dates = pd.date_range(start=start_date, periods=total_days, freq="D")

    # Means
    mu_Y = 150.0 * M
    mu_TSE = 150.0 * M
    mu_TSEC = 10.0 * M

    # Std (Yd)
    sd_Y = (15.0 * M) if yd_level == "HUR" else (1.5 * M)

    # Std (Solar: TSE + TSEC)
    if solar_level == "HUR":
        sd_TSE = 15.0 * M
        sd_TSEC = 1.0 * M
    else:
        sd_TSE = 1.5 * M
        sd_TSEC = 0.1 * M

    Yd = rng.normal(mu_Y, sd_Y, size=total_days).astype(np.float32)
    TSE = rng.normal(mu_TSE, sd_TSE, size=total_days).astype(np.float32)
    TSEC = rng.normal(mu_TSEC, sd_TSEC, size=total_days).astype(np.float32)

    if clip_nonnegative:
        Yd = np.maximum(Yd, 0.0)
        TSE = np.maximum(TSE, 0.0)
        TSEC = np.maximum(TSEC, 0.0)

    split = np.array(["history"] * history_days + ["test"] * test_days)
    return pd.DataFrame({
        "date": dates,
        "Yd": Yd,
        "TSE": TSE,
        "TSEC": TSEC,
        "split": split
    })
