import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

def make_time_features(dates: pd.DatetimeIndex) -> np.ndarray:
    """Date-only features (known for future dates)."""
    dow = dates.dayofweek.values.astype(np.float32)
    doy = dates.dayofyear.values.astype(np.float32)
    n = len(dates)

    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)
    doy_sin = np.sin(2 * np.pi * (doy - 1.0) / 365.25)
    doy_cos = np.cos(2 * np.pi * (doy - 1.0) / 365.25)
    trend = (np.arange(n, dtype=np.float32) / max(n - 1, 1)).astype(np.float32)

    return np.column_stack([dow_sin, dow_cos, doy_sin, doy_cos, trend]).astype(np.float32)

def enumerate_cases(M_list: List[int], yd_levels: List[str], solar_levels: List[str]) -> List[Dict]:
    """Full Cartesian product: 7 M values * 2 Yd levels * 2 solar levels = 28 cases."""
    cases = []
    for M in M_list:
        for ylv in yd_levels:
            for slv in solar_levels:
                case_id = f"M{M}_Y{ylv}_S{slv}"
                cases.append({"case_id": case_id, "M": M, "yd_level": ylv, "solar_level": slv})
    return cases

def dist_params(M: int, yd_level: str, solar_level: str) -> Dict[str, Tuple[float, float]]:
    """Return (mean, std) for each variable according to the experimental setting."""
    mu_y = 150.0 * M
    mu_s = 150.0 * M
    mu_c = 10.0 * M

    yd_level = yd_level.upper()
    solar_level = solar_level.upper()

    sd_y = 15.0 * M if yd_level == "HUR" else 1.5 * M
    if solar_level == "HUR":
        sd_s = 15.0 * M
        sd_c = 1.0 * M
    else:
        sd_s = 1.5 * M
        sd_c = 0.1 * M

    return {"Yd": (mu_y, sd_y), "TSE": (mu_s, sd_s), "TSEC": (mu_c, sd_c)}

def generate_case_series(
    start_date: str,
    history_days: int,
    test_days: int,
    random_seed: int,
    M: int,
    yd_level: str,
    solar_level: str,
) -> pd.DataFrame:
    """Generate one case: i.i.d. normal per day (clipped at 0)."""
    params = dist_params(M, yd_level, solar_level)
    total_days = history_days + test_days
    dates = pd.date_range(start=start_date, periods=total_days, freq="D")

    # deterministic per case
    seed = int(random_seed) + int(M) * 100 + (0 if yd_level.upper() == "HUR" else 1) * 10 + (0 if solar_level.upper() == "HUR" else 1)
    rng = np.random.default_rng(seed)

    Yd = rng.normal(params["Yd"][0], params["Yd"][1], size=total_days).astype(np.float32)
    TSE = rng.normal(params["TSE"][0], params["TSE"][1], size=total_days).astype(np.float32)
    TSEC = rng.normal(params["TSEC"][0], params["TSEC"][1], size=total_days).astype(np.float32)

    Yd = np.maximum(Yd, 0.0)
    TSE = np.maximum(TSE, 0.0)
    TSEC = np.maximum(TSEC, 0.0)

    split = np.array(["history"] * history_days + ["test"] * test_days)
    return pd.DataFrame({"date": dates, "Yd": Yd, "TSE": TSE, "TSEC": TSEC, "split": split})

def save_case_meta(case_dir: str, case: Dict):
    import os
    os.makedirs(case_dir, exist_ok=True)
    with open(os.path.join(case_dir, "case_meta.json"), "w", encoding="utf-8") as f:
        json.dump(case, f, indent=2)
