import numpy as np

def mean_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    mu = np.mean(history, axis=0)
    return np.tile(mu.reshape(1, -1), (horizon, 1)).astype(np.float32)

def ets_forecast(history: np.ndarray, horizon: int,
                 seasonal_periods: int = 7,
                 use_seasonal: bool = False) -> np.ndarray:
    """ETS baseline using statsmodels Holt-Winters family."""
    history = np.asarray(history, dtype=float)
    if history.ndim == 1:
        history = history[:, None]
    T, V = history.shape
    out = np.zeros((horizon, V), dtype=np.float32)

    def _fallback(j: int):
        mu = float(np.mean(history[:, j])) if T else 0.0
        out[:, j] = np.full((horizon,), mu, dtype=np.float32)

    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
    except Exception:
        for j in range(V):
            _fallback(j)
        return out

    for j in range(V):
        y = history[:, j]
        if T < 3:
            _fallback(j)
            continue
        try:
            if use_seasonal and (T >= 2 * seasonal_periods):
                mod = ExponentialSmoothing(
                    y,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=seasonal_periods,
                    initialization_method="estimated",
                )
                res = mod.fit(optimized=True)
                fc = res.forecast(horizon)
            else:
                res = SimpleExpSmoothing(y, initialization_method="estimated").fit(optimized=True)
                fc = res.forecast(horizon)
            out[:, j] = np.asarray(fc, dtype=np.float32).reshape(-1)
        except Exception:
            _fallback(j)
    return out

def baseline_forecast(history: np.ndarray, horizon: int, baseline: str = "ETS",
                      ets_use_seasonal: bool = False, ets_seasonal_periods: int = 7) -> np.ndarray:
    baseline = baseline.upper()
    if baseline == "MEAN":
        return mean_forecast(history, horizon)
    if baseline == "ETS":
        return ets_forecast(history, horizon, seasonal_periods=ets_seasonal_periods, use_seasonal=ets_use_seasonal)
    raise ValueError("baseline must be 'MEAN' or 'ETS'")

def baseline_sigma(history: np.ndarray) -> np.ndarray:
    h = np.asarray(history, dtype=float)
    mu = np.mean(h, axis=0, keepdims=True)
    return np.std(h - mu, axis=0).astype(np.float32)
