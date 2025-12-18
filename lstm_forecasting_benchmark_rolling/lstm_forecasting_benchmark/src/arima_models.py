\
import numpy as np
from typing import Tuple

def arima_forecast_per_series(
    history: np.ndarray,
    horizon: int,
    order: Tuple[int, int, int] = (2, 0, 2),
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 7),
    use_auto_arima: bool = False,
    random_seed: int = 42,
) -> np.ndarray:
    history = np.asarray(history, dtype=float)

    if use_auto_arima:
        try:
            import pmdarima as pm
            model = pm.auto_arima(
                history,
                seasonal=True,
                m=seasonal_order[3],
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                random=True,
                random_state=random_seed,
                n_fits=30,
            )
            return np.asarray(model.predict(n_periods=horizon), dtype=np.float32)
        except Exception:
            pass

    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        import warnings
        try:
            from statsmodels.tools.sm_exceptions import ConvergenceWarning
        except Exception:
            # fallback if statsmodels version differs
            ConvergenceWarning = Warning

        mod = SARIMAX(
            history,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        res = None
        # First fit attempt: suppress ConvergenceWarning to avoid log spam
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            try:
                res = mod.fit(disp=False, maxiter=200)
            except Exception:
                res = None

        # If first fit failed or did not converge, try an alternative optimizer once
        if res is None or (hasattr(res, "mle_retvals") and not res.mle_retvals.get("converged", True)):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                try:
                    res = mod.fit(disp=False, method="powell", maxiter=400)
                except Exception:
                    res = None

        if res is None:
            last = float(history[-1]) if len(history) else 0.0
            return np.full((horizon,), last, dtype=np.float32)

        return np.asarray(res.forecast(steps=horizon), dtype=np.float32)
    except Exception:
        last = float(history[-1]) if len(history) else 0.0
        return np.full((horizon,), last, dtype=np.float32)

def arima_forecast_multivariate(
    history_3d: np.ndarray,
    horizon: int,
    order=(2, 0, 2),
    seasonal_order=(1, 0, 1, 7),
    use_auto_arima=False,
    random_seed: int = 42
) -> np.ndarray:
    outs = []
    for j in range(history_3d.shape[1]):
        fc = arima_forecast_per_series(
            history_3d[:, j],
            horizon=horizon,
            order=order,
            seasonal_order=seasonal_order,
            use_auto_arima=use_auto_arima,
            random_seed=random_seed
        )
        outs.append(fc.reshape(-1, 1))
    return np.concatenate(outs, axis=1).astype(np.float32)
