import numpy as np
import pandas as pd

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(200.0 * np.mean(np.abs(y_true - y_pred) / denom))

def wape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sum(np.abs(y_true - y_pred)) / np.maximum(np.sum(np.abs(y_true)), eps))

def mase_scale(y_insample: np.ndarray, m: int = 1, eps: float = 1e-8) -> np.ndarray:
    y = np.asarray(y_insample, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if y.shape[0] <= m:
        return np.full(y.shape[1], np.nan, dtype=float)
    diff = np.abs(y[m:] - y[:-m])
    s = np.mean(diff, axis=0)
    s = np.maximum(s, eps)
    return s

def evaluate_rolling(
    y_true: np.ndarray,   # (O, H, V)
    y_pred: np.ndarray,   # (O, H, V)
    model_name: str,
    var_names=("Yd", "TSE", "TSEC"),
    y_insample: np.ndarray | None = None,
    mase_m: int = 1,
) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    O, H, V = y_true.shape

    scale = None
    if y_insample is not None:
        scale = mase_scale(y_insample, m=mase_m)

    def _mase(_mae: float, j: int) -> float:
        if scale is None or (not np.isfinite(scale[j])):
            return np.nan
        return float(_mae / scale[j])

    rows = []
    yt_all = y_true.reshape(-1, V)
    yp_all = y_pred.reshape(-1, V)

    for j, v in enumerate(var_names):
        _mae = mae(yt_all[:, j], yp_all[:, j])
        rows.append({
            "model": model_name,
            "variable": v,
            "bucket": "ALL",
            "MAE": _mae,
            "RMSE": rmse(yt_all[:, j], yp_all[:, j]),
            "sMAPE": smape(yt_all[:, j], yp_all[:, j]),
            "WAPE": wape(yt_all[:, j], yp_all[:, j]),
            "MASE": _mase(_mae, j),
        })

    for h in range(H):
        for j, v in enumerate(var_names):
            _mae = mae(y_true[:, h, j], y_pred[:, h, j])
            rows.append({
                "model": model_name,
                "variable": v,
                "bucket": f"step_{h+1}",
                "MAE": _mae,
                "RMSE": rmse(y_true[:, h, j], y_pred[:, h, j]),
                "sMAPE": smape(y_true[:, h, j], y_pred[:, h, j]),
                "WAPE": wape(y_true[:, h, j], y_pred[:, h, j]),
                "MASE": _mase(_mae, j),
            })

    return pd.DataFrame(rows)
