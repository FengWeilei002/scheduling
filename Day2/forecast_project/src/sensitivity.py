import numpy as np
import pandas as pd
from typing import Dict

def normal_quantile(p: float) -> float:
    try:
        from scipy.stats import norm
        return float(norm.ppf(p))
    except Exception:
        # Acklam approximation (sufficient for sensitivity curves)
        a = [-3.969683028665376e+01,  2.209460984245205e+02,
             -2.759285104469687e+02,  1.383577518672690e+02,
             -3.066479806614716e+01,  2.506628277459239e+00]
        b = [-5.447609879822406e+01,  1.615858368580409e+02,
             -1.556989798598866e+02,  6.680131188771972e+01,
             -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01,
             -2.400758277161838e+00, -2.549732539343734e+00,
              4.374664141464968e+00,  2.938163982698783e+00]
        d = [ 7.784695709041462e-03,  3.224671290700398e-01,
              2.445134137142996e+00,  3.754408661907416e+00]
        plow = 0.02425
        phigh = 1 - plow
        if p < plow:
            q = np.sqrt(-2*np.log(p))
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /                    ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        if phigh < p:
            q = np.sqrt(-2*np.log(1-p))
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /                     ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        q = p - 0.5
        r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /                (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

def newsvendor_decision(mu: np.ndarray, sigma: float, c_over: float, c_under: float) -> np.ndarray:
    p = c_under / (c_under + c_over)
    z = normal_quantile(p)
    return mu + sigma * z

def newsvendor_cost_series(true_d: np.ndarray, q: np.ndarray, c_prod: float, c_over: float, c_under: float) -> np.ndarray:
    over = np.maximum(q - true_d, 0.0)
    under = np.maximum(true_d - q, 0.0)
    return c_prod * q + c_over * over + c_under * under

def run_sensitivity_step1(
    true_d: np.ndarray,
    forecast_mu: Dict[str, np.ndarray],
    sigma_used: Dict[str, float],
    c_prod: float, c_over: float, c_under: float
) -> pd.DataFrame:
    true_d = np.asarray(true_d, dtype=float)

    q_perf = true_d.copy()
    cost_perf = newsvendor_cost_series(true_d, q_perf, c_prod, c_over, c_under)
    perfect_total = float(np.sum(cost_perf))
    perfect_mean = float(np.mean(cost_perf))

    rows = []
    for name, mu in forecast_mu.items():
        mu = np.asarray(mu, dtype=float)
        sigma = float(sigma_used.get(name, np.std(true_d - mu)))
        q = newsvendor_decision(mu, sigma, c_over, c_under)
        cost = newsvendor_cost_series(true_d, q, c_prod, c_over, c_under)
        total = float(np.sum(cost))
        mean = float(np.mean(cost))
        rows.append({
            "model": name,
            "sigma_used": sigma,
            "mean_cost": mean,
            "total_cost": total,
            "gap_vs_perfect_%": 100.0 * (total - perfect_total) / max(perfect_total, 1e-9),
        })

    rows.append({
        "model": "PERFECT",
        "sigma_used": 0.0,
        "mean_cost": perfect_mean,
        "total_cost": perfect_total,
        "gap_vs_perfect_%": 0.0,
    })

    return pd.DataFrame(rows).sort_values("total_cost")
