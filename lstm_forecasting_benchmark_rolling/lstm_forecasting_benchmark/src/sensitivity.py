
import numpy as np
import pandas as pd
from typing import Dict

def normal_quantile(p: float) -> float:
    try:
        from scipy.stats import norm
        return float(norm.ppf(p))
    except Exception:
        # Acklam approximation
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
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        if phigh < p:
            q = np.sqrt(-2*np.log(1-p))
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                    ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        q = p - 0.5
        r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

def newsvendor_cost(true_demand, forecast_mean, sigma, c_prod, c_over, c_under) -> float:
    p = c_under / (c_under + c_over)
    z = normal_quantile(p)
    x = forecast_mean + sigma * z
    over = np.maximum(x - true_demand, 0.0)
    under = np.maximum(true_demand - x, 0.0)
    return float(c_prod * np.sum(x) + c_over * np.sum(over) + c_under * np.sum(under))

def run_sensitivity(df_test: pd.DataFrame, forecasts: Dict[str, np.ndarray], sigmas: Dict[str, np.ndarray],
                    c_prod: float, c_over: float, c_under: float) -> pd.DataFrame:
    true_d = df_test["Yd"].values.astype(float)
    rows = []
    for name, fc in forecasts.items():
        mu = fc[:, 0].astype(float)
        sigma = float(sigmas.get(name, np.array([np.std(true_d - mu)]))[0])
        cost = newsvendor_cost(true_d, mu, sigma, c_prod, c_over, c_under)
        rows.append({"model": name, "cost": cost, "sigma_used": sigma})

    perfect = newsvendor_cost(true_d, true_d, 0.0, c_prod, c_over, c_under)
    rows.append({"model": "PERFECT", "cost": perfect, "sigma_used": 0.0})
    df = pd.DataFrame(rows).sort_values("cost")
    df["cost_gap_vs_perfect_%"] = 100.0 * (df["cost"] - perfect) / max(perfect, 1e-9)
    return df
