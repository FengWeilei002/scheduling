import argparse
import os
import numpy as np
import pandas as pd

from src.config import Config
from src.data import generate_case_series, save_case_meta
from src.baselines import baseline_forecast, baseline_sigma
from src.arima_models import arima_forecast_multivariate, arima_sigma
from src.metrics import evaluate_rolling
from src.deep_models import train_deep_model, deep_forward
from src.sensitivity import run_sensitivity_step1


def rolling_origins(history_days: int, rolling_test_days: int, horizon: int, step: int = 1):
    start = history_days
    end = history_days + rolling_test_days  # exclusive
    return list(range(start, end - horizon + 1, step))


def run_one_case(cfg: Config, case_id: str, M: int, yd_level: str, solar_level: str, out_root: str):
    case_dir = os.path.join(out_root, "cases", case_id)
    os.makedirs(case_dir, exist_ok=True)
    save_case_meta(case_dir, {"case_id": case_id, "M": M, "yd_level": yd_level, "solar_level": solar_level})

    df = generate_case_series(cfg.start_date, cfg.history_days, cfg.rolling_test_days, cfg.random_seed, M, yd_level, solar_level)
    df.to_csv(os.path.join(case_dir, "data.csv"), index=False)

    df_all = df.copy()
    df_hist0 = df_all.iloc[:cfg.history_days].copy()

    var_names = ["Yd", "TSE", "TSEC"]
    y_insample = df_hist0[var_names].values.astype(np.float32)

    # Train deep models once on initial history
    deep_rnn = train_deep_model(
        df_history=df_hist0[["date"] + var_names],
        seq_length=cfg.seq_length,
        horizon=cfg.rolling_horizon,
        train_split=cfg.train_split,
        val_split=cfg.val_split,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        patience=cfg.patience,
        model_type="rnn",
        random_seed=cfg.random_seed,
    )
    deep_lstm = train_deep_model(
        df_history=df_hist0[["date"] + var_names],
        seq_length=cfg.seq_length,
        horizon=cfg.rolling_horizon,
        train_split=cfg.train_split,
        val_split=cfg.val_split,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        patience=cfg.patience,
        model_type="lstm",
        random_seed=cfg.random_seed + 7,
    )

    H = cfg.rolling_horizon
    origins = rolling_origins(cfg.history_days, cfg.rolling_test_days, H, cfg.rolling_step)
    O = len(origins)

    V = len(var_names)
    y_true = np.zeros((O, H, V), dtype=np.float32)
    pred_base = np.zeros((O, H, V), dtype=np.float32)
    pred_arima = np.zeros((O, H, V), dtype=np.float32)
    pred_rnn = np.zeros((O, H, V), dtype=np.float32)
    pred_lstm = np.zeros((O, H, V), dtype=np.float32)

    tidy = []
    step1_rows = []

    for oi, origin in enumerate(origins):
        df_avail = df_all.iloc[:origin].copy()
        df_fut = df_all.iloc[origin:origin + H].copy()

        y_hist = df_avail[var_names].values.astype(np.float32)
        y_true[oi] = df_fut[var_names].values.astype(np.float32)

        pred_base[oi] = baseline_forecast(
            y_hist, H,
            baseline=cfg.baseline,
            ets_use_seasonal=cfg.ets_use_seasonal,
            ets_seasonal_periods=cfg.ets_seasonal_periods,
        )

        pred_arima[oi] = arima_forecast_multivariate(
            y_hist, H,
            order=cfg.arima_order,
            seasonal_order=cfg.seasonal_order,
            use_auto_arima=cfg.use_auto_arima,
            random_seed=cfg.random_seed,
        )

        dates_hist = pd.DatetimeIndex(pd.to_datetime(df_avail["date"]))
        dates_fut = pd.DatetimeIndex(pd.to_datetime(df_fut["date"]))
        pred_rnn[oi] = deep_forward(deep_rnn, y_hist, dates_hist, dates_fut, seq_length=cfg.seq_length)
        pred_lstm[oi] = deep_forward(deep_lstm, y_hist, dates_hist, dates_fut, seq_length=cfg.seq_length)

        step1_rows.append({
            "origin_date": df_all["date"].iloc[origin],
            "true_Yd_step1": float(y_true[oi, 0, 0]),
            f"{cfg.baseline}_Yd_step1": float(pred_base[oi, 0, 0]),
            "ARIMA_Yd_step1": float(pred_arima[oi, 0, 0]),
            "RNN_Yd_step1": float(pred_rnn[oi, 0, 0]),
            "LSTM_Yd_step1": float(pred_lstm[oi, 0, 0]),
        })

        origin_date = df_all["date"].iloc[origin]
        for h in range(H):
            target_date = df_all["date"].iloc[origin + h]
            for j, v in enumerate(var_names):
                tidy.append({
                    "origin_date": origin_date,
                    "target_date": target_date,
                    "step": h + 1,
                    "variable": v,
                    "true": float(y_true[oi, h, j]),
                    cfg.baseline: float(pred_base[oi, h, j]),
                    "ARIMA": float(pred_arima[oi, h, j]),
                    "RNN": float(pred_rnn[oi, h, j]),
                    "LSTM": float(pred_lstm[oi, h, j]),
                })

    df_metrics = pd.concat([
        evaluate_rolling(y_true, pred_base, cfg.baseline, y_insample=y_insample, mase_m=cfg.mase_m),
        evaluate_rolling(y_true, pred_arima, "ARIMA", y_insample=y_insample, mase_m=cfg.mase_m),
        evaluate_rolling(y_true, pred_rnn, "RNN", y_insample=y_insample, mase_m=cfg.mase_m),
        evaluate_rolling(y_true, pred_lstm, "LSTM", y_insample=y_insample, mase_m=cfg.mase_m),
    ], ignore_index=True)
    df_metrics.insert(0, "case_id", case_id)
    df_metrics.to_csv(os.path.join(case_dir, "metrics_rolling.csv"), index=False)

    pd.DataFrame(tidy).to_csv(os.path.join(case_dir, "rolling_predictions_tidy.csv"), index=False)

    df_step1 = pd.DataFrame(step1_rows)
    df_step1.to_csv(os.path.join(case_dir, "rolling_step1_table.csv"), index=False)

    mu = {
        cfg.baseline: df_step1[f"{cfg.baseline}_Yd_step1"].values.astype(float),
        "ARIMA": df_step1["ARIMA_Yd_step1"].values.astype(float),
        "RNN": df_step1["RNN_Yd_step1"].values.astype(float),
        "LSTM": df_step1["LSTM_Yd_step1"].values.astype(float),
    }
    sigma_map = {
        cfg.baseline: float(baseline_sigma(y_insample)[0]),
        "ARIMA": float(arima_sigma(y_insample)[0]),
        "RNN": float(deep_rnn.val_sigma[0]),
        "LSTM": float(deep_lstm.val_sigma[0]),
    }
    df_sens = run_sensitivity_step1(
        true_d=df_step1["true_Yd_step1"].values.astype(float),
        forecast_mu=mu,
        sigma_used=sigma_map,
        c_prod=cfg.unit_production_cost,
        c_over=cfg.overage_cost,
        c_under=cfg.underage_cost,
    )
    df_sens.insert(0, "case_id", case_id)
    df_sens.to_csv(os.path.join(case_dir, "sensitivity_step1.csv"), index=False)

    return df_metrics, df_sens


def main():
    cfg = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-id", type=str, default="M4_YHUR_SHUR")
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--yd-level", type=str, default="HUR")
    ap.add_argument("--solar-level", type=str, default="HUR")
    ap.add_argument("--out", type=str, default=cfg.outputs_root)
    args = ap.parse_args()

    run_one_case(cfg, args.case_id, args.M, args.yd_level, args.solar_level, args.out)
    print(f"Done. Case outputs in {os.path.join(args.out, 'cases', args.case_id)}")


if __name__ == "__main__":
    main()
