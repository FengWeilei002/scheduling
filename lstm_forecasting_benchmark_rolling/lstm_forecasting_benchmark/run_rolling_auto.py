import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.data_M_related import generate_synthetic_series, generate_case_series
from src.baselines import mean_forecast
from src.arima_models import arima_forecast_multivariate
from src.metrics import evaluate_rolling
from src.sensitivity import run_sensitivity
from src.deep_models import train_and_forecast_deep, make_time_features


def rolling_origins(history_days: int, n_origins: int, step: int = 1):
    """
    n_origins: number of rolling origins to evaluate.
    """
    start = history_days
    end = history_days + n_origins  # exclusive
    return list(range(start, end, step))


def run_one(cfg: Config, out_dir: Path,
            M: int | None,
            yd_level: str | None,
            solar_level: str | None,
            seed: int | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)

    H = cfg.rolling_horizon
    seed = cfg.random_seed if seed is None else int(seed)

    # ensure last origin has H-day truth
    required_test_len = cfg.rolling_test_days + H - 1
    test_len = max(cfg.test_days, required_test_len)

    # data
    if M is None:
        df = generate_synthetic_series(cfg.start_date, cfg.history_days, test_len, seed)
        case_id = "synthetic"
        meta = {"case_id": case_id, "M": np.nan, "yd_level": "NA", "solar_level": "NA"}
    else:
        if yd_level is None or solar_level is None:
            raise ValueError("When M is provided, you must provide both yd_level and solar_level.")
        yd_level = yd_level.upper().strip()
        solar_level = solar_level.upper().strip()
        df = generate_case_series(cfg.start_date, cfg.history_days, test_len, M, yd_level, solar_level, seed)
        case_id = f"M{M}_Y{yd_level}_S{solar_level}"
        meta = {"case_id": case_id, "M": int(M), "yd_level": yd_level, "solar_level": solar_level}

    df.to_csv(out_dir / "data.csv", index=False)

    df_all = df.copy()
    df_hist0 = df_all.iloc[:cfg.history_days].copy()
    origins = rolling_origins(cfg.history_days, cfg.rolling_test_days, cfg.rolling_step)
    O = len(origins)

    # time features on full timeline
    full_dates = pd.DatetimeIndex(pd.to_datetime(df_all["date"]))
    tf_full = make_time_features(full_dates)

    # train deep models once
    df_future_stub = pd.DataFrame({
        "date": pd.to_datetime(df_all["date"].iloc[cfg.history_days:cfg.history_days + H]).values
    })

    rnn_res = train_and_forecast_deep(
        df_history=df_hist0[["date", "Yd", "TSE", "TSEC"]],
        df_future=df_future_stub,
        seq_length=cfg.seq_length,
        horizon=H,
        train_split=cfg.train_split,
        val_split=cfg.val_split,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        patience=cfg.patience,
        model_type="rnn",
        random_seed=seed
    )

    lstm_res = train_and_forecast_deep(
        df_history=df_hist0[["date", "Yd", "TSE", "TSEC"]],
        df_future=df_future_stub,
        seq_length=cfg.seq_length,
        horizon=H,
        train_split=cfg.train_split,
        val_split=cfg.val_split,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        patience=cfg.patience,
        model_type="lstm",
        random_seed=seed
    )

    # containers
    y_true = np.zeros((O, H, 3), dtype=np.float32)
    pred_mean = np.zeros((O, H, 3), dtype=np.float32)
    pred_arima = np.zeros((O, H, 3), dtype=np.float32)
    pred_rnn = np.zeros((O, H, 3), dtype=np.float32)
    pred_lstm = np.zeros((O, H, 3), dtype=np.float32)

    step1_rows = []

    for oi, origin in enumerate(origins):
        if origin + H > len(df_all):
            raise IndexError("Not enough future truth; increase test_len.")

        df_avail = df_all.iloc[:origin].copy()
        df_fut = df_all.iloc[origin:origin + H].copy()

        y_hist = df_avail[["Yd", "TSE", "TSEC"]].values.astype(np.float32)
        y_true[oi] = df_fut[["Yd", "TSE", "TSEC"]].values.astype(np.float32)

        pred_mean[oi] = mean_forecast(y_hist, H)
        pred_arima[oi] = arima_forecast_multivariate(
            y_hist, H,
            order=cfg.arima_order,
            seasonal_order=cfg.seasonal_order,
            use_auto_arima=cfg.use_auto_arima,
            random_seed=seed
        )

        tf_hist = tf_full[:origin]
        tf_fut = tf_full[origin:origin + H]
        tf_past = tf_hist[-cfg.seq_length:]
        fut_tf_input = tf_fut[None, ...].astype(np.float32)

        # RNN
        y_scaled_rnn = np.zeros_like(y_hist, dtype=np.float32)
        for j, sc in enumerate(rnn_res.scalers):
            y_scaled_rnn[:, j] = sc.transform(y_hist[:, j:j+1]).ravel().astype(np.float32)
        past_input_rnn = np.concatenate([y_scaled_rnn[-cfg.seq_length:], tf_past], axis=1)[None, ...].astype(np.float32)
        ps_rnn = rnn_res.model.predict([past_input_rnn, fut_tf_input], verbose=0)[0]
        pr_rnn = np.zeros_like(ps_rnn, dtype=np.float32)
        for j, sc in enumerate(rnn_res.scalers):
            pr_rnn[:, j] = sc.inverse_transform(ps_rnn[:, j:j+1]).ravel().astype(np.float32)
        pred_rnn[oi] = pr_rnn

        # LSTM (use its own scalers)
        y_scaled_lstm = np.zeros_like(y_hist, dtype=np.float32)
        for j, sc in enumerate(lstm_res.scalers):
            y_scaled_lstm[:, j] = sc.transform(y_hist[:, j:j+1]).ravel().astype(np.float32)
        past_input_lstm = np.concatenate([y_scaled_lstm[-cfg.seq_length:], tf_past], axis=1)[None, ...].astype(np.float32)
        ps_lstm = lstm_res.model.predict([past_input_lstm, fut_tf_input], verbose=0)[0]
        pr_lstm = np.zeros_like(ps_lstm, dtype=np.float32)
        for j, sc in enumerate(lstm_res.scalers):
            pr_lstm[:, j] = sc.inverse_transform(ps_lstm[:, j:j+1]).ravel().astype(np.float32)
        pred_lstm[oi] = pr_lstm

        step1_rows.append({
            **meta,
            "origin_date": df_all["date"].iloc[origin],
            "true_Yd_step1": float(y_true[oi, 0, 0]),
            "MEAN_Yd_step1": float(pred_mean[oi, 0, 0]),
            "ARIMA_Yd_step1": float(pred_arima[oi, 0, 0]),
            "RNN_Yd_step1": float(pred_rnn[oi, 0, 0]),
            "LSTM_Yd_step1": float(pred_lstm[oi, 0, 0]),
        })

    # metrics
    y_insample = df_hist0[["Yd", "TSE", "TSEC"]].values  # 你的固定历史窗口
    df_metrics = pd.concat([
        evaluate_rolling(y_true, pred_mean,  "MEAN",  y_insample=y_insample, mase_m=1),
        evaluate_rolling(y_true, pred_arima, "ARIMA", y_insample=y_insample, mase_m=1),
        evaluate_rolling(y_true, pred_rnn,   "RNN",   y_insample=y_insample, mase_m=1),
        evaluate_rolling(y_true, pred_lstm,  "LSTM",  y_insample=y_insample, mase_m=1),
    ], ignore_index=True)


    for k, v in reversed(list(meta.items())):
        df_metrics.insert(0, k, v)
    df_metrics.to_csv(out_dir / "metrics_summary_rolling.csv", index=False)

    # tidy predictions
    tidy = []
    var_names = ["Yd", "TSE", "TSEC"]
    for oi, origin in enumerate(origins):
        origin_date = df_all["date"].iloc[origin]
        for h in range(H):
            target_date = df_all["date"].iloc[origin + h]
            for j, v in enumerate(var_names):
                tidy.append({
                    **meta,
                    "origin_date": origin_date,
                    "target_date": target_date,
                    "step": h + 1,
                    "variable": v,
                    "true": float(y_true[oi, h, j]),
                    "MEAN": float(pred_mean[oi, h, j]),
                    "ARIMA": float(pred_arima[oi, h, j]),
                    "RNN": float(pred_rnn[oi, h, j]),
                    "LSTM": float(pred_lstm[oi, h, j]),
                })
    pd.DataFrame(tidy).to_csv(out_dir / "rolling_predictions_tidy.csv", index=False)

    # step1 table
    pd.DataFrame(step1_rows).to_csv(out_dir / "rolling_step1_table.csv", index=False)

    # sensitivity (step1, Yd only)
    df_step1 = pd.DataFrame(step1_rows)
    df_sens = pd.DataFrame({"Yd": df_step1["true_Yd_step1"].values})
    forecasts = {
        "MEAN": np.column_stack([df_step1["MEAN_Yd_step1"].values, np.zeros(len(df_step1)), np.zeros(len(df_step1))]),
        "ARIMA": np.column_stack([df_step1["ARIMA_Yd_step1"].values, np.zeros(len(df_step1)), np.zeros(len(df_step1))]),
        "RNN": np.column_stack([df_step1["RNN_Yd_step1"].values, np.zeros(len(df_step1)), np.zeros(len(df_step1))]),
        "LSTM": np.column_stack([df_step1["LSTM_Yd_step1"].values, np.zeros(len(df_step1)), np.zeros(len(df_step1))]),
    }

    sigma_mean = np.std(
        df_hist0[["Yd", "TSE", "TSEC"]].values
        - np.mean(df_hist0[["Yd", "TSE", "TSEC"]].values, axis=0),
        axis=0
    )
    sigma_arima = np.std(
        df_hist0[["Yd", "TSE", "TSEC"]].values[1:]
        - df_hist0[["Yd", "TSE", "TSEC"]].values[:-1],
        axis=0
    )
    sigmas = {"MEAN": sigma_mean, "ARIMA": sigma_arima, "RNN": rnn_res.val_sigma, "LSTM": lstm_res.val_sigma}

    df_cost = run_sensitivity(
        df_test=df_sens,
        forecasts=forecasts,
        sigmas=sigmas,
        c_prod=cfg.unit_production_cost,
        c_over=cfg.overage_cost,
        c_under=cfg.underage_cost
    )
    for k, v in reversed(list(meta.items())):
        df_cost.insert(0, k, v)
    df_cost.to_csv(out_dir / "sensitivity_costs_rolling_step1.csv", index=False)

    print(f"[OK] {meta['case_id']} -> {out_dir.resolve()}")
    return df_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=None)
    p.add_argument("--yd", type=str, default=None, choices=["HUR", "LUR", "hur", "lur"])
    p.add_argument("--solar", type=str, default=None, choices=["HUR", "LUR", "hur", "lur"])
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    cfg = Config()

    if args.M is None:
        out_dir = Path(args.outdir) if args.outdir else Path("outputs")
        run_one(cfg, out_dir=out_dir, M=None, yd_level=None, solar_level=None, seed=args.seed)
    else:
        yd_level = args.yd.upper() if args.yd else None
        solar_level = args.solar.upper() if args.solar else None
        case_id = f"M{args.M}_Y{yd_level}_S{solar_level}"
        out_dir = Path(args.outdir) if args.outdir else (Path("outputs") / "cases" / case_id)
        run_one(cfg, out_dir=out_dir, M=args.M, yd_level=yd_level, solar_level=solar_level, seed=args.seed)


if __name__ == "__main__":
    main()
