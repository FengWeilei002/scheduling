import os
import numpy as np
import pandas as pd

from src.config import Config
from src.data import generate_synthetic_series
from src.baselines import mean_forecast
from src.arima_models import arima_forecast_multivariate
from src.metrics import evaluate_rolling
from src.sensitivity import run_sensitivity
from src.deep_models import train_and_forecast_deep, make_time_features


def rolling_origins(history_days: int, rolling_test_days: int, step: int = 1):
    """
    rolling_test_days is interpreted as the NUMBER OF ROLLING ORIGINS (days evaluated),
    not the length of the test segment.

    Example: rolling_test_days=90, step=1 -> 90 origins:
             origin = history_days ... history_days+89
    """
    start = history_days
    end = history_days + rolling_test_days  # exclusive
    return list(range(start, end, step))


def main():
    cfg = Config()
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    H = cfg.rolling_horizon

    # Basic safety checks for rolling setup
    if cfg.history_days < cfg.seq_length:
        raise ValueError(f"history_days ({cfg.history_days}) must be >= seq_length ({cfg.seq_length}) for deep models.")

    # IMPORTANT: ensure we generate enough future days so that the LAST origin still has H-step truth.
    # If we evaluate N origins (rolling_test_days), the test segment must be at least N + H - 1 days long.
    required_test_len = cfg.rolling_test_days + H - 1
    test_len = max(cfg.test_days, required_test_len)

    df = generate_synthetic_series(cfg.start_date, cfg.history_days, test_len, cfg.random_seed)
    df.to_csv(cfg.data_csv, index=False)

    df_all = df.copy()
    df_hist0 = df_all.iloc[:cfg.history_days].copy()

    origins = rolling_origins(cfg.history_days, cfg.rolling_test_days, cfg.rolling_step)
    O = len(origins)

    # Precompute time features ONCE on the full timeline to avoid feature definition drift (trend rescaling)
    full_dates = pd.DatetimeIndex(pd.to_datetime(df_all["date"]))
    tf_full = make_time_features(full_dates)

    # Train deep models once on initial history (horizon = H)
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
        random_seed=cfg.random_seed
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
        random_seed=cfg.random_seed
    )

    # Containers
    y_true = np.zeros((O, H, 3), dtype=np.float32)
    pred_mean = np.zeros((O, H, 3), dtype=np.float32)
    pred_arima = np.zeros((O, H, 3), dtype=np.float32)
    pred_rnn = np.zeros((O, H, 3), dtype=np.float32)
    pred_lstm = np.zeros((O, H, 3), dtype=np.float32)

    step1_rows = []

    for oi, origin in enumerate(origins):
        # Ensure we have enough future truth
        if origin + H > len(df_all):
            raise IndexError(
                f"Not enough future days at origin={origin}. "
                f"Need origin+H <= {len(df_all)}. Increase test_len."
            )

        df_avail = df_all.iloc[:origin].copy()
        df_fut = df_all.iloc[origin:origin + H].copy()

        y_hist = df_avail[["Yd", "TSE", "TSEC"]].values.astype(np.float32)
        y_true[oi] = df_fut[["Yd", "TSE", "TSEC"]].values.astype(np.float32)

        # Baselines (recomputed each origin with newly available truth)
        pred_mean[oi] = mean_forecast(y_hist, H)
        pred_arima[oi] = arima_forecast_multivariate(
            y_hist, H,
            order=cfg.arima_order,
            seasonal_order=cfg.seasonal_order,
            use_auto_arima=cfg.use_auto_arima,
            random_seed=cfg.random_seed
        )

        # Deep forward pass with updated history
        tf_hist = tf_full[:origin]
        tf_fut = tf_full[origin:origin + H]  # (H, time_dim)

        tf_past = tf_hist[-cfg.seq_length:]  # (seq_length, time_dim)
        fut_tf_input = tf_fut[None, ...].astype(np.float32)

        # --- RNN input scaling (use RNN scalers) ---
        y_scaled_rnn = np.zeros_like(y_hist, dtype=np.float32)
        for j, sc in enumerate(rnn_res.scalers):
            y_scaled_rnn[:, j] = sc.transform(y_hist[:, j:j+1]).ravel().astype(np.float32)

        y_past_rnn = y_scaled_rnn[-cfg.seq_length:]
        past_input_rnn = np.concatenate([y_past_rnn, tf_past], axis=1)[None, ...].astype(np.float32)

        ps_rnn = rnn_res.model.predict([past_input_rnn, fut_tf_input], verbose=0)[0]
        pr_rnn = np.zeros_like(ps_rnn, dtype=np.float32)
        for j, sc in enumerate(rnn_res.scalers):
            pr_rnn[:, j] = sc.inverse_transform(ps_rnn[:, j:j+1]).ravel().astype(np.float32)
        pred_rnn[oi] = pr_rnn

        # --- LSTM input scaling (use LSTM scalers; DO NOT reuse RNN-scaled input) ---
        y_scaled_lstm = np.zeros_like(y_hist, dtype=np.float32)
        for j, sc in enumerate(lstm_res.scalers):
            y_scaled_lstm[:, j] = sc.transform(y_hist[:, j:j+1]).ravel().astype(np.float32)

        y_past_lstm = y_scaled_lstm[-cfg.seq_length:]
        past_input_lstm = np.concatenate([y_past_lstm, tf_past], axis=1)[None, ...].astype(np.float32)

        ps_lstm = lstm_res.model.predict([past_input_lstm, fut_tf_input], verbose=0)[0]
        pr_lstm = np.zeros_like(ps_lstm, dtype=np.float32)
        for j, sc in enumerate(lstm_res.scalers):
            pr_lstm[:, j] = sc.inverse_transform(ps_lstm[:, j:j+1]).ravel().astype(np.float32)
        pred_lstm[oi] = pr_lstm

        step1_rows.append({
            "origin_date": df_all["date"].iloc[origin],
            "true_Yd_step1": float(y_true[oi, 0, 0]),
            "MEAN_Yd_step1": float(pred_mean[oi, 0, 0]),
            "ARIMA_Yd_step1": float(pred_arima[oi, 0, 0]),
            "RNN_Yd_step1": float(pred_rnn[oi, 0, 0]),
            "LSTM_Yd_step1": float(pred_lstm[oi, 0, 0]),
        })

    # Metrics
    df_metrics = pd.concat([
        evaluate_rolling(y_true, pred_mean, "MEAN"),
        evaluate_rolling(y_true, pred_arima, "ARIMA"),
        evaluate_rolling(y_true, pred_rnn, "RNN"),
        evaluate_rolling(y_true, pred_lstm, "LSTM"),
    ], ignore_index=True)
    df_metrics.to_csv("outputs/metrics_summary_rolling.csv", index=False)

    # Tidy predictions
    tidy = []
    var_names = ["Yd", "TSE", "TSEC"]
    for oi, origin in enumerate(origins):
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
                    "MEAN": float(pred_mean[oi, h, j]),
                    "ARIMA": float(pred_arima[oi, h, j]),
                    "RNN": float(pred_rnn[oi, h, j]),
                    "LSTM": float(pred_lstm[oi, h, j]),
                })
    pd.DataFrame(tidy).to_csv("outputs/rolling_predictions_tidy.csv", index=False)

    # Step-1 table + sensitivity
    df_step1 = pd.DataFrame(step1_rows)
    df_step1.to_csv("outputs/rolling_step1_table.csv", index=False)

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
    df_cost.to_csv("outputs/sensitivity_costs_rolling_step1.csv", index=False)

    print("Rolling evaluation done. See outputs/.")


if __name__ == "__main__":
    main()
