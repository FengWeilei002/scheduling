\
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Any

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


def make_time_features(dates: pd.DatetimeIndex) -> np.ndarray:
    dow = dates.dayofweek.values.astype(np.float32)
    doy = dates.dayofyear.values.astype(np.float32)
    n = len(dates)

    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)

    doy_sin = np.sin(2 * np.pi * (doy - 1.0) / 365.25)
    doy_cos = np.cos(2 * np.pi * (doy - 1.0) / 365.25)

    trend = (np.arange(n, dtype=np.float32) / max(n - 1, 1)).astype(np.float32)

    return np.column_stack([dow_sin, dow_cos, doy_sin, doy_cos, trend]).astype(np.float32)


def build_samples(
    y_scaled: np.ndarray,        # (N, 3)
    time_feats: np.ndarray,      # (N, time_dim)
    seq_length: int,
    horizon: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_past, X_future_tf, Y = [], [], []
    N = len(y_scaled)
    for i in range(N - seq_length - horizon + 1):
        y_past = y_scaled[i:i + seq_length]
        tf_past = time_feats[i:i + seq_length]
        x_past = np.concatenate([y_past, tf_past], axis=1)

        tf_future = time_feats[i + seq_length:i + seq_length + horizon]
        y_future = y_scaled[i + seq_length:i + seq_length + horizon]

        X_past.append(x_past)
        X_future_tf.append(tf_future)
        Y.append(y_future)

    return (np.asarray(X_past, dtype=np.float32),
            np.asarray(X_future_tf, dtype=np.float32),
            np.asarray(Y, dtype=np.float32))


def split_by_ratio(n: int, train_ratio: float, val_ratio: float):
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return slice(0, train_end), slice(train_end, val_end)


def build_encoder_decoder(model_type: str, seq_length: int, horizon: int, past_dim: int, time_dim: int) -> tf.keras.Model:
    past_in = layers.Input(shape=(seq_length, past_dim), name="past_in")
    future_tf_in = layers.Input(shape=(horizon, time_dim), name="future_time_in")

    if model_type == "rnn":
        enc = layers.SimpleRNN(64, return_sequences=False)(past_in)
        dec_cell = layers.SimpleRNN
    elif model_type == "lstm":
        enc = layers.LSTM(64, return_sequences=False)(past_in)
        dec_cell = layers.LSTM
    else:
        raise ValueError("model_type must be 'rnn' or 'lstm'")

    enc = layers.Dropout(0.2)(enc)
    rep = layers.RepeatVector(horizon)(enc)
    x = layers.Concatenate(axis=-1)([rep, future_tf_in])

    x = dec_cell(64, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.TimeDistributed(layers.Dense(32, activation="relu"))(x)
    out = layers.TimeDistributed(layers.Dense(3), name="y_out")(x)

    model = models.Model(inputs=[past_in, future_tf_in], outputs=out)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


@dataclass
class DeepForecastResult:
    forecast: np.ndarray              # (horizon, 3) original scale
    val_sigma: np.ndarray             # (3,) original scale residual std
    model: Any
    scalers: Tuple[StandardScaler, StandardScaler, StandardScaler]


def train_and_forecast_deep(
    df_history: pd.DataFrame,       # date,Yd,TSE,TSEC
    df_future: pd.DataFrame,        # date
    seq_length: int,
    horizon: int,
    train_split: float,
    val_split: float,
    epochs: int,
    batch_size: int,
    patience: int,
    model_type: str = "lstm",
    random_seed: int = 42,
) -> DeepForecastResult:
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    y = df_history[["Yd", "TSE", "TSEC"]].values.astype(np.float32)
    dates_hist = pd.DatetimeIndex(pd.to_datetime(df_history["date"]))
    dates_fut = pd.DatetimeIndex(pd.to_datetime(df_future["date"]))

    all_dates = dates_hist.append(dates_fut)
    all_tf = make_time_features(all_dates)
    tf_hist = all_tf[:len(df_history)]
    tf_fut = all_tf[len(df_history):]  # (horizon, time_dim)

    scalers = []
    y_scaled = np.zeros_like(y, dtype=np.float32)
    for j in range(3):
        sc = StandardScaler()
        sc.fit(y[:, j:j+1])
        y_scaled[:, j] = sc.transform(y[:, j:j+1]).ravel().astype(np.float32)
        scalers.append(sc)

    X_past, X_future_tf, Y_lab = build_samples(y_scaled, tf_hist, seq_length, horizon)
    n = X_past.shape[0]
    tr, va = split_by_ratio(n, train_split, val_split)

    time_dim = tf_hist.shape[1]
    past_dim = 3 + time_dim

    model = build_encoder_decoder(model_type, seq_length, horizon, past_dim, time_dim)
    es = callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

    model.fit(
        [X_past[tr], X_future_tf[tr]], Y_lab[tr],
        validation_data=([X_past[va], X_future_tf[va]], Y_lab[va]),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=[es]
    )

    val_pred = model.predict([X_past[va], X_future_tf[va]], verbose=0)
    val_true = Y_lab[va]
    sigma_scaled = np.std((val_true - val_pred).reshape(-1, 3), axis=0)

    sigma = np.zeros((3,), dtype=np.float32)
    for j, sc in enumerate(scalers):
        sigma[j] = float(sigma_scaled[j] * sc.scale_[0])

    # Forecast from last seq window
    y_past = y_scaled[-seq_length:]
    tf_past = tf_hist[-seq_length:]
    past_input = np.concatenate([y_past, tf_past], axis=1)[None, ...].astype(np.float32)
    fut_tf_input = tf_fut[None, ...].astype(np.float32)

    pred_scaled = model.predict([past_input, fut_tf_input], verbose=0)[0]
    pred = np.zeros_like(pred_scaled, dtype=np.float32)
    for j, sc in enumerate(scalers):
        pred[:, j] = sc.inverse_transform(pred_scaled[:, j:j+1]).ravel().astype(np.float32)

    return DeepForecastResult(
        forecast=pred,
        val_sigma=sigma,
        model=model,
        scalers=(scalers[0], scalers[1], scalers[2])
    )
