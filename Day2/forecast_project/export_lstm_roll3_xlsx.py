"""
Export LSTM rolling 3-day forecasts to Excel for all 28 cases.

Outputs:
- For each case, create one Excel file under outputs/lstm_xlsx (default):
    predictions_<case_id>_LSTM.xlsx
  with 4 sheets:
    1) test_data         (90-day test truth from data.csv)
    2) Yd_predictions    (90x3 rolling forecasts)
    3) TSE_predictions   (90x3 rolling forecasts)
    4) TSEC_predictions  (90x3 rolling forecasts)

Rolling logic (test days t = 1..90):
- After observing today's true values (history + first t test days),
  forecast next 3 days (t+1..t+3) for Yd/TSE/TSEC.

Yd special rule (ENFORCED):
- True Yd for test days 81..90 is set to 0 (written into test_data sheet).
- Any Yd forecast whose TARGET day is in 81..90 OR > 90 is forced to 0 in output.
  This also ensures the last few rows (whose targets go to 91..93) are all zeros.

Usage:
  python export_lstm_roll3_xlsx.py

Optional:
  python export_lstm_roll3_xlsx.py --no-cache
  python export_lstm_roll3_xlsx.py --retrain-daily   (very slow)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from src.config import Config
from src.data import enumerate_cases, generate_case_series, save_case_meta
from src.deep_models import train_deep_model, deep_forward, DeepForecastResult


PRED_SHEETS = {
    "Yd": "Yd_predictions",
    "TSE": "TSE_predictions",
    "TSEC": "TSEC_predictions",
}

TEST_SHEET_NAME = "test_data"


def ensure_case_data(cfg: Config, case: dict, case_dir: Path) -> pd.DataFrame:
    """
    Prefer existing data.csv in case folder; otherwise generate and save it.
    """
    data_path = case_dir / "data.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        for c in ["date", "Yd", "TSE", "TSEC", "split"]:
            if c not in df.columns:
                raise ValueError(f"{data_path} missing column '{c}'")
        return df

    df = generate_case_series(
        start_date=cfg.start_date,
        history_days=cfg.history_days,
        test_days=cfg.rolling_test_days,
        random_seed=cfg.random_seed,
        M=int(case["M"]),
        yd_level=str(case["yd_level"]),
        solar_level=str(case["solar_level"]),
    )
    case_dir.mkdir(parents=True, exist_ok=True)
    save_case_meta(str(case_dir), case)
    df.to_csv(data_path, index=False)
    return df


def force_true_yd_last10_to_zero(df: pd.DataFrame, history_days: int, test_days: int) -> pd.DataFrame:
    """
    Force TRUE Yd values for the last 10 days of the test window to 0.
    Test window indices in full series: [history_days .. history_days+test_days-1]
    Last 10 test days correspond to test_day in [test_days-9 .. test_days] => indices [end-10 .. end-1]
    """
    df2 = df.copy()
    start = history_days
    end = history_days + test_days  # exclusive
    last10_start = end - 10
    df2.loc[last10_start:end - 1, "Yd"] = 0.0
    return df2


def apply_yd_zero_targets(yd_mat: np.ndarray, test_days: int, horizon: int) -> np.ndarray:
    """
    yd_mat shape: (test_days, horizon)
    Row t (1..test_days) stores forecasts for target days (t+1..t+horizon).

    Force output to 0 if:
      - target_day in last 10 test days: [test_days-9 .. test_days]
      - OR target_day > test_days (out of range, e.g., 91..93)
    """
    yd = yd_mat.copy()
    last10_lo = test_days - 9
    last10_hi = test_days
    for t in range(1, test_days + 1):
        for h in range(1, horizon + 1):
            target_day = t + h
            if (last10_lo <= target_day <= last10_hi) or (target_day > test_days):
                yd[t - 1, h - 1] = 0.0
    return yd


# ---- Optional per-case cache (to avoid retraining each run) ----
def save_lstm_cache(case_dir: Path, model: tf.keras.Model, scalers: list[StandardScaler], meta: dict):
    case_dir.mkdir(parents=True, exist_ok=True)
    model_path = case_dir / "lstm_export_model.keras"
    model.save(model_path, include_optimizer=False)

    means = [float(sc.mean_[0]) for sc in scalers]
    scales = [float(sc.scale_[0]) for sc in scalers]
    bundle = {"means": means, "scales": scales, "meta": meta}
    (case_dir / "lstm_export_scalers.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")


def load_lstm_cache(case_dir: Path, horizon: int) -> DeepForecastResult | None:
    model_path = case_dir / "lstm_export_model.keras"
    scaler_path = case_dir / "lstm_export_scalers.json"
    if not (model_path.exists() and scaler_path.exists()):
        return None

    try:
        model = tf.keras.models.load_model(model_path)
        bundle = json.loads(scaler_path.read_text(encoding="utf-8"))
        means = bundle["means"]
        scales = bundle["scales"]
        meta = bundle.get("meta", {})
        V = int(meta.get("V", 3))
        time_dim = int(meta.get("time_dim", 5))

        scalers: list[StandardScaler] = []
        for j in range(V):
            sc = StandardScaler()
            sc.mean_ = np.array([means[j]], dtype=float)
            sc.scale_ = np.array([scales[j]], dtype=float)
            sc.var_ = sc.scale_ ** 2
            sc.n_samples_seen_ = 1
            scalers.append(sc)

        return DeepForecastResult(
            forecast=np.zeros((horizon, V), dtype=np.float32),
            val_sigma=np.zeros((V,), dtype=np.float32),
            model=model,
            scalers=tuple(scalers),
            time_dim=time_dim,
            V=V,
        )
    except Exception:
        return None


def write_excel_with_test_and_preds(
    df_test: pd.DataFrame,
    preds: dict[str, np.ndarray],
    out_path: Path,
):
    """
    Write one workbook containing:
      - test_data sheet (truth)
      - 3 prediction sheets
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # test truth sheet
        df_test.to_excel(writer, sheet_name=TEST_SHEET_NAME, index=False)

        # prediction sheets
        for var, sheet in PRED_SHEETS.items():
            mat = preds[var]
            df_out = pd.DataFrame(mat, columns=[f"{var}_day_{i}" for i in range(1, mat.shape[1] + 1)])
            df_out.to_excel(writer, sheet_name=sheet, index=False)


def main():
    cfg = Config()

    ap = argparse.ArgumentParser(description="Export LSTM rolling 3-day forecasts into Excel for all 28 cases.")
    ap.add_argument("--cases-root", type=str, default=str(Path(cfg.outputs_root) / "cases"),
                    help="Folder containing case folders (default: outputs/cases).")
    ap.add_argument("--out-dir", type=str, default=str(Path(cfg.outputs_root) / "lstm_xlsx"),
                    help="Where to write 28 xlsx files (default: outputs/lstm_xlsx).")

    ap.add_argument("--horizon", type=int, default=3, help="Forecast horizon (must be 3).")
    ap.add_argument("--test-days", type=int, default=cfg.rolling_test_days, help="Number of test days (default 90).")

    ap.add_argument("--no-cache", action="store_true", help="Disable per-case model cache; retrain once per case.")
    ap.add_argument("--retrain-daily", action="store_true",
                    help="Strict walk-forward: retrain LSTM every test day (VERY SLOW).")

    ap.add_argument("--epochs", type=int, default=cfg.epochs, help="Training epochs.")
    ap.add_argument("--patience", type=int, default=cfg.patience, help="Early stopping patience.")
    ap.add_argument("--seed", type=int, default=cfg.random_seed + 7, help="Random seed for LSTM training.")
    args = ap.parse_args()

    horizon = int(args.horizon)
    test_days = int(args.test_days)
    if horizon != 3:
        raise ValueError("This exporter is designed for horizon=3 exactly.")
    if test_days != 90:
        # 允许改，但你的需求是90；不强制，只提醒
        print(f"[WARN] test_days={test_days} (your paper setup is 90). Proceeding anyway.")

    cases_root = Path(args.cases_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = enumerate_cases(cfg.M_list, cfg.yd_levels, cfg.solar_levels)

    ok, fail = 0, 0
    for case in cases:
        case_id = case["case_id"]
        case_dir = cases_root / case_id

        try:
            # 1) load/generate data
            df = ensure_case_data(cfg, case, case_dir)
            df["date"] = pd.to_datetime(df["date"])

            # 2) ENFORCE: true Yd last10 test days = 0
            df = force_true_yd_last10_to_zero(df, cfg.history_days, test_days)

            # 3) extract test truth (90 rows) and build test_data sheet
            df_test = df[df["split"].astype(str).str.lower() == "test"].copy()
            if len(df_test) != test_days:
                # fallback: slice last test_days from the "test" segment
                df_test = df_test.tail(test_days).copy()
            df_test = df_test.reset_index(drop=True)
            df_test.insert(0, "test_day", np.arange(1, len(df_test) + 1))
            # Keep only columns you want in excel truth sheet
            df_test = df_test[["test_day", "date", "Yd", "TSE", "TSEC"]]

            # 4) train/load LSTM once per case (unless retrain-daily)
            res = None
            if (not args.no_cache) and (not args.retrain_daily):
                res = load_lstm_cache(case_dir, horizon=horizon)

            if res is None and (not args.retrain_daily):
                df_hist0 = df.iloc[:cfg.history_days].copy()
                res = train_deep_model(
                    df_history=df_hist0[["date", "Yd", "TSE", "TSEC"]],
                    seq_length=cfg.seq_length,
                    horizon=horizon,
                    train_split=cfg.train_split,
                    val_split=cfg.val_split,
                    epochs=int(args.epochs),
                    batch_size=cfg.batch_size,
                    patience=int(args.patience),
                    model_type="lstm",
                    random_seed=int(args.seed),
                )
                if not args.no_cache:
                    save_lstm_cache(case_dir, res.model, list(res.scalers), meta={"V": res.V, "time_dim": res.time_dim})

            # 5) rolling forecasts
            var_names = ["Yd", "TSE", "TSEC"]
            preds = {v: np.zeros((test_days, horizon), dtype=np.float32) for v in var_names}

            for t in range(1, test_days + 1):
                # available data includes: initial history + first t days of test (today observed)
                origin_len = cfg.history_days + t
                df_avail = df.iloc[:origin_len].copy()

                y_hist = df_avail[var_names].values.astype(np.float32)
                dates_hist = pd.DatetimeIndex(df_avail["date"])

                last_date = df_avail["date"].iloc[-1]
                dates_fut = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

                if args.retrain_daily:
                    # strict: retrain using all currently available data
                    df_hist_today = df_avail[["date", "Yd", "TSE", "TSEC"]].copy()
                    res_today = train_deep_model(
                        df_history=df_hist_today,
                        seq_length=cfg.seq_length,
                        horizon=horizon,
                        train_split=cfg.train_split,
                        val_split=cfg.val_split,
                        epochs=int(args.epochs),
                        batch_size=cfg.batch_size,
                        patience=int(args.patience),
                        model_type="lstm",
                        random_seed=int(args.seed) + t,
                    )
                    pred = deep_forward(res_today, y_hist, dates_hist, dates_fut, seq_length=cfg.seq_length)
                else:
                    pred = deep_forward(res, y_hist, dates_hist, dates_fut, seq_length=cfg.seq_length)

                for j, v in enumerate(var_names):
                    preds[v][t - 1, :] = pred[:, j]

            # 6) ENFORCE: Yd forecast targets in last10 OR beyond test_days are 0
            preds["Yd"] = apply_yd_zero_targets(preds["Yd"], test_days=test_days, horizon=horizon)

            # 7) write xlsx
            out_path = out_dir / f"predictions_{case_id}_LSTM.xlsx"
            write_excel_with_test_and_preds(df_test, preds, out_path)

            ok += 1
            print(f"[OK] {case_id} -> {out_path}")
        except Exception as e:
            fail += 1
            print(f"[FAIL] {case_id}: {e}")

    print(f"Done. Cases succeeded: {ok}, failed: {fail}")
    print(f"XLSX output folder: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
