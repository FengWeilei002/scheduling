from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METHODS = ["MEAN", "ARIMA", "RNN", "LSTM"]
VARS = ["Yd", "TSE", "TSEC"]
METRICS = ["MAE", "RMSE", "sMAPE", "WAPE", "MASE"]


# ---------------- Style & IO ----------------
def set_plot_style():
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_fig(fig: plt.Figure, out_dir: Path, name_base: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name_base}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{name_base}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------- Load metrics ----------------
def load_metrics_all(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"case_id", "model", "variable", "bucket", "MAE", "RMSE", "sMAPE", "WAPE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in metrics file: {missing}")
    return df


def parse_list(arg: str, allowed: list[str], name: str) -> list[str]:
    if arg.lower() == "all":
        return allowed
    items = [x.strip() for x in arg.split(",") if x.strip()]
    bad = [x for x in items if x not in allowed]
    if bad:
        raise ValueError(f"Invalid {name}: {bad}. Allowed: {allowed} or 'all'")
    return items


def maybe_to_percent(metric: str, vals: np.ndarray) -> tuple[np.ndarray, str]:
    """Some metrics might be stored as 0-1 ratio; convert to percent if needed."""
    if metric not in ("sMAPE", "WAPE"):
        return vals, metric
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return vals, f"{metric} (%)"
    if np.nanmedian(finite) <= 2.0:  # heuristic: treat as ratio
        return vals * 100.0, f"{metric} (%)"
    return vals, f"{metric} (%)"


# ---------------- MASE support ----------------
def _mase_scale_from_series(y: np.ndarray, m: int = 1) -> float:
    """
    MASE scale = mean(|y_t - y_{t-m}|)
    """
    y = np.asarray(y, dtype=float)
    if y.size <= m:
        return np.nan
    diffs = np.abs(y[m:] - y[:-m])
    s = np.nanmean(diffs)
    return float(s) if np.isfinite(s) and s > 0 else np.nan


def add_mase_if_missing(df: pd.DataFrame, cases_root: Path, m: int = 1) -> pd.DataFrame:
    """
    If df does not already contain MASE, compute it from per-case stored files:
    expects each case folder to have y_true.csv and y_pred_<MODEL>.csv OR similar.
    This is a fallback; if your metrics file already has MASE, it won't recompute.
    """
    if "MASE" in df.columns:
        return df

    df = df.copy()
    df["MASE"] = np.nan

    # Try to infer file layout; adjust if your case structure differs.
    # This function is intentionally conservative: it will skip if files are missing.
    for (case_id, variable, bucket), sub in df.groupby(["case_id", "variable", "bucket"]):
        case_dir = cases_root / str(case_id)
        y_true_path = case_dir / f"truth_{variable}_{bucket}.csv"
        if not y_true_path.exists():
            y_true_path = case_dir / "y_true.csv"
        if not y_true_path.exists():
            continue

        try:
            y_true = pd.read_csv(y_true_path).values.squeeze()
        except Exception:
            continue

        scale = _mase_scale_from_series(y_true, m=m)
        if not np.isfinite(scale) or scale <= 0:
            continue

        for idx, row in sub.iterrows():
            model = row["model"]
            y_pred_path = case_dir / f"pred_{model}_{variable}_{bucket}.csv"
            if not y_pred_path.exists():
                y_pred_path = case_dir / f"y_pred_{model}.csv"
            if not y_pred_path.exists():
                continue

            try:
                y_pred = pd.read_csv(y_pred_path).values.squeeze()
            except Exception:
                continue

            mae = float(np.nanmean(np.abs(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float))))
            if np.isfinite(mae):
                df.loc[idx, "MASE"] = mae / scale

    return df


# ---------------- Summaries ----------------
def summarize_by_model(sub: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for m in METHODS:
        x = pd.to_numeric(sub[sub["model"] == m][metric], errors="coerce").dropna().values.astype(float)
        if x.size == 0:
            rows.append({"model": m, "n": 0, "median": np.nan, "mean": np.nan, "q1": np.nan, "q3": np.nan, "iqr": np.nan})
            continue
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        rows.append({
            "model": m,
            "n": int(x.size),
            "median": float(np.median(x)),
            "mean": float(np.mean(x)),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(q3 - q1),
        })
    return pd.DataFrame(rows)


# ---------------- Plot ----------------
def plot_metric_distribution_across_cases(
    df_metrics: pd.DataFrame,
    variable: str,
    bucket: str,
    metric: str,
    out_dir: Path,
    kind: str,
    annotate: bool = True,
    save_summary: bool = True,
):
    sub = df_metrics[
        (df_metrics["variable"] == variable)
        & (df_metrics["bucket"] == bucket)
        & (df_metrics["model"].isin(METHODS))
    ].copy()
    if sub.empty:
        raise ValueError(f"No rows for variable={variable}, bucket={bucket}")

    # One value per (case, model)
    sub = sub.groupby(["case_id", "model"], as_index=False)[metric].mean()

    # Convert unit if needed
    sub2 = sub.copy()
    for m in METHODS:
        idx = sub2["model"] == m
        v = pd.to_numeric(sub2.loc[idx, metric], errors="coerce").values.astype(float)
        v, metric_label = maybe_to_percent(metric, v)
        sub2.loc[idx, metric] = v

    summary = summarize_by_model(sub2, metric)

    # Prepare data in order
    data = []
    for m in METHODS:
        vals = pd.to_numeric(sub2[sub2["model"] == m][metric], errors="coerce").dropna().values.astype(float)
        data.append(vals)

    fig, ax = plt.subplots(figsize=(11, 4))

    if kind == "box":
        bp = ax.boxplot(
            data,
            labels=METHODS,
            showmeans=True,
            meanline=True,
            patch_artist=True,
        )
        for patch in bp["boxes"]:
            patch.set_alpha(0.25)
        for median_line in bp["medians"]:
            median_line.set_linewidth(1.8)
        for mean_line in bp["means"]:
            mean_line.set_linewidth(1.6)
    elif kind == "violin":
        vp = ax.violinplot(data, showmeans=True, showextrema=True)
        for body in vp["bodies"]:
            body.set_alpha(0.25)
        ax.set_xticks(np.arange(1, len(METHODS) + 1))
        ax.set_xticklabels(METHODS)
    else:
        raise ValueError("kind must be 'box' or 'violin'")

    ax.set_title(f"{metric_label} across cases by model — {variable} ({bucket})")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric_label)

    # Annotate median/IQR and delta-to-best
    if annotate:
        ymin, ymax = ax.get_ylim()
        pad = 0.06 * (ymax - ymin + 1e-9)
        ax.set_ylim(ymin, ymax + 2.8 * pad)
        y_text = ymax + 0.8 * pad

        best_median = np.nanmin(summary["median"].values.astype(float))
        for i, m in enumerate(METHODS, start=1):
            row = summary[summary["model"] == m].iloc[0]
            if not np.isfinite(row["median"]):
                continue
            med = float(row["median"])
            iqr = float(row["iqr"])
            d_best = med - best_median
            ax.text(
                i, y_text,
                f"med={med:.2f}\nIQR={iqr:.2f}\nΔbest={d_best:.2f}",
                ha="center", va="bottom",
            )

    name = f"dist_cases_{metric}_{bucket}_{variable}_{kind}"
    save_fig(fig, out_dir, name)

    if save_summary:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_dir / f"summary_{metric}_{bucket}_{variable}.csv", index=False)


# ---------------- Main ----------------
def main():
    p = argparse.ArgumentParser(description="Batch-plot ALL error metrics across all cases.")
    p.add_argument("--metrics", type=str, default="outputs/metrics_all_cases.csv")
    p.add_argument("--cases-root", type=str, default="outputs/cases",
                   help="Needed for MASE if metrics file does not already contain it.")
    p.add_argument("--out", type=str, default="outputs/plots_compare_allcases")

    p.add_argument("--vars", type=str, default="all", help="Yd,TSE,TSEC or 'all'")
    p.add_argument("--buckets", type=str, default="step_1", help="e.g. step_1,step_2,ALL or 'all'")

    # NEW: metrics can be 'all'
    p.add_argument("--metrics-list", type=str, default="all",
                   help="MAE,RMSE,sMAPE,WAPE,MASE or 'all'")

    p.add_argument("--dist", type=str, default="box", choices=["box", "violin", "both"])
    p.add_argument("--mase-m", type=int, default=1, help="Seasonality for MASE scale (default 1).")

    # Default annotate ON (you can disable)
    p.add_argument("--no-annotate", action="store_true", help="Disable numeric annotations.")
    p.add_argument("--no-summary", action="store_true", help="Do not write summary_*.csv files.")

    args = p.parse_args()

    set_plot_style()

    df = load_metrics_all(Path(args.metrics))

    vars_to_plot = parse_list(args.vars, VARS, "vars")
    buckets_to_plot = parse_list(args.buckets, sorted(df["bucket"].unique().tolist()), "buckets")
    metrics_to_plot = parse_list(args.metrics_list, METRICS, "metrics")

    # Ensure MASE exists if requested
    if "MASE" in metrics_to_plot:
        df = add_mase_if_missing(df, Path(args.cases_root), m=int(args.mase_m))

    out_dir = Path(args.out) / "distribution"
    annotate = not bool(args.no_annotate)
    save_summary = not bool(args.no_summary)

    for bucket in buckets_to_plot:
        for v in vars_to_plot:
            for metric in metrics_to_plot:
                if args.dist in ("box", "both"):
                    plot_metric_distribution_across_cases(
                        df, v, bucket, metric, out_dir / bucket / v,
                        kind="box", annotate=annotate, save_summary=save_summary
                    )
                if args.dist in ("violin", "both"):
                    plot_metric_distribution_across_cases(
                        df, v, bucket, metric, out_dir / bucket / v,
                        kind="violin", annotate=annotate, save_summary=save_summary
                    )

    print(f"Done. Figures saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
