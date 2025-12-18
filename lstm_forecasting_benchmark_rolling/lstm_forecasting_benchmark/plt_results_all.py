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
    # tight_layout helps keep title/annotations from colliding in the saved figure
    try:
        fig.tight_layout()
    except Exception:
        pass
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


# ---------------- MASE support ----------------
def mase_scale_from_history(x: np.ndarray, m: int = 1, eps: float = 1e-8) -> float:
    """Denominator for MASE: mean absolute naive(m) in-sample error."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= m:
        return np.nan
    scale = float(np.mean(np.abs(x[m:] - x[:-m])))
    return max(scale, eps)


def add_mase_if_missing(df_metrics: pd.DataFrame, cases_root: Path, m: int = 1) -> pd.DataFrame:
    """
    If MASE exists and has at least some valid values, keep it.
    Otherwise compute MASE = MAE / scale, where scale is computed from each case's history segment
    in outputs/cases/<case_id>/data.csv (split=='history').
    """
    if "MASE" in df_metrics.columns and df_metrics["MASE"].notna().any():
        return df_metrics

    scales: dict[tuple[str, str], float] = {}  # (case_id, variable) -> scale

    for case_id in df_metrics["case_id"].unique():
        data_path = cases_root / str(case_id) / "data.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Cannot compute MASE: missing {data_path}")

        df_data = pd.read_csv(data_path)
        if "split" not in df_data.columns:
            raise ValueError(f"{data_path} has no 'split' column; cannot identify history segment for MASE scale.")
        df_hist = df_data[df_data["split"] == "history"].copy()
        if df_hist.empty:
            raise ValueError(f"{data_path}: no rows with split=='history'.")

        # compute per-variable scale once per case
        for v in VARS:
            if v not in df_hist.columns:
                continue
            scales[(str(case_id), v)] = mase_scale_from_history(df_hist[v].values, m=m)

    df = df_metrics.copy()
    df["MASE"] = [
        np.nan
        if not np.isfinite(scales.get((str(r["case_id"]), str(r["variable"])), np.nan))
        else (
            float(r["MAE"]) / float(scales[(str(r["case_id"]), str(r["variable"]))])
            if np.isfinite(float(r["MAE"])) else np.nan
        )
        for _, r in df.iterrows()
    ]
    return df


# ---------------- Helpers ----------------
def parse_list(arg: str, allowed: list[str], name: str) -> list[str]:
    if arg is None:
        return allowed[:]
    if arg.lower() == "all":
        return allowed[:]
    items = [x.strip() for x in arg.split(",") if x.strip()]
    bad = [x for x in items if x not in allowed]
    if bad:
        raise ValueError(f"Invalid {name}: {bad}. Allowed: {allowed} or 'all'")
    return items


def maybe_to_percent(metric: str, values: np.ndarray) -> tuple[np.ndarray, str]:
    # Convert sMAPE/WAPE to percent if stored as ratios
    if metric not in {"sMAPE", "WAPE"}:
        return values, metric
    vals = values.copy()
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return vals, f"{metric} (%)"
    if np.nanmedian(finite) <= 2.0:  # heuristic: treat as ratio
        return vals * 100.0, f"{metric} (%)"
    return vals, f"{metric} (%)"


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
    metric_label = metric
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

    # Slightly taller canvas to reduce crowding (same idea as plt_results_all.py)
    fig, ax = plt.subplots(figsize=(11, 4.8))

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

    # Increase title padding so it stays clear of the plot area
    ax.set_title(f"{metric_label} across cases by model — {variable} ({bucket})", pad=16)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric_label)

    # Annotate median/IQR and delta-to-best (INSIDE axes to avoid title overlap)
    if annotate:
        # add headroom so top-of-axis text doesn't sit on whiskers/outliers
        ymin, ymax = ax.get_ylim()
        pad_y = 0.12 * (ymax - ymin + 1e-9)
        ax.set_ylim(ymin, ymax + pad_y)

        best_median = float(np.nanmin(summary["median"].values.astype(float))) if summary["median"].notna().any() else np.nan

        # y position in AXES coordinates (0..1): stable and won't collide with the title
        y_ax = 0.98
        for i, m in enumerate(METHODS, start=1):
            row = summary[summary["model"] == m].iloc[0]
            if not np.isfinite(row["median"]):
                continue
            delta = row["median"] - best_median if np.isfinite(best_median) else np.nan
            delta_txt = "" if not np.isfinite(delta) else f"\nΔbest={delta:.3g}"

            ax.text(
                i, y_ax,
                f"med={row['median']:.4g}\nIQR={row['iqr']:.4g}{delta_txt}",
                transform=ax.get_xaxis_transform(),  # x=data coord, y=axes coord
                ha="center",
                va="top",
                fontsize=10,
                clip_on=False,
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

    p.add_argument("--metrics-list", type=str, default="all",
                   help="MAE,RMSE,sMAPE,WAPE,MASE or 'all'")

    p.add_argument("--dist", type=str, default="box", choices=["box", "violin", "both"])
    p.add_argument("--mase-m", type=int, default=1, help="Seasonality for MASE scale (default 1).")

    # Default annotations ON (you can disable)
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
