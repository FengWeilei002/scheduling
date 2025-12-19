from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METHODS_PREF = ["ETS", "MEAN", "ARIMA", "RNN", "LSTM"]
VARS = ["Yd", "TSE", "TSEC"]
METRICS = ["MAE", "RMSE", "sMAPE", "WAPE", "MASE"]

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

def load_metrics_all(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"case_id", "model", "variable", "bucket", *METRICS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    for c in ["case_id", "model", "variable", "bucket"]:
        df[c] = df[c].astype(str)
    for c in METRICS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def parse_list(arg: str, allowed: list[str], name: str) -> list[str]:
    s = (arg or "").strip()
    if s.lower() in {"all", "*", ""}:
        return allowed[:]
    items = [x.strip() for x in s.split(",") if x.strip()]
    bad = [x for x in items if x not in allowed]
    if bad:
        raise ValueError(f"Unknown {name}: {bad}. Allowed: {allowed}")
    return items

def summarize_by_model(sub: pd.DataFrame, metric: str, methods: list[str]) -> pd.DataFrame:
    rows = []
    for m in methods:
        x = pd.to_numeric(sub[sub["model"] == m][metric], errors="coerce").dropna().values.astype(float)
        if x.size == 0:
            rows.append({"model": m, "n": 0, "median": np.nan, "mean": np.nan, "q1": np.nan, "q3": np.nan, "iqr": np.nan})
            continue
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        rows.append({"model": m, "n": int(x.size), "median": float(np.median(x)), "mean": float(np.mean(x)),
                     "q1": float(q1), "q3": float(q3), "iqr": float(q3-q1)})
    return pd.DataFrame(rows)

def draw_stats_table(ax: plt.Axes, summary: pd.DataFrame, bottom_pad: float = 0.34):
    tbl = summary[["model", "n", "median", "iqr", "mean"]].copy()
    fmt = lambda x: "" if not np.isfinite(x) else f"{x:.4g}"
    tbl["median"] = tbl["median"].map(fmt)
    tbl["iqr"] = tbl["iqr"].map(fmt)
    tbl["mean"] = tbl["mean"].map(fmt)

    cell_text = [tbl["n"].tolist(), tbl["median"].tolist(), tbl["iqr"].tolist(), tbl["mean"].tolist()]
    row_labels = ["n", "median", "IQR", "mean"]
    col_labels = tbl["model"].tolist()

    table = ax.table(
        cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
        loc="bottom", cellLoc="center", rowLoc="center",
        bbox=[0.0, -bottom_pad, 1.0, bottom_pad - 0.08],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    plt.subplots_adjust(bottom=bottom_pad)

def plot_metric_distribution(df_metrics: pd.DataFrame, variable: str, bucket: str, metric: str,
                             out_dir: Path, kind: str, methods: list[str],
                             annotate: bool = True, table: bool = True, save_summary: bool = True):
    sub = df_metrics[(df_metrics["variable"] == variable) & (df_metrics["bucket"] == bucket)].copy()
    sub = sub[sub["model"].isin(methods)]
    if sub.empty:
        return

    # one value per (case, model)
    sub = sub.groupby(["case_id", "model"], as_index=False)[metric].mean()

    # percent labeling
    metric_label = metric
    if metric in {"sMAPE", "WAPE"}:
        metric_label = f"{metric} (%)"
        # if values look like fractions, scale up
        if np.nanmedian(sub[metric].values.astype(float)) <= 2.0:
            sub[metric] = sub[metric] * 100.0

    summary = summarize_by_model(sub, metric, methods)
    best_median = np.nanmin(summary["median"].values.astype(float)) if summary["median"].notna().any() else np.nan

    data = [pd.to_numeric(sub[sub["model"] == m][metric], errors="coerce").dropna().values.astype(float) for m in methods]

    fig, ax = plt.subplots(figsize=(11.4, 5.4))
    if kind == "box":
        bp = ax.boxplot(data, labels=methods, patch_artist=True, showmeans=True, meanline=True)
        for patch in bp["boxes"]:
            patch.set_alpha(0.25)
        for med in bp["medians"]:
            med.set_linewidth(1.8)
        for mean in bp["means"]:
            mean.set_linewidth(1.6)
    else:
        vp = ax.violinplot(data, showmeans=True, showextrema=True)
        for body in vp["bodies"]:
            body.set_alpha(0.25)
        ax.set_xticks(np.arange(1, len(methods) + 1))
        ax.set_xticklabels(methods)

    ax.set_title(f"{metric_label} across 28 cases by model — {variable} ({bucket})")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric_label)

    if annotate:
        ymin, ymax = ax.get_ylim()
        pad = 0.06 * (ymax - ymin + 1e-9)
        ax.set_ylim(ymin, ymax + 2.8 * pad)
        y_text = ymax + 0.8 * pad
        for i, m in enumerate(methods, start=1):
            row = summary[summary["model"] == m].iloc[0]
            if not np.isfinite(row["median"]):
                continue
            delta = row["median"] - best_median if np.isfinite(best_median) else np.nan
            delta_txt = "" if not np.isfinite(delta) else f"\nΔbest={delta:.3g}"
            ax.text(i, y_text, f"med={row['median']:.4g}\nIQR={row['iqr']:.4g}{delta_txt}",
                    ha="center", va="bottom", fontsize=10)

    if table:
        draw_stats_table(ax, summary, bottom_pad=0.34)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, out_dir, f"dist_cases_{metric}_{bucket}_{variable}_{kind}")
    if save_summary:
        summary.to_csv(out_dir / f"summary_{metric}_{bucket}_{variable}.csv", index=False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", type=str, default="outputs/metrics_all_cases.csv")
    p.add_argument("--out", type=str, default="outputs/plots_compare_allcases")
    p.add_argument("--vars", type=str, default="all")
    p.add_argument("--buckets", type=str, default="step_1")
    p.add_argument("--metrics-list", type=str, default="all")
    p.add_argument("--dist", type=str, default="box", choices=["box", "violin", "both"])
    p.add_argument("--no-annotate", action="store_true")
    p.add_argument("--no-table", action="store_true")
    p.add_argument("--no-summary", action="store_true")
    args = p.parse_args()

    set_plot_style()
    df = load_metrics_all(Path(args.metrics))

    vars_to_plot = parse_list(args.vars, VARS, "vars")
    buckets_to_plot = parse_list(args.buckets, sorted(df["bucket"].unique().tolist()), "buckets")
    metrics_to_plot = parse_list(args.metrics_list, METRICS, "metrics")

    present = sorted(df["model"].unique().tolist())
    methods = [m for m in METHODS_PREF if m in present] + [m for m in present if m not in METHODS_PREF]
    if not methods:
        raise ValueError("No methods in csv.")

    out_dir = Path(args.out) / "distribution"
    annotate = not args.no_annotate
    table = not args.no_table
    save_summary = not args.no_summary

    for bucket in buckets_to_plot:
        for v in vars_to_plot:
            for metric in metrics_to_plot:
                if args.dist in ("box", "both"):
                    plot_metric_distribution(df, v, bucket, metric, out_dir / bucket / v, "box",
                                             methods, annotate, table, save_summary)
                if args.dist in ("violin", "both"):
                    plot_metric_distribution(df, v, bucket, metric, out_dir / bucket / v, "violin",
                                             methods, annotate, table, save_summary)

    print(f"Done. Saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
