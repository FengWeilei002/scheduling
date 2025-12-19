from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def summarize(df: pd.DataFrame, metric: str, methods: list[str]) -> pd.DataFrame:
    rows = []
    for m in methods:
        x = pd.to_numeric(df[df["model"] == m][metric], errors="coerce").dropna().values.astype(float)
        if x.size == 0:
            rows.append({"model": m, "n": 0, "median": np.nan, "iqr": np.nan, "mean": np.nan})
            continue
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        rows.append({"model": m, "n": int(x.size), "median": float(np.median(x)), "iqr": float(q3-q1), "mean": float(np.mean(x))})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="outputs/sensitivity_all_cases.csv")
    ap.add_argument("--out", type=str, default="outputs/plots_compare_allcases/sensitivity")
    ap.add_argument("--metric", type=str, default="gap_vs_perfect_%", choices=["gap_vs_perfect_%", "mean_cost", "total_cost", "sigma_used"])
    args = ap.parse_args()

    set_plot_style()
    df = pd.read_csv(args.csv)
    if args.metric not in df.columns:
        raise ValueError(f"{args.metric} not in csv.")

    df = df[df["model"] != "PERFECT"].copy()

    pref = ["ETS", "MEAN", "ARIMA", "RNN", "LSTM"]
    present = df["model"].unique().tolist()
    methods = [m for m in pref if m in present] + [m for m in present if m not in pref]

    data = [pd.to_numeric(df[df["model"] == m][args.metric], errors="coerce").dropna().values for m in methods]

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    bp = ax.boxplot(data, labels=methods, patch_artist=True, showmeans=True, meanline=True)
    for patch in bp["boxes"]:
        patch.set_alpha(0.25)
    for med in bp["medians"]:
        med.set_linewidth(1.8)
    for mean in bp["means"]:
        mean.set_linewidth(1.6)

    ax.set_title(f"Sensitivity across cases — {args.metric}")
    ax.set_xlabel("Model")
    ax.set_ylabel(args.metric)

    summary = summarize(df, args.metric, methods)
    best_median = np.nanmin(summary["median"].values.astype(float)) if summary["median"].notna().any() else np.nan

    ymin, ymax = ax.get_ylim()
    pad = 0.06 * (ymax - ymin + 1e-9)
    ax.set_ylim(ymin, ymax + 2.6 * pad)
    y_text = ymax + 0.8 * pad
    for i, m in enumerate(methods, start=1):
        row = summary[summary["model"] == m].iloc[0]
        if not np.isfinite(row["median"]):
            continue
        delta = row["median"] - best_median if np.isfinite(best_median) else np.nan
        delta_txt = "" if not np.isfinite(delta) else f"\nΔbest={delta:.3g}"
        ax.text(i, y_text, f"med={row['median']:.4g}\nIQR={row['iqr']:.4g}{delta_txt}",
                ha="center", va="bottom", fontsize=10)

    out_dir = Path(args.out)
    save_fig(fig, out_dir, f"sensitivity_{args.metric}_box")
    summary.to_csv(out_dir / f"summary_sensitivity_{args.metric}.csv", index=False)
    print(f"Done. Saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
