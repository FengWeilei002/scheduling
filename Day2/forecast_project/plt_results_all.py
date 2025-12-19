from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DEFAULT_METHODS = ["ETS", "MEAN", "ARIMA", "RNN", "LSTM"]
VARS = ["Yd", "TSE", "TSEC"]
ERR_METRICS = ["abs", "signed", "smape", "mase"]

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

def _date_axis(ax: plt.Axes):
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

def load_tidy(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["origin_date", "target_date"])
    required = {"origin_date", "target_date", "step", "variable", "true"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df["step"] = df["step"].astype(int)
    df["variable"] = df["variable"].astype(str)
    return df

def pick_methods(df: pd.DataFrame) -> list[str]:
    present = [c for c in df.columns if c not in {"origin_date", "target_date", "step", "variable", "true"}]
    methods = [m for m in DEFAULT_METHODS if m in present]
    for m in present:
        if m not in methods:
            methods.append(m)
    return methods

def select_rows_fixed_step(df: pd.DataFrame, variable: str, step: int, latest_per_target: bool) -> pd.DataFrame:
    sub = df[(df["variable"] == variable) & (df["step"] == step)].copy()
    if sub.empty:
        raise ValueError(f"No data for variable={variable}, step={step}")
    if latest_per_target:
        sub = (sub.sort_values(["target_date", "origin_date"])
                 .groupby("target_date", as_index=False)
                 .tail(1)
                 .sort_values("target_date"))
    else:
        sub = sub.sort_values(["origin_date", "target_date"])
    return sub

def select_rows_stitched_latest(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Build a full-length series by taking the *latest available forecast* for each target_date (across all steps)."""
    sub = df[df["variable"] == variable].copy()
    if sub.empty:
        raise ValueError(f"No data for variable={variable}")
    sub = (sub.sort_values(["target_date", "origin_date"])
             .groupby("target_date", as_index=False)
             .tail(1)
             .sort_values("target_date"))
    return sub

def mase_scale_from_history(x: np.ndarray, m: int = 1, eps: float = 1e-8) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= m:
        return np.nan
    s = float(np.mean(np.abs(x[m:] - x[:-m])))
    return max(s, eps)

def compute_case_mase_scales(case_dir: Path, vars_to_plot: list[str], mase_m: int) -> dict[str, float]:
    data_path = case_dir / "data.csv"
    df = pd.read_csv(data_path)
    hist = df[df["split"] == "history"].copy()
    scales = {}
    for v in vars_to_plot:
        scales[v] = mase_scale_from_history(hist[v].values, m=mase_m)
    return scales

def plot_timeseries(df_sub: pd.DataFrame, variable: str, methods: list[str], out_dir: Path, title: str, name_base: str):
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(df_sub["target_date"], df_sub["true"], label="Truth", linewidth=2.6)
    for m in methods:
        ax.plot(df_sub["target_date"], df_sub[m], label=m, linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(variable)
    _date_axis(ax)
    ax.legend(ncol=min(5, len(methods)+1), loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    save_fig(fig, out_dir, name_base)

def compute_error_table(df_sub: pd.DataFrame, methods: list[str], error_metric: str, mase_scale: float | None = None) -> pd.DataFrame:
    y = df_sub["true"].astype(float).values
    eps = 1e-8
    rows = []
    if error_metric == "mase" and (mase_scale is None or not np.isfinite(mase_scale)):
        raise ValueError("MASE requested but mase_scale missing/invalid.")
    for m in methods:
        yhat = df_sub[m].astype(float).values
        e = yhat - y
        if error_metric == "abs":
            err = np.abs(e)
        elif error_metric == "signed":
            err = e
        elif error_metric == "smape":
            denom = np.maximum(np.abs(y) + np.abs(yhat), eps)
            err = 200.0 * np.abs(e) / denom
        elif error_metric == "mase":
            err = np.abs(e) / float(mase_scale)
        else:
            raise ValueError("Unknown error_metric")
        rows.append(pd.DataFrame({"model": m, "err": err}))
    return pd.concat(rows, ignore_index=True)

def stats_by_model(df_err: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    out = []
    for m in methods:
        x = df_err[df_err["model"] == m]["err"].dropna().values.astype(float)
        if x.size == 0:
            out.append({"model": m, "n": 0, "median": np.nan, "iqr": np.nan, "mean": np.nan})
            continue
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        out.append({"model": m, "n": int(x.size), "median": float(np.median(x)), "iqr": float(q3-q1), "mean": float(np.mean(x))})
    return pd.DataFrame(out)

def plot_error_distribution(df_err: pd.DataFrame, methods: list[str], out_dir: Path, kind: str,
                            title: str, ylabel: str, name_base: str, annotate: bool, save_stats: bool):
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    data = [df_err[df_err["model"] == m]["err"].dropna().values for m in methods]

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

    rng = np.random.default_rng(42)
    for i, m in enumerate(methods, start=1):
        vals = df_err[df_err["model"] == m]["err"].dropna().values
        if vals.size == 0:
            continue
        if vals.size > 2000:
            vals = rng.choice(vals, size=2000, replace=False)
        jitter = rng.uniform(-0.08, 0.08, size=vals.size)
        ax.scatter(np.full(vals.size, i) + jitter, vals, s=10, alpha=0.35)

    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)

    stats = stats_by_model(df_err, methods)
    best_median = np.nanmin(stats["median"].values.astype(float)) if stats["median"].notna().any() else np.nan

    if annotate:
        ymin, ymax = ax.get_ylim()
        pad = 0.06 * (ymax - ymin + 1e-9)
        ax.set_ylim(ymin, ymax + 2.6 * pad)
        y_text = ymax + 0.8 * pad
        for i, m in enumerate(methods, start=1):
            row = stats[stats["model"] == m].iloc[0]
            if not np.isfinite(row["median"]):
                continue
            delta = row["median"] - best_median if np.isfinite(best_median) else np.nan
            delta_txt = "" if not np.isfinite(delta) else f"\nΔbest={delta:.3g}"
            ax.text(i, y_text, f"med={row['median']:.4g}\nIQR={row['iqr']:.4g}{delta_txt}",
                    ha="center", va="bottom", fontsize=10)

    save_fig(fig, out_dir, name_base)

    if save_stats:
        stats.to_csv(out_dir / f"{name_base}_stats.csv", index=False)

def iter_case_tidy_files(cases_root: Path, filename: str):
    for p in sorted(cases_root.glob(f"*/{filename}")):
        if p.is_file():
            yield p.parent.name, p

def plot_one_case(case_id: str, tidy_path: Path, out_subdir: str,
                  step: int, vars_to_plot: list[str],
                  mode: str, dist_kind: str, err_metric: str,
                  mase_m: int, annotate: bool, save_stats: bool, no_ts: bool):
    df = load_tidy(tidy_path)
    methods = pick_methods(df)
    case_dir = tidy_path.parent
    out_dir = case_dir if out_subdir in ("", ".", "./") else (case_dir / out_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    need_mase = (err_metric in {"mase", "all"})
    scales = compute_case_mase_scales(case_dir, vars_to_plot, mase_m) if need_mase else {}

    err_list = ERR_METRICS if err_metric == "all" else [err_metric]

    for variable in vars_to_plot:
        if mode == "stitched":
            df_sub = select_rows_stitched_latest(df, variable)
            title = f"{case_id} | Stitched (latest-available) forecast vs truth — {variable}"
            ts_name = f"ts_stitched_{variable}_actual_vs_pred"
        else:
            latest_per_target = (mode == "latest_step")
            df_sub = select_rows_fixed_step(df, variable, step, latest_per_target)
            title = f"{case_id} | Actual vs Predicted — {variable} (step={step}, mode={mode})"
            ts_name = f"ts_step{step}_{variable}_{mode}_actual_vs_pred"

        if not no_ts:
            plot_timeseries(df_sub, variable, methods, out_dir, title, ts_name)

        for em in err_list:
            scale = scales.get(variable) if em == "mase" else None
            df_err = compute_error_table(df_sub, methods, em, mase_scale=scale)

            if em == "abs":
                ylabel, title_metric = "|prediction − truth|", "Absolute error"
            elif em == "signed":
                ylabel, title_metric = "prediction − truth", "Signed error"
            elif em == "smape":
                ylabel, title_metric = "sMAPE per point (%)", "sMAPE distribution"
            else:
                ylabel, title_metric = "Absolute scaled error (|e| / scale)", "MASE-point distribution"

            if mode == "stitched":
                ttl = f"{case_id} | {title_metric} — {variable} (stitched)"
                base = f"dist_{dist_kind}_stitched_{variable}_{em}"
            else:
                ttl = f"{case_id} | {title_metric} — {variable} (step={step}, mode={mode})"
                base = f"dist_{dist_kind}_step{step}_{variable}_{mode}_{em}"

            if dist_kind in ("box", "both"):
                plot_error_distribution(df_err, methods, out_dir, "box", ttl, ylabel, base + "_box", annotate, save_stats)
            if dist_kind in ("violin", "both"):
                plot_error_distribution(df_err, methods, out_dir, "violin", ttl, ylabel, base + "_violin", annotate, save_stats)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-root", type=str, default="outputs/cases")
    parser.add_argument("--filename", type=str, default="rolling_predictions_tidy.csv")
    parser.add_argument("--out-subdir", type=str, default="plots")

    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--vars", type=str, default="Yd,TSE,TSEC")

    parser.add_argument("--mode", type=str, default="stitched",
                        choices=["stitched", "latest_step", "all_forecasts"],
                        help="stitched: latest forecast per target_date across steps; "
                             "latest_step: fixed-step with latest origin per target; "
                             "all_forecasts: fixed-step showing all origins (no stitching).")

    parser.add_argument("--dist", type=str, default="box", choices=["box", "violin", "both"])
    parser.add_argument("--err", type=str, default="all", choices=["abs", "signed", "smape", "mase", "all"])
    parser.add_argument("--mase-m", type=int, default=1)

    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--no-annotate", action="store_true")
    parser.add_argument("--save-stats", action="store_true")
    parser.add_argument("--no-ts", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    set_plot_style()
    cases_root = Path(args.cases_root)
    if not cases_root.exists():
        raise FileNotFoundError(cases_root)

    vars_to_plot = [v.strip() for v in args.vars.split(",") if v.strip()]
    for v in vars_to_plot:
        if v not in VARS:
            raise ValueError(f"Unknown variable {v}")

    annotate = True
    if args.no_annotate:
        annotate = False
    elif args.annotate:
        annotate = True

    ok, fail = 0, 0
    for case_id, tidy_path in iter_case_tidy_files(cases_root, args.filename):
        try:
            plot_one_case(case_id, tidy_path, args.out_subdir, args.step, vars_to_plot,
                          mode=args.mode, dist_kind=args.dist, err_metric=args.err, mase_m=int(args.mase_m),
                          annotate=annotate, save_stats=bool(args.save_stats), no_ts=bool(args.no_ts))
            ok += 1
            print(f"[OK] {case_id}")
        except Exception as e:
            fail += 1
            print(f"[FAIL] {case_id}: {e}")
            if args.strict:
                raise
    print(f"Done. Cases succeeded: {ok}, failed: {fail}")

if __name__ == "__main__":
    main()
