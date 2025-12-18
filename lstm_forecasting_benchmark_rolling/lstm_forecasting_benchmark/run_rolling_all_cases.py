from pathlib import Path
import pandas as pd

from src.config import Config
from run_rolling_auto import run_one


def case_seed(base: int, M: int, yd_level: str, solar_level: str) -> int:
    """
    Deterministic per-case seed so different cases do not reuse identical random draws.
    """
    y = 0 if yd_level == "HUR" else 1
    s = 0 if solar_level == "HUR" else 1
    return base + M * 100 + y * 10 + s


def main():
    cfg = Config()
    out_root = Path("outputs") / "cases"
    out_root.mkdir(parents=True, exist_ok=True)

    Ms = list(range(4, 11))
    yd_levels = ["HUR", "LUR"]      # rice demand volatility
    solar_levels = ["HUR", "LUR"]   # solar volatility (TSE + TSEC)

    all_metrics = []
    for M in Ms:
        for yd in yd_levels:
            for sol in solar_levels:
                case_id = f"M{M}_Y{yd}_S{sol}"
                out_dir = out_root / case_id
                seed = case_seed(cfg.random_seed, M, yd, sol)
                df_metrics = run_one(cfg, out_dir=out_dir, M=M, yd_level=yd, solar_level=sol, seed=seed)
                all_metrics.append(df_metrics)

    df_all = pd.concat(all_metrics, ignore_index=True)
    df_all.to_csv(Path("outputs") / "metrics_all_cases.csv", index=False)

    # Optional quick pivot for step_1 WAPE
    step1 = df_all[df_all["bucket"] == "step_1"].copy()
    pivot = (
        step1.pivot_table(
            index=["M", "yd_level", "solar_level", "variable"],
            columns="model",
            values="WAPE",
            aggfunc="mean"
        )
        .reset_index()
        .sort_values(["yd_level", "solar_level", "M", "variable"])
    )
    pivot.to_csv(Path("outputs") / "metrics_all_cases_step1_wape_pivot.csv", index=False)

    print("[OK] All 28 cases finished.")
    print("  - outputs/metrics_all_cases.csv")
    print("  - outputs/metrics_all_cases_step1_wape_pivot.csv")


if __name__ == "__main__":
    main()
