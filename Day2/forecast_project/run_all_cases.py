import argparse
import os
import pandas as pd

from src.config import Config
from src.data import enumerate_cases
from run_rolling import run_one_case

def main():
    cfg = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=cfg.outputs_root)
    ap.add_argument("--strict", action="store_true", help="Stop at first failing case.")
    args = ap.parse_args()

    out_root = args.out
    os.makedirs(out_root, exist_ok=True)

    cases = enumerate_cases(cfg.M_list, cfg.yd_levels, cfg.solar_levels)

    all_metrics = []
    all_sens = []

    ok, fail = 0, 0
    for case in cases:
        try:
            df_m, df_s = run_one_case(
                cfg=cfg,
                case_id=case["case_id"],
                M=case["M"],
                yd_level=case["yd_level"],
                solar_level=case["solar_level"],
                out_root=out_root,
            )
            all_metrics.append(df_m)
            all_sens.append(df_s)
            ok += 1
            print(f"[OK] {case['case_id']}")
        except Exception as e:
            fail += 1
            print(f"[FAIL] {case['case_id']}: {e}")
            if args.strict:
                raise

    if all_metrics:
        df_allm = pd.concat(all_metrics, ignore_index=True)
        df_allm.to_csv(os.path.join(out_root, "metrics_all_cases.csv"), index=False)
        print(f"Saved: {os.path.join(out_root, 'metrics_all_cases.csv')}")

    if all_sens:
        df_alls = pd.concat(all_sens, ignore_index=True)
        df_alls.to_csv(os.path.join(out_root, "sensitivity_all_cases.csv"), index=False)
        print(f"Saved: {os.path.join(out_root, 'sensitivity_all_cases.csv')}")

    print(f"Done. Cases succeeded: {ok}, failed: {fail}")

if __name__ == "__main__":
    main()
