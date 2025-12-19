# Forecasting + Rolling Evaluation (28 cases) — ETS/MEAN, ARIMA, RNN, LSTM

This project generates **28 synthetic cases** and compares **4 forecasting methods**
under a **rolling-origin (daily update) evaluation**. It also computes multiple
forecast error metrics and runs a **newsvendor-style sensitivity analysis** showing
how forecast uncertainty affects downstream decisions.

## Methods (4)
- **ETS** (Exponential Smoothing baseline; can switch to MEAN if desired)
- **ARIMA** (per-variable SARIMAX)
- **RNN** (sequence-to-sequence with future time features)
- **LSTM** (same architecture but LSTM)

## Metrics
Per variable (`Yd`, `TSE`, `TSEC`) and per bucket (`ALL`, `step_1..step_H`):
- MAE, RMSE, sMAPE, WAPE, MASE

## Folder layout
- `src/` core library
- `run_rolling.py` run a **single case** and produce per-case outputs
- `run_all_cases.py` run **all 28 cases** and aggregate outputs
- `plt_results.py` plot **summary (across cases)** distributions for metrics
- `plt_results_all.py` plot **per-case** time series + error distributions
- `plt_sensitivity.py` plot sensitivity (cost gap) across cases

Outputs:
- `outputs/cases/<case_id>/...` per-case artifacts
- `outputs/metrics_all_cases.csv` aggregated metrics
- `outputs/sensitivity_all_cases.csv` aggregated sensitivity results
- `outputs/plots_compare_allcases/...` summary plots

## Quick start

Create env + install:
```bash
pip install -r requirements.txt
```

Run all 28 cases (rolling horizon default 3):
```bash
python run_all_cases.py
```

Summary plots (all metrics, step_1, all variables):
```bash
python plt_results.py --metrics outputs/metrics_all_cases.csv --metrics-list all --buckets step_1 --vars all --dist box
```

Per-case plots (all error views, step=1, latest-per-target):
```bash
python plt_results_all.py --cases-root outputs/cases --step 1 --err all --dist box --latest --annotate --save-stats
```

Sensitivity plots:
```bash
python plt_sensitivity.py --csv outputs/sensitivity_all_cases.csv
```

## Notes
- The default case generator follows your paper-style normal distributions:
  - `Yd ~ N(150*M, 15*M)` (HUR) or `N(150*M, 1.5*M)` (LUR)
  - `TSE ~ N(150*M, 15*M)` (HUR) or `N(150*M, 1.5*M)` (LUR)
  - `TSEC ~ N(10*M, 1*M)`  (HUR) or `N(10*M, 0.1*M)` (LUR, tied to solar variability)
- If you want to use **MEAN** baseline instead of ETS: set `baseline="MEAN"` in `Config` (see `src/config.py`).
