\
from dataclasses import dataclass

@dataclass
class Config:
    # Data
    random_seed: int = 42
    start_date: str = "2023-01-01"
    history_days: int = 730

    # Direct mode (one-shot forecast)
    test_days: int = 90
    seq_length: int = 60
    horizon: int = 90

    # Rolling mode (daily update)
    rolling_horizon: int = 3
    rolling_test_days: int = 90   # number of days evaluated by rolling origins (>= rolling_horizon)
    rolling_step: int = 1

    # Train/val split over training samples (built from history)
    train_split: float = 0.75
    val_split: float = 0.15

    # Deep model training
    epochs: int = 80
    batch_size: int = 32
    patience: int = 10

    # ARIMA settings (per series)
    use_auto_arima: bool = False
    arima_order: tuple = (2, 0, 2)
    seasonal_order: tuple = (1, 0, 1, 7)  # weekly seasonality

    # Sensitivity (newsvendor) costs
    unit_production_cost: float = 1.0
    overage_cost: float = 0.2
    underage_cost: float = 1.0

    # Paths
    data_csv: str = "data/synthetic_daily.csv"
    out_xlsx: str = "outputs/forecast_comparison.xlsx"
    metrics_csv: str = "outputs/metrics_summary.csv"
    sensitivity_csv: str = "outputs/sensitivity_costs.csv"
