from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class Config:
    # -------------------------
    # Case design (28 cases)
    # -------------------------
    M_list: List[int] = field(default_factory=lambda: [4, 5, 6, 7, 8, 9, 10])
    yd_levels: List[str] = field(default_factory=lambda: ["HUR", "LUR"])   # rice yield uncertainty: High/Low
    solar_levels: List[str] = field(default_factory=lambda: ["HUR", "LUR"])# solar uncertainty: High/Low (drives TSE & TSEC std)

    # -------------------------
    # Data length
    # -------------------------
    random_seed: int = 42
    start_date: str = "2023-01-01"
    history_days: int = 730

    # Rolling evaluation
    rolling_horizon: int = 3
    rolling_test_days: int = 90   # number of days evaluated by rolling origins (>= rolling_horizon)
    rolling_step: int = 1

    # Deep model window and training
    seq_length: int = 60
    train_split: float = 0.75
    val_split: float = 0.15
    epochs: int = 40
    batch_size: int = 32
    patience: int = 8

    # Baseline selection: "ETS" (recommended) or "MEAN"
    baseline: str = "ETS"
    ets_use_seasonal: bool = False
    ets_seasonal_periods: int = 7

    # ARIMA settings (per series)
    use_auto_arima: bool = False
    arima_order: Tuple[int, int, int] = (2, 0, 2)
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 7)

    # MASE scale: naive(m)
    mase_m: int = 1

    # Sensitivity (newsvendor) costs
    unit_production_cost: float = 1.0
    overage_cost: float = 0.2
    underage_cost: float = 1.0

    # Outputs
    outputs_root: str = "outputs"
