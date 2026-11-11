"""The experiment specification, frozen before the September evaluation."""

from dataclasses import asdict, dataclass

import pandas as pd

DATA_START = pd.Timestamp("2026-06-01")
DATA_END = pd.Timestamp("2026-10-01")  # exclusive
FINAL_ORIGIN = pd.Timestamp("2026-09-01")
FORECAST_ORIGIN = DATA_END
FORECAST_END = pd.Timestamp("2026-11-01")
DEVELOPMENT_ORIGINS = ("2026-07-07", "2026-07-21", "2026-08-04")
FEATURES = (
    "hour",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "days_since_start",
)
PARAMETERS = {
    "max_iter": 200,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "random_state": 42,
    "early_stopping": False,
}
SPARSE_DAY_RATIO = 0.1
CALIBRATION_WEEKS = 3


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    exclude_sparse: bool = False


# This order is the final simplicity tie-break: baseline, point, quantile;
# retain observations before introducing an exclusion rule.
CANDIDATES = (
    Candidate("seasonal_naive", "naive"),
    Candidate("poisson_raw", "poisson"),
    Candidate("poisson_filtered", "poisson", True),
    Candidate("quantile_raw", "quantile"),
    Candidate("quantile_filtered", "quantile", True),
)


def specification():
    return {
        "features": list(FEATURES),
        "parameters": PARAMETERS,
        "candidates": [asdict(c) for c in CANDIDATES],
        "sparse_day_ratio": SPARSE_DAY_RATIO,
        "calibration_weeks": CALIBRATION_WEEKS,
        "development_origins": list(DEVELOPMENT_ORIGINS),
        "development_days": 28,
        "final_origin": str(FINAL_ORIGIN.date()),
        "final_days": 30,
        "selection_rule": [
            "mean development hourly WAPE",
            "mean development daily WAPE",
            "candidate order (simplicity)",
        ],
    }
