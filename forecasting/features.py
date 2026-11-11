"""Calendar-only features: no arrival counts or future observations."""

import numpy as np
import pandas as pd

from .config import DATA_START, FEATURES


def calendar_features(index):
    index = pd.DatetimeIndex(index)
    if index.tz is not None or index.hasnans:
        raise ValueError("Features require valid timezone-naive timestamps")
    hour = index.hour.to_numpy()
    weekday = index.day_of_week.to_numpy()
    return pd.DataFrame(
        {
            "hour": hour,
            "day_of_week": weekday,
            "is_weekend": (weekday >= 5).astype(int),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "weekday_sin": np.sin(2 * np.pi * weekday / 7),
            "weekday_cos": np.cos(2 * np.pi * weekday / 7),
            "days_since_start": (index.normalize() - DATA_START).days,
        },
        index=index,
    )[list(FEATURES)]
