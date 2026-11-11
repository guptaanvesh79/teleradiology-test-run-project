"""Small persistence and evaluation helpers shared by the LLM flow."""

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import pandas as pd
from forecasting.data import sha256_file


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".pending-", delete=False
        ) as file:
            temporary = Path(file.name)
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def write_frame(path, frame):
    atomic_bytes(
        path,
        frame.to_csv(
            index=True, date_format="%Y-%m-%d %H:%M:%S", float_format="%.10f"
        ).encode(),
    )


def hashes(root, paths):
    return {
        str(Path(path).relative_to(root)): sha256_file(path)
        for path in sorted(paths)
        if path.is_file()
    }


def verify_hashes(root, expected):
    for name, checksum in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f"Frozen file changed or is missing: {name}")


def runtime():
    return {
        "python": platform.python_version(),
        **{
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "httpx", "pydantic")
        },
    }


def repeatability(frames):
    if len(frames) != 3 or any(
        not frame.index.equals(frames[0].index) for frame in frames
    ):
        raise ValueError("Repeatability requires three complete aligned runs")
    matrix = pd.concat(
        [frame.forecast.rename(f"run{i}") for i, frame in enumerate(frames, 1)], axis=1
    )
    daily = matrix.resample("D").sum()
    totals = matrix.sum()
    return {
        "monthly_totals": totals.to_dict(),
        "monthly_total_range": float(totals.max() - totals.min()),
        "pairwise_hourly_mae": [
            {
                "runs": [f"run{i+1}", f"run{j+1}"],
                "mae": float((matrix.iloc[:, i] - matrix.iloc[:, j]).abs().mean()),
            }
            for i in range(3)
            for j in range(i + 1, 3)
        ],
        "mean_hourly_std": float(matrix.std(axis=1, ddof=0).mean()),
        "max_hourly_std": float(matrix.std(axis=1, ddof=0).max()),
        "daily_total_std": {
            str(day.date()): float(value)
            for day, value in daily.std(axis=1, ddof=0).items()
        },
        "std_definition": "population standard deviation across the three realized runs",
    }
