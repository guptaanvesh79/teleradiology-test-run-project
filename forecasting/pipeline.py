"""Explicit development, frozen holdout, and production-artifact stages."""

import hashlib
import importlib.metadata
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .artifacts import validate_daily, validate_hourly
from .config import (
    CANDIDATES,
    DATA_END,
    DATA_START,
    DEVELOPMENT_ORIGINS,
    FINAL_ORIGIN,
    FORECAST_END,
    FORECAST_ORIGIN,
    specification,
)
from .data import (
    day_audit,
    hourly_counts,
    load_studies,
    sha256_file,
    source_consistency,
)
from .metrics import evaluate
from .models import forecast_candidate, ordered_band, run_origin

ROOT = Path(__file__).resolve().parents[1]
FROZEN_FILES = (
    "forecasting/config.py",
    "forecasting/data.py",
    "forecasting/features.py",
    "forecasting/models.py",
    "forecasting/metrics.py",
    "forecasting/artifacts.py",
    "forecasting/pipeline.py",
    "requirements.txt",
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(frame, path, date_format="%Y-%m-%d %H:%M:%S"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=True, date_format=date_format, float_format="%.10f")


def fingerprint(root=ROOT):
    return {name: sha256_file(root / name) for name in FROZEN_FILES}


def runtime_versions():
    return {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "scikit-learn")
    }


def dataset(root=ROOT):
    frame = load_studies(root / "data/studies.csv")
    return frame, hourly_counts(frame)


def evaluate_windows(series, origins, days, output):
    records = []
    for origin in origins:
        origin = pd.Timestamp(origin)
        label = str(origin.date())
        write_csv(
            day_audit(series.loc[series.index < origin]),
            output / f"training_audit_{label}.csv",
            "%Y-%m-%d",
        )
        for candidate in CANDIDATES:
            print(f"{label}: {candidate.name}, {days} days", flush=True)
            prediction, daily = run_origin(series, origin, days, candidate)
            actual = series.reindex(prediction.index)
            if actual.isna().any():
                raise ValueError(
                    "Evaluation window extends beyond available observations"
                )
            write_csv(
                prediction.assign(actual=actual),
                output / f"{label}_{candidate.name}_hourly.csv",
            )
            write_csv(
                daily.assign(actual=actual.resample("D").sum()),
                output / f"{label}_{candidate.name}_daily.csv",
                "%Y-%m-%d",
            )
            scores = evaluate(actual, prediction, daily, candidate.family == "quantile")
            for scope, metrics in scores.items():
                records.append(
                    {
                        "origin": label,
                        "candidate": candidate.name,
                        "scope": scope,
                        **metrics,
                    }
                )
    result = pd.DataFrame(records)
    result.to_csv(output / "metrics.csv", index=False, float_format="%.10f")
    return result


def rank_candidates(metrics):
    full = metrics.loc[metrics.scope == "all"]
    ranks = full.groupby("candidate")[["hourly_wape", "daily_wape"]].mean()
    ranks["simplicity"] = [
        next(i for i, c in enumerate(CANDIDATES) if c.name == name)
        for name in ranks.index
    ]
    return ranks.sort_values(["hourly_wape", "daily_wape", "simplicity"]).reset_index()


def develop(root=ROOT):
    if (root / "artifacts/final/manifest.json").exists():
        raise ValueError("Development is frozen: September has already been evaluated")
    frame, full_series = dataset(root)
    # September targets are excluded before any exploratory statistics or fitting.
    history = full_series.loc[full_series.index < FINAL_ORIGIN]
    historical_rows = frame.loc[frame.arrival_time < FINAL_ORIGIN]
    output = root / "artifacts/development"
    output.mkdir(parents=True, exist_ok=True)
    audit = day_audit(history)
    write_json(
        root / "data/checksums.json",
        {
            name: sha256_file(root / "data" / name)
            for name in ("studies.csv", "DATA_DICTIONARY.md")
        },
    )
    write_json(
        output / "data_quality.json",
        {
            "exploration_cutoff_exclusive": str(FINAL_ORIGIN),
            "rows": len(historical_rows),
            "hours": len(history),
            "empty_hours": int(history.eq(0).sum()),
            "near_empty_days": [
                str(date.date()) for date in audit.index[audit.near_empty]
            ],
            "arrivals_by_stored_hour": {
                str(h): int(v)
                for h, v in history.groupby(history.index.hour).sum().items()
            },
            **source_consistency(historical_rows),
        },
    )
    metrics = evaluate_windows(history, DEVELOPMENT_ORIGINS, 28, output)
    ranks = rank_candidates(metrics)
    ranks.to_csv(output / "ranking.csv", index=False, float_format="%.10f")
    selection = {
        "selected_candidate": str(ranks.iloc[0].candidate),
        "specification": specification(),
        "implementation_sha256": fingerprint(root),
        "data_sha256": sha256_file(root / "data/studies.csv"),
        "runtime_versions": runtime_versions(),
        "development_metrics_sha256": sha256_file(output / "metrics.csv"),
        "development_ranking_sha256": sha256_file(output / "ranking.csv"),
        "selection_uses_only_data_before": str(FINAL_ORIGIN),
    }
    write_json(root / "artifacts/selection.json", selection)
    print(ranks.to_string(index=False), flush=True)
    print(
        "Commit code, selection, and development results before evaluate-final.",
        flush=True,
    )


def frozen_selection(root=ROOT):
    """Validate the frozen settings, inputs, results, and original Git snapshot.

    Source hashes describe the archived pre-evaluation implementation. Later
    source edits are tracked in Git rather than a separate maintenance file.
    """
    path = root / "artifacts/selection.json"
    selection = read_json(path)
    if selection["specification"] != specification():
        raise ValueError(
            "Frozen model configuration changed; do not tune after September"
        )
    if selection["data_sha256"] != sha256_file(root / "data/studies.csv"):
        raise ValueError("Input data changed after development")
    if selection["runtime_versions"] != runtime_versions():
        raise ValueError("Modelling dependency versions changed after development")
    for filename, field in (
        ("metrics.csv", "development_metrics_sha256"),
        ("ranking.csv", "development_ranking_sha256"),
    ):
        if sha256_file(root / "artifacts/development" / filename) != selection[field]:
            raise ValueError("Frozen development results changed")
    try:
        committed = subprocess.check_output(
            ["git", "show", "HEAD:artifacts/selection.json"],
            cwd=root,
            stderr=subprocess.PIPE,
        )
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        reference_commit = commit
        manifest_path = root / "artifacts/final/manifest.json"
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            if manifest["selection_sha256"] != sha256_file(path):
                raise ValueError(
                    "Final evaluation belongs to a different frozen selection"
                )
            reference_commit = manifest["pre_evaluation_commit"]
        for name, checksum in selection["implementation_sha256"].items():
            content = subprocess.check_output(
                ["git", "show", f"{reference_commit}:{name}"],
                cwd=root,
                stderr=subprocess.PIPE,
            )
            if hashlib.sha256(content).hexdigest() != checksum:
                raise ValueError(
                    f"Archived implementation does not match its frozen hash: {name}"
                )
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise ValueError(
            "Commit the modelling code and selection before evaluating September"
        ) from error
    if committed != path.read_bytes():
        raise ValueError("The selection must match its committed version")
    return selection, commit


def evaluate_final(root=ROOT):
    selection, commit = frozen_selection(root)
    output = root / "artifacts/final"
    manifest = output / "manifest.json"
    if manifest.exists():
        saved = read_json(manifest)
        if saved["selection_sha256"] != sha256_file(root / "artifacts/selection.json"):
            raise ValueError("Final evaluation belongs to a different frozen selection")
        if saved["metrics_sha256"] != sha256_file(output / "metrics.csv"):
            raise ValueError("Saved September metrics were modified")
        print(
            "September was already evaluated; preserving the frozen result.", flush=True
        )
        return
    _, series = dataset(root)
    output.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_windows(series, [FINAL_ORIGIN], 30, output)
    write_json(
        manifest,
        {
            "selection_sha256": sha256_file(root / "artifacts/selection.json"),
            "metrics_sha256": sha256_file(output / "metrics.csv"),
            "pre_evaluation_commit": commit,
            "evaluated_at_utc": datetime.now(UTC).isoformat(),
            "selected_candidate": selection["selected_candidate"],
            "policy": "September is evaluation only. No reselection or tuning.",
        },
    )
    print(
        metrics.loc[
            metrics.scope == "all",
            ["candidate", "hourly_wape", "daily_wape", "bias", "coverage"],
        ].to_string(index=False),
        flush=True,
    )


def export_forecast(prediction, daily, root):
    hourly = prediction[["forecast", "p10", "p90"]].round(10)
    hourly.index.name = "hour_start"
    write_csv(hourly, root / "forecast.csv")
    parsed = validate_hourly(pd.read_csv(root / "forecast.csv"))
    daily = daily.copy()
    daily["forecast"] = parsed.forecast.resample("D").sum()
    daily["p10"], daily["p90"] = ordered_band(daily.forecast, daily.p10, daily.p90)
    write_csv(daily, root / "artifacts/daily_forecast.csv", "%Y-%m-%d")
    validate_daily(pd.read_csv(root / "artifacts/daily_forecast.csv"), parsed)
    return parsed, daily


def forecast(root=ROOT):
    selection, _ = frozen_selection(root)
    manifest_path = root / "artifacts/final/manifest.json"
    if not manifest_path.exists():
        raise ValueError(
            "Evaluate the frozen September holdout before exporting October"
        )
    manifest = read_json(manifest_path)
    if manifest["selection_sha256"] != sha256_file(root / "artifacts/selection.json"):
        raise ValueError("September and October must use the same frozen selection")
    if manifest["metrics_sha256"] != sha256_file(root / "artifacts/final/metrics.csv"):
        raise ValueError("Saved final evaluation metrics changed")
    _, series = dataset(root)
    candidate = next(c for c in CANDIDATES if c.name == selection["selected_candidate"])
    future = pd.date_range(
        FORECAST_ORIGIN, FORECAST_END, freq="h", inclusive="left", name="hour_start"
    )
    prediction, daily = forecast_candidate(series, future, candidate)
    hourly, daily = export_forecast(prediction, daily, root)
    write_csv(
        day_audit(series),
        root / "artifacts/final/training_audit_2026-10-01.csv",
        "%Y-%m-%d",
    )
    write_json(
        root / "artifacts/forecast_metadata.json",
        {
            "model": candidate.name,
            "forecast_origin": "2026-10-01",
            "training_start": str(DATA_START),
            "training_end_exclusive": str(DATA_END),
            "timezone": "unspecified",
            "point_estimate": "hourly_median"
            if candidate.family == "quantile"
            else "hourly_point_forecast",
            "daily_point_estimate": "sum_of_hourly_medians"
            if candidate.family == "quantile"
            else "sum_of_hourly_point_forecasts",
            "hourly_interval": "direct_quantiles"
            if candidate.family == "quantile"
            else "hour_conditioned_calibration_residual_percentiles",
            "daily_interval": "percentiles of whole-day residuals from three prior weekly forecasts",
            "daily_bounds_are_not_sums_of_hourly_quantiles": True,
            "hours": len(hourly),
            "days": len(daily),
            "forecast_total": float(hourly.forecast.sum()),
            "selection_sha256": sha256_file(root / "artifacts/selection.json"),
            "data_sha256": selection["data_sha256"],
            "pre_holdout_commit": manifest["pre_evaluation_commit"],
            "forecast_csv_sha256": sha256_file(root / "forecast.csv"),
            "daily_csv_sha256": sha256_file(root / "artifacts/daily_forecast.csv"),
        },
    )
    print(
        f"Saved {len(hourly)} October hours; model={candidate.name}; total={hourly.forecast.sum():,.2f}",
        flush=True,
    )
