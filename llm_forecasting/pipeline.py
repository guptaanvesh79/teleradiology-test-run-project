"""Independent resumable hybrid experiment; real stages are launched manually."""

from contextlib import contextmanager
from datetime import UTC, datetime
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from forecasting.artifacts import validate_hourly
from forecasting.config import DEVELOPMENT_ORIGINS
from forecasting.data import hourly_counts, load_studies, sha256_file
from forecasting.metrics import evaluate, fraction
from forecasting.pipeline import FROZEN_FILES, frozen_selection
from llm_forecasting.client import (
    GENERATION,
    GenerationError,
    LLMClient,
    PartialResponseError,
    digest,
)
from .storage import (
    atomic_bytes,
    write_json,
    write_frame,
    read_json,
    hashes,
    verify_hashes,
    runtime,
    repeatability,
)
from .prompts import (
    ALLOCATION_POLICY,
    BATCH_DAYS,
    CALIBRATION_POLICY,
    SYSTEM_PROMPT,
    allocate,
    batches,
    date_keys,
    empirical_ranges,
    history_summary,
    hourly_proportions,
    inspect_reply,
    messages,
    response_schema,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = 5
EXPORT_NAME = "llm_forecast.csv"
REQUEST_COUNTS = {
    "development": {"main": 42, "calibration": 36},
    "final": {"main": 45, "calibration": 36},
    "october": {"main": 16, "calibration": 12},
}
EXPECTED_BATCHES = sum(sum(counts.values()) for counts in REQUEST_COUNTS.values())
MAX_ATTEMPTS = EXPECTED_BATCHES * (GENERATION["retries"] + 1)
SOURCE_FILES = (
    tuple(
        f"llm_forecasting/{name}.py"
        for name in ("__init__", "__main__", "client", "prompts", "pipeline", "storage")
    )
    + FROZEN_FILES
)

RETRY_POLICY = {
    "first_valid_date": True,
    "pending_dates_only": True,
    "maximum_calls_per_batch_invocation": 4,
    "delays_seconds": [1, 2, 4],
}
EXPLORATORY = "Exploratory: September outcomes were already inspected before designing this hybrid."


class IncompleteForecastError(ValueError):
    def __init__(self, keys):
        self.keys = tuple(keys)
        super().__init__(
            f"Stage incomplete: {len(keys)} missing batches. Accepted daily totals are retained; rerun this stage.\n"
            + "\n".join(keys)
        )


def output(root):
    return Path(root) / "artifacts/llm"


@contextmanager
def exclusive(root):
    path = output(root) / ".pipeline.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another LLM pipeline is running") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def promoted_evidence(root):
    path = output(root) / "protocol.json"
    if not path.exists():
        return None
    saved = read_json(path)
    promotion = saved.get("promotion") if isinstance(saved, dict) else None
    if not promotion:
        return None
    original = promotion["original_protocol"]
    if (
        original.get("version") != 1
        or promotion["original_protocol_sha256"]
        != hashlib.sha256(promotion["original_protocol_text"].encode()).hexdigest()
        or json.loads(promotion["original_protocol_text"]) != original
    ):
        raise ValueError("Promotion provenance is invalid")
    verify_journal(root, original)
    prefix = (output(root) / "requests.jsonl").read_bytes()[: original["journal_bytes"]]
    return original, len(prefix.splitlines())


def records(root):
    path = output(root) / "requests.jsonl"
    if not path.exists():
        return []
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        promoted = promoted_evidence(root)
        for index, row in enumerate(rows):
            legacy = promoted is not None and index < promoted[1]
            if legacy and (
                row["protocol_version"] != 1
                or row["stage"] != "development"
                or row["experiment_sha256"] != promoted[0]["experiment_sha256"]
            ):
                raise ValueError("Invalid promoted development evidence")
            if (
                (not legacy and row["protocol_version"] != PROTOCOL_VERSION)
                or row["stage"] not in REQUEST_COUNTS
                or row["phase"] not in ("main", "calibration")
                or row["status"]
                not in (
                    "valid",
                    "partial",
                    "invalid",
                    "http_error",
                    "transport_error",
                    "interrupted",
                )
                or not isinstance(row["request"], dict)
                or row["request_sha256"] != digest(row["request"])
                or not isinstance(row["experiment_sha256"], str)
                or len(row["experiment_sha256"]) != 64
                or not isinstance(row["requested_dates"], list)
                or not row["key"].startswith(
                    f"{row['stage']}/{row['origin']}/{row['run']}/{row['phase']}/"
                )
            ):
                raise ValueError("Incompatible hybrid request journal")
            date_keys(pd.DatetimeIndex(row["requested_dates"]))
        return rows
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        raise ValueError(
            "Malformed or incompatible hybrid journal; preserve it for inspection"
        ) from error


def append_record(root, event):
    path = output(root) / "requests.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def journal_snapshot(root):
    content = (output(root) / "requests.jsonl").read_bytes()
    return {
        "journal_bytes": len(content),
        "journal_sha256": hashlib.sha256(content).hexdigest(),
    }


def verify_journal(root, saved):
    content = (output(root) / "requests.jsonl").read_bytes()[: saved["journal_bytes"]]
    if (
        len(content) != saved["journal_bytes"]
        or hashlib.sha256(content).hexdigest() != saved["journal_sha256"]
    ):
        raise ValueError("Hybrid request evidence changed")


def baseline_files(root):
    frozen_selection(root)
    paths = [
        Path(root) / "forecast.csv",
        Path(root) / "artifacts/selection.json",
        Path(root) / "artifacts/daily_forecast.csv",
        Path(root) / "artifacts/forecast_metadata.json",
    ]
    for name in ("development", "final"):
        paths.extend((Path(root) / "artifacts" / name).rglob("*"))
    return hashes(root, paths)


def protocol_base(root, client):
    if runtime()["python"] != "3.13.18":
        raise ValueError("Use main_env Python 3.13.18")
    return {
        "version": PROTOCOL_VERSION,
        "batch_days": BATCH_DAYS,
        "system_prompt": SYSTEM_PROMPT,
        "schema_example": response_schema(pd.date_range("2026-07-07", periods=2)),
        "schema_policy": "exact requested ISO date keys; strict envelope; independent daily numeric validation",
        "allocation": ALLOCATION_POLICY,
        "calibration": CALIBRATION_POLICY,
        "retry_policy": RETRY_POLICY,
        "generation": dict(GENERATION),
        "provider": client.settings.public(),
        "identity": client.identity(),
        "runtime": runtime(),
        "data_sha256": sha256_file(Path(root) / "data/studies.csv"),
        "sources": hashes(root, [Path(root) / name for name in SOURCE_FILES]),
        "baseline_files": baseline_files(root),
        "development_origins": list(DEVELOPMENT_ORIGINS),
        "development_days": 28,
        "final_origin": "2026-09-01",
        "final_days": 30,
        "final_runs": 3,
        "forecast_origin": "2026-10-01",
        "forecast_days": 31,
        "request_counts": REQUEST_COUNTS,
        "expected_batches": EXPECTED_BATCHES,
        "maximum_attempts": MAX_ATTEMPTS,
        "evaluation_status": EXPLORATORY,
        "api_budget_usd": 0,
        "coherence": "daily/hourly equality enforced by Python, not LLM arithmetic",
    }


def frozen_protocol(root):
    return validate_protocol(root)


def validate_protocol(root, *, committed=True):
    path = output(root) / "protocol.json"
    saved = read_json(path)
    required = {
        "sources",
        "baseline_files",
        "development_files",
        "journal_bytes",
        "journal_sha256",
        "data_sha256",
        "experiment_sha256",
        "provider",
        "identity",
    }
    if not isinstance(saved, dict) or not required <= set(saved):
        raise ValueError("Malformed hybrid protocol; preserve it for inspection")
    base = {
        key: value
        for key, value in saved.items()
        if key
        not in (
            "experiment_sha256",
            "development_files",
            "journal_bytes",
            "journal_sha256",
        )
    }
    if (
        saved.get("version") != PROTOCOL_VERSION
        or saved.get("experiment_sha256") != digest(base)
        or saved.get("allocation") != ALLOCATION_POLICY
        or saved.get("calibration") != CALIBRATION_POLICY
        or saved.get("generation") != GENERATION
        or saved.get("retry_policy") != RETRY_POLICY
        or saved.get("request_counts") != REQUEST_COUNTS
        or saved.get("maximum_attempts") != MAX_ATTEMPTS
        or saved.get("system_prompt") != SYSTEM_PROMPT
        or saved.get("runtime") != runtime()
    ):
        raise ValueError(
            "Hybrid protocol or runtime is incompatible; preserve the experiment"
        )
    for field in ("sources", "baseline_files", "development_files"):
        verify_hashes(root, saved[field])
    if sha256_file(Path(root) / "data/studies.csv") != saved["data_sha256"]:
        raise ValueError("Input data changed after hybrid development")
    verify_journal(root, saved)
    if committed:
        try:
            relative = str(path.relative_to(root))
            if (
                subprocess.check_output(
                    ["git", "show", f"HEAD:{relative}"],
                    cwd=root,
                    stderr=subprocess.PIPE,
                )
                != path.read_bytes()
            ):
                raise ValueError("Commit the exact hybrid protocol before September")
            for name, checksum in {
                **saved["sources"],
                **saved["development_files"],
            }.items():
                value = subprocess.check_output(
                    ["git", "show", f"HEAD:{name}"], cwd=root, stderr=subprocess.PIPE
                )
                if hashlib.sha256(value).hexdigest() != checksum:
                    raise ValueError(
                        "Commit the exact hybrid sources and development results"
                    )
            content = subprocess.check_output(
                ["git", "show", "HEAD:artifacts/llm/requests.jsonl"],
                cwd=root,
                stderr=subprocess.PIPE,
            )
            if (
                len(content) < saved["journal_bytes"]
                or hashlib.sha256(content[: saved["journal_bytes"]]).hexdigest()
                != saved["journal_sha256"]
            ):
                raise ValueError("Commit hybrid development request evidence")
        except subprocess.CalledProcessError as error:
            raise ValueError(
                "Commit hybrid sources, protocol, development and journal first"
            ) from error
    return saved


def check_client(client, saved):
    if (
        client.settings.public() != saved["provider"]
        or client.identity() != saved["identity"]
    ):
        raise ValueError("Provider, model or Ollama changed after hybrid development")


def recover_batch(
    summary, dates, root, client, key, stage, phase, origin, run, experiment_sha
):
    accepted = {}
    previous = records(root)
    if any(
        row["experiment_sha256"] != experiment_sha
        and row["protocol_version"] == PROTOCOL_VERSION
        for row in previous
    ):
        raise ValueError("Hybrid evidence belongs to another experiment; preserve it")

    def pending():
        return dates[[key not in accepted for key in date_keys(dates)]]

    def request():
        missing = pending()
        return messages(summary, dates, missing), response_schema(missing)

    for row in (row for row in previous if row["key"] == key):
        if pending().empty:
            raise ValueError("Cached requests exist after a batch was complete")
        prompt, schema = request()
        _, payload = client.payload(prompt, schema, GENERATION["max_output_tokens"])
        if row["request"] != payload or row["requested_dates"] != date_keys(pending()):
            raise ValueError("Cached hybrid prompt, schema or settings mismatch")
        inspection = row.get("day_validation")
        if inspection is not None:
            normalized, _, _, truncated = client._normalize(
                json.loads(row["raw_response"])
            )
            if truncated or normalized != row["text"]:
                raise ValueError(
                    "Cached daily text differs from its raw provider response"
                )
            if row["status"] not in ("valid", "partial", "invalid"):
                raise ValueError("Cached validation has an incompatible status")
            checked = inspect_reply(row["text"], pending())
            if checked != inspection:
                raise ValueError("Cached daily values differ from the raw response")
            expected_status = (
                "valid"
                if not checked["failed_days"]
                else ("partial" if checked["accepted_days"] else "invalid")
            )
            if row["status"] != expected_status:
                raise ValueError("Cached daily acceptance status was modified")
            accepted.update(checked["accepted_days"])
        elif row["status"] in ("valid", "partial"):
            raise ValueError("Cached success lacks daily validation evidence")
    if pending().empty:
        return accepted
    state = {}

    def factory():
        state.clear()
        state.update(dates=pending(), inspection=None)
        print(f"{key}: requesting dates {date_keys(state['dates'])}", flush=True)
        return request()

    def validate(text):
        inspection = inspect_reply(text, state["dates"])
        state["inspection"] = inspection
        accepted.update(inspection["accepted_days"])
        if inspection["failed_days"]:
            error = PartialResponseError if inspection["accepted_days"] else ValueError
            raise error(
                "Missing or invalid daily totals: "
                + ", ".join(inspection["failed_days"])
            )
        return accepted

    def before_attempt():
        if len(records(root)) >= MAX_ATTEMPTS:
            raise ValueError(
                f"The fixed {MAX_ATTEMPTS}-attempt hybrid budget is exhausted"
            )

    def record(event):
        append_record(
            root,
            {
                **event,
                "protocol_version": PROTOCOL_VERSION,
                "key": key,
                "stage": stage,
                "phase": phase,
                "origin": str(origin),
                "run": run,
                "experiment_sha256": experiment_sha,
                "requested_dates": date_keys(state["dates"]),
                "day_validation": state["inspection"],
            },
        )

    return client.generate(
        [],
        {},
        validate,
        request_factory=factory,
        before_attempt=before_attempt,
        record=record,
        max_output_tokens=GENERATION["max_output_tokens"],
    )


def predict_period(
    series,
    period_origin,
    days,
    root,
    client,
    stage,
    phase,
    outer_origin,
    run,
    experiment_sha,
):
    summary = history_summary(series, period_origin)
    accepted, failures = {}, []
    for dates in batches(period_origin, days):
        key = f"{stage}/{outer_origin}/{run}/{phase}/{pd.Timestamp(period_origin).isoformat()}/{dates[0].isoformat()}"
        try:
            values = recover_batch(
                summary,
                dates,
                root,
                client,
                key,
                stage,
                phase,
                outer_origin,
                run,
                experiment_sha,
            )
            accepted.update(values)
        except GenerationError as error:
            if any(
                row.get("http_status") in (400, 401, 403, 404, 405, 422)
                or (
                    row.get("input_tokens") is not None
                    and row.get("input_token_budget") is not None
                    and row["input_tokens"] > row["input_token_budget"]
                )
                for row in error.records
            ):
                raise
            failures.append(key)
            print(f"{key}: exhausted retries; continuing. {error}", flush=True)
    if failures:
        raise IncompleteForecastError(failures)
    totals = {
        str(day.date()): accepted[str(day.date())]
        for day in pd.date_range(period_origin, periods=days)
    }
    proportions = hourly_proportions(series, period_origin)
    return allocate(totals, proportions), proportions, totals


def forecast_window(series, origin, days, run, stage, root, client, experiment_sha):
    started = time.perf_counter()
    wall_started = datetime.now(UTC)
    failures, calibration, profiles = [], [], []
    periods = [
        (pd.Timestamp(origin) - pd.Timedelta(days=offset), 7, "calibration")
        for offset in (21, 14, 7)
    ]
    periods.append((pd.Timestamp(origin), days, "main"))
    main = None
    for period_origin, period_days, phase in periods:
        try:
            hourly, proportions, totals = predict_period(
                series,
                period_origin,
                period_days,
                root,
                client,
                stage,
                phase,
                origin,
                run,
                experiment_sha,
            )
        except IncompleteForecastError as error:
            failures.extend(error.keys)
            continue
        profiles.append(
            proportions.assign(phase=phase, profile_origin=str(period_origin))
        )
        if phase == "main":
            main = hourly, totals
        else:
            actual = series.reindex(hourly.index)
            if actual.isna().any():
                raise ValueError(
                    "Calibration requires observed counts before the outer origin"
                )
            calibration.append(
                hourly.assign(
                    actual=actual,
                    residual=actual - hourly.forecast,
                    calibration_origin=str(period_origin),
                )
            )
    if failures:
        raise IncompleteForecastError(failures)
    calibration = pd.concat(calibration)
    hourly, daily = empirical_ranges(main[0], calibration.residual)
    daily["llm_total"] = list(main[1].values())
    daily["allocation_error"] = daily.forecast - daily.llm_total
    daily["coherent"] = daily.allocation_error.abs() <= 1e-6
    prior_starts = [
        pd.Timestamp(row["started_at"]).to_pydatetime()
        for row in records(root)
        if row["stage"] == stage and row["origin"] == str(origin) and row["run"] == run
    ]
    return {
        "hourly": hourly,
        "daily": daily,
        "calibration": calibration,
        "proportions": pd.concat(profiles, ignore_index=True),
        "invocation_seconds": time.perf_counter() - started,
        "complete_forecast_seconds": (
            datetime.now(UTC) - min([wall_started, *prior_starts])
        ).total_seconds(),
    }


def reliability(events):
    replies = [row for row in events if row["received_reply"]]
    latencies = [row["latency_seconds"] for row in events]
    invalid = sum(row["status"] in ("invalid", "partial") for row in replies)
    return {
        "attempts": len(events),
        "received_replies": len(replies),
        "invalid_replies": invalid,
        "invalid_reply_share": fraction(invalid, len(replies)),
        "partial_replies": sum(row["status"] == "partial" for row in replies),
        "retry_calls": len(events) - len({row["key"] for row in events}),
        "accepted_daily_values": sum(
            len((row.get("day_validation") or {}).get("accepted_days", {}))
            for row in events
        ),
        "requested_daily_values": sum(len(row["requested_dates"]) for row in events),
        "transport_failures": sum(row["status"] == "transport_error" for row in events),
        "http_failures": sum(row["status"] == "http_error" for row in events),
        "interrupted_attempts": sum(row["status"] == "interrupted" for row in events),
        "mean_call_seconds": float(np.mean(latencies)) if latencies else None,
        "active_request_seconds": float(
            sum(latencies) + sum(row.get("retry_delay_seconds", 0) for row in events)
        ),
        "input_tokens": sum(row.get("input_tokens") or 0 for row in events),
        "output_tokens": sum(row.get("output_tokens") or 0 for row in events),
        "usage_missing_attempts": sum(
            row.get("input_tokens") is None or row.get("output_tokens") is None
            for row in events
        ),
        "api_cost_usd": 0.0,
        "cost_excludes": "hardware and electricity",
        "per_call": [
            {
                "key": row["key"],
                "attempt": row["attempt"],
                "status": row["status"],
                "latency_seconds": row["latency_seconds"],
                "input_tokens": row.get("input_tokens"),
                "output_tokens": row.get("output_tokens"),
            }
            for row in events
        ],
    }


def stage_reliability(root, stage):
    events = [row for row in records(root) if row["stage"] == stage]
    windows = []
    for origin, run in sorted({(row["origin"], row["run"]) for row in events}):
        subset = [
            row for row in events if row["origin"] == origin and row["run"] == run
        ]
        starts = [pd.Timestamp(row["started_at"]) for row in subset]
        ends = [
            pd.Timestamp(row["started_at"])
            + pd.Timedelta(
                seconds=row["latency_seconds"] + row.get("retry_delay_seconds", 0)
            )
            for row in subset
        ]
        windows.append(
            {
                "origin": origin,
                "run": run,
                "elapsed_wall_seconds": (max(ends) - min(starts)).total_seconds(),
                **reliability(subset),
            }
        )
    return {
        "combined": reliability(events),
        "main": reliability([r for r in events if r["phase"] == "main"]),
        "calibration": reliability([r for r in events if r["phase"] == "calibration"]),
        "windows": windows,
        "timing_note": "Call latency includes loading/unloading; active time includes retries. Wall spans include manual pauses on resume and exclude final file-writing overhead.",
    }


def collect_windows(series, windows, stage, root, client, experiment_sha):
    completed, failures = [], []
    for origin, days, run in windows:
        try:
            frames = forecast_window(
                series, origin, days, run, stage, root, client, experiment_sha
            )
            completed.append((origin, days, run, frames))
        except IncompleteForecastError as error:
            failures.extend(error.keys)
    if failures:
        raise IncompleteForecastError(failures)
    return completed


def save_windows(root, series, completed, stage):
    metrics = []
    for origin, days, run, frames in completed:
        prefix = output(root) / stage / f"{origin}_run{run}"
        actual = series.reindex(frames["hourly"].index)
        if stage != "october" and actual.isna().any():
            raise ValueError("Backtest extends beyond observed data")
        hourly, daily = frames["hourly"].copy(), frames["daily"].copy()
        if stage != "october":
            hourly["actual"] = actual
            daily["actual"] = actual.resample("D").sum()
            scores = evaluate(
                actual,
                hourly.assign(crossed_before_sort=False),
                daily,
                quantile_model=True,
            )
            for scope, values in scores.items():
                metrics.append(
                    {
                        "origin": origin,
                        "candidate": f"hybrid_run{run}",
                        "scope": scope,
                        "run": run,
                        "interval_method": "empirical_hybrid_residuals",
                        **values,
                    }
                )
        for name, frame in (
            ("hourly", hourly),
            ("daily", daily),
            ("calibration", frames["calibration"]),
        ):
            write_frame(Path(f"{prefix}_{name}.csv"), frame)
        write_frame(
            Path(f"{prefix}_proportions.csv"),
            frames["proportions"].set_index(
                ["phase", "profile_origin", "weekday", "hour"]
            ),
        )
    base = output(root) / stage
    if metrics:
        write_frame(
            base / "metrics.csv",
            pd.DataFrame(metrics).set_index(["origin", "candidate", "scope"]),
        )
    stats = stage_reliability(root, stage)
    timings = {(origin, run): frames for origin, _, run, frames in completed}
    for window in stats["windows"]:
        frames = timings[(window["origin"], window["run"])]
        window.update(
            complete_forecast_seconds=frames["complete_forecast_seconds"],
            last_invocation_seconds=frames["invocation_seconds"],
        )
    stats["complete_timing_note"] = (
        "Complete forecast time includes calibration, loading, retries and allocation; resumed windows include manual pauses. Last invocation excludes earlier attempts. Final file-writing time is excluded."
    )
    write_json(base / "reliability.json", stats)
    return metrics


def develop(root=ROOT, client=None):
    root = Path(root)
    with exclusive(root):
        if (output(root) / "final/manifest.json").exists():
            raise ValueError(
                "Hybrid September evaluation has started; development is frozen"
            )
        if (output(root) / "protocol.json").exists():
            validate_protocol(root, committed=False)
            print(
                "Hybrid development already complete; commit it and run evaluate-final."
            )
            return
        client = client or LLMClient()
        protocol = protocol_base(root, client)
        experiment_sha = digest(protocol)
        if any(row["experiment_sha256"] != experiment_sha for row in records(root)):
            raise ValueError(
                "Existing hybrid journal belongs to another protocol; preserve it"
            )
        series = hourly_counts(load_studies(root / "data/studies.csv"))
        completed = collect_windows(
            series,
            [(origin, 28, 1) for origin in DEVELOPMENT_ORIGINS],
            "development",
            root,
            client,
            experiment_sha,
        )
        save_windows(root, series, completed, "development")
        protocol.update(
            experiment_sha256=experiment_sha,
            development_files=hashes(root, (output(root) / "development").rglob("*")),
            **journal_snapshot(root),
        )
        write_json(output(root) / "protocol.json", protocol)
        print(
            "Saved hybrid development. Manually commit hybrid sources, protocol, journal and results before evaluate-final."
        )


def verify_manifest(root, manifest):
    if manifest.get("protocol_sha256") != sha256_file(output(root) / "protocol.json"):
        raise ValueError("Hybrid results belong to another protocol")
    if manifest.get("complete"):
        verify_hashes(root, manifest["files"])
        verify_journal(root, manifest)


def evaluate_final(root=ROOT, client=None):
    root = Path(root)
    with exclusive(root):
        protocol = validate_protocol(root)
        path = output(root) / "final/manifest.json"
        if path.exists():
            manifest = read_json(path)
            verify_manifest(root, manifest)
            if manifest.get("complete"):
                print("Preserving completed hybrid September results; no calls made.")
                return
        else:
            manifest = {
                "protocol_sha256": sha256_file(output(root) / "protocol.json"),
                "pre_evaluation_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=root, text=True
                ).strip(),
                "complete": False,
                "evaluation_status": EXPLORATORY,
            }
            write_json(path, manifest)
        client = client or LLMClient()
        check_client(client, protocol)
        series = hourly_counts(load_studies(root / "data/studies.csv"))
        completed = collect_windows(
            series,
            [("2026-09-01", 30, run) for run in (1, 2, 3)],
            "final",
            root,
            client,
            protocol["experiment_sha256"],
        )
        save_windows(root, series, completed, "final")
        write_json(
            output(root) / "final/repeatability.json",
            repeatability([frames["hourly"] for _, _, _, frames in completed]),
        )
        manifest.update(
            complete=True,
            files=hashes(root, [p for p in path.parent.rglob("*") if p != path]),
            **journal_snapshot(root),
        )
        write_json(path, manifest)
        print("Saved three exploratory September hybrid runs.")


def export_october(hourly, path):
    content = (
        hourly[["forecast", "p10", "p90"]]
        .rename_axis("hour_start")
        .to_csv(date_format="%Y-%m-%d %H:%M:%S", float_format="%.10f")
    )
    validate_hourly(pd.read_csv(io.StringIO(content)))
    if Path(path).exists():
        raise ValueError("Refusing to overwrite an existing hybrid October submission")
    atomic_bytes(path, content.encode())


def forecast(root=ROOT, client=None):
    root = Path(root)
    with exclusive(root):
        protocol = validate_protocol(root)
        final = read_json(output(root) / "final/manifest.json")
        verify_manifest(root, final)
        if not final.get("complete"):
            raise ValueError("Complete three hybrid September runs before October")
        path = output(root) / "forecast_manifest.json"
        if path.exists():
            verify_manifest(root, read_json(path))
            print("Preserving validated hybrid October forecast; no calls made.")
            return
        if (root / EXPORT_NAME).exists():
            raise ValueError(
                "Existing hybrid submission has no verified manifest; preserve it"
            )
        client = client or LLMClient()
        check_client(client, protocol)
        series = hourly_counts(load_studies(root / "data/studies.csv"))
        completed = collect_windows(
            series,
            [("2026-10-01", 31, 1)],
            "october",
            root,
            client,
            protocol["experiment_sha256"],
        )
        save_windows(root, series, completed, "october")
        export_october(completed[0][3]["hourly"], root / EXPORT_NAME)
        paths = [root / EXPORT_NAME, *(output(root) / "october").rglob("*")]
        write_json(
            path,
            {
                "complete": True,
                "protocol_sha256": sha256_file(output(root) / "protocol.json"),
                "files": hashes(root, paths),
                **journal_snapshot(root),
            },
        )
        print(f"Saved {EXPORT_NAME}: 744 October hours.")
