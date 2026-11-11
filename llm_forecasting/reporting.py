"""Offline hybrid comparison; replace only this approach's tagged report cells."""

from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.api import load_artifacts
from .explain import TEMPLATE, explanation_facts, render, validate_explanation

from .pipeline import (
    EXPLORATORY,
    ROOT,
    atomic_bytes,
    exclusive,
    output,
    read_json,
    validate_protocol,
    verify_manifest,
    write_frame,
)

TAG = "llm-forecasting"


def saved_report_data(root=ROOT):
    root = Path(root)
    protocol = validate_protocol(root)
    for name in ("final/manifest.json", "forecast_manifest.json"):
        manifest = read_json(output(root) / name)
        if not manifest.get("complete"):
            raise ValueError("Complete hybrid September and October before reporting")
        verify_manifest(root, manifest)
    tables = []
    for stage in ("development", "final"):
        for method, path in (
            ("Part 1", root / "artifacts" / stage / "metrics.csv"),
            ("LLM hybrid", output(root) / stage / "metrics.csv"),
        ):
            frame = pd.read_csv(path)
            tables.append(frame.assign(method_source=method, stage=stage))
    comparison = pd.concat(tables, ignore_index=True)
    selection = read_json(root / "artifacts/selection.json")["selected_candidate"]
    diagnostics = []
    for origin, days, run in [(o, 28, 1) for o in protocol["development_origins"]] + [
        ("2026-09-01", 30, i) for i in (1, 2, 3)
    ]:
        stage = "final" if origin == "2026-09-01" else "development"
        prefix = output(root) / stage / f"{origin}_run{run}"
        hourly = pd.read_csv(
            f"{prefix}_hourly.csv", parse_dates=["hour_start"]
        ).set_index("hour_start")
        expected = pd.date_range(origin, periods=days * 24, freq="h")
        if not hourly.index.equals(expected) or hourly.actual.isna().any():
            raise ValueError("Hybrid comparison requires complete observed windows")
        reference_name = (
            f"{origin}_{selection}_hourly.csv" if stage == "final" else None
        )
        if reference_name is not None:
            reference = pd.read_csv(
                root / "artifacts/final" / reference_name, parse_dates=["hour_start"]
            ).set_index("hour_start")
            if not reference.index.equals(hourly.index) or not np.array_equal(
                reference.actual, hourly.actual
            ):
                raise ValueError(
                    "Comparison methods do not share identical September actuals"
                )
        profiles = pd.read_csv(f"{prefix}_proportions.csv")
        main = profiles.loc[profiles.phase == "main"].pivot(
            index="weekday", columns="hour", values="proportion"
        )
        actual_daily = hourly.actual.resample("D").sum()
        oracle = np.array(
            [
                actual_daily.loc[t.normalize()] * main.loc[t.day_of_week, t.hour]
                for t in hourly.index
            ]
        )
        denominator = float(hourly.actual.sum())
        diagnostics.append(
            {
                "origin": origin,
                "run": run,
                "daily_volume_absolute_error": float(
                    (hourly.forecast.resample("D").sum() - actual_daily).abs().sum()
                ),
                "oracle_daily_volume_hourly_wape": (
                    float(np.abs(oracle - hourly.actual).sum() / denominator)
                    if denominator
                    else None
                ),
                "note": "Oracle uses actual daily totals for diagnosis only; not a valid forecast or additive error decomposition.",
            }
        )
    september = comparison.loc[
        (comparison.stage == "final") & (comparison.scope == "all")
    ]
    classical = float(
        september.loc[
            (september.method_source == "Part 1") & (september.candidate == selection),
            "hourly_wape",
        ].iloc[0]
    )
    hybrid = float(
        september.loc[september.method_source == "LLM hybrid", "hourly_wape"].mean()
    )
    recommendation = (
        f"Mean September hourly WAPE: selected Part 1 {classical:.2%}, LLM hybrid {hybrid:.2%}. "
        "Keep Part 1 as the established operational baseline and use validated LLM explanations; "
        "any hybrid promotion requires fresh future evaluation and interval checks. "
        + EXPLORATORY
    )
    examples = read_json(output(root) / "explanations.json")
    part1_hourly, part1_daily, _ = load_artifacts(root)
    facts = explanation_facts(part1_hourly, part1_daily)
    live, fallback = examples["live"], examples["simulated_provider_failure"]
    accepted = [row for row in live["records"] if row["status"] == "valid"]
    if (
        live["source"] != "llm"
        or live["facts"] != facts
        or not accepted
        or validate_explanation(accepted[-1]["text"], facts) != live["explanation"]
    ):
        raise ValueError("Saved live explanation does not match validated Part 1 facts")
    if (
        fallback["source"] != "template"
        or fallback["facts"] != facts
        or fallback["explanation"] != render(TEMPLATE, facts)
    ):
        raise ValueError("Saved fallback explanation does not match Part 1 facts")
    daily_failures = []
    for run in (1, 2, 3):
        daily = pd.read_csv(output(root) / f"final/2026-09-01_run{run}_daily.csv")
        daily["run"] = run
        daily["absolute_error"] = (daily.forecast - daily.actual).abs()
        daily_failures.append(daily)
    failures = (
        pd.concat(daily_failures)
        .sort_values("absolute_error", ascending=False)
        .drop_duplicates("date")
        .head(3)
    )
    return {
        "comparison": comparison,
        "explanations": examples,
        "failure_examples": failures,
        "diagnostics": pd.DataFrame(diagnostics),
        "recommendation": recommendation,
        "reliability": {
            stage: read_json(output(root) / stage / "reliability.json")
            for stage in ("development", "final", "october")
        },
        "repeatability": read_json(output(root) / "final/repeatability.json"),
    }


def notebook_cells(root=ROOT):
    import nbformat

    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    root = Path(root).resolve()
    required = [
        "protocol.json",
        "final/manifest.json",
        "forecast_manifest.json",
        "explanations.json",
    ]
    if not all((output(root) / name).exists() for name in required) or not read_json(
        output(root) / "final/manifest.json"
    ).get("complete"):
        return [
            md(
                "## Part 2 status\n\nThe daily-volume LLM hybrid is implemented. Complete its manual evaluation, October forecast and explanation demonstration before generating the final report."
            )
        ]
    cells = [
        md("""## LLM hybrid — daily volume and historical hourly allocation

**Exploratory comparison:** September outcomes were known before this method was designed.
Qwen predicts two daily point totals per call. Python allocates each total using historical
weekday/hour volume proportions; these points are not directly predicted hourly medians.
Daily/hourly coherence is enforced by Python, not evidence of LLM arithmetic accuracy.

Three prior seven-day hybrid forecasts supply empirical hourly and daily residual bands.
Daily ranges are calibrated separately, never sums of hourly bounds. Only 21 residuals per
clock hour are available. These ranges have no guaranteed 80% coverage and seven-day
calibration may miss late-month uncertainty. Sparse days remain recorded observations.
Timezone is unspecified. Profiles cannot anticipate new hospitals, holidays or demand shifts.

This is a hybrid deviation from L1: the LLM predicts daily totals rather than direct hourly
10th/50th/90th quantiles. The previous direct experiment and its evidence were deleted at
user request after selecting this approach; they are not presented as retained results.
"""),
        code("""from pathlib import Path
from io import BytesIO
import json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
from llm_forecasting.reporting import saved_report_data
HYBRID_ROOT = Path.cwd().resolve()
hybrid_report = saved_report_data(HYBRID_ROOT)
display(hybrid_report['comparison'])
display(hybrid_report['diagnostics'])
print(hybrid_report['recommendation'])
"""),
        code("""for stage, result in hybrid_report['reliability'].items():
    print(stage, json.dumps({k: {field: value for field, value in result[k].items() if field != 'per_call'} for k in ('main', 'calibration', 'combined')}, indent=2))
    display(pd.DataFrame(result['windows'])[['origin', 'run', 'complete_forecast_seconds', 'last_invocation_seconds', 'active_request_seconds']])
print('September repeatability:', json.dumps(hybrid_report['repeatability'], indent=2))
print('US$0 API charges; hardware and electricity excluded. Timing includes calibration and retries.')
daily = pd.read_csv(HYBRID_ROOT / 'artifacts/llm/october/2026-10-01_run1_daily.csv', parse_dates=['date'])
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(daily.date, daily.forecast, label='Hybrid daily point total')
ax.fill_between(daily.date, daily.p10, daily.p90, alpha=0.2, label='Separate empirical daily range')
ax.set(title='October hybrid daily volume', ylabel='Recorded studies', xlabel='Source-clock date')
ax.legend()
fig.autofmt_xdate()
chart_buffer = BytesIO()
fig.savefig(chart_buffer, format='png', bbox_inches='tight')
display(Image(data=chart_buffer.getvalue()))
plt.close(fig)
"""),
    ]
    cells.append(code("""display(hybrid_report['failure_examples'])
print('These are measured forecast errors, not invalid JSON examples.')
print('Live validated explanation:', hybrid_report['explanations']['live']['explanation'])
print('Simulated outage fallback:', hybrid_report['explanations']['simulated_provider_failure']['explanation'])
print('Coherence is enforced by Python allocation and does not test model arithmetic.')
"""))
    for cell in cells:
        cell.metadata["tags"] = [TAG]
    return cells


def build_report(root=ROOT):
    import nbformat
    from nbclient import NotebookClient

    root = Path(root)
    with exclusive(root):
        data = saved_report_data(root)
        path = root / "report.ipynb"
        notebook = nbformat.read(path, as_version=4)
        extra = nbformat.v4.new_notebook(
            cells=notebook_cells(root), metadata=dict(notebook.metadata)
        )
        executed = NotebookClient(extra, timeout=120, kernel_name="main_env").execute(
            cwd=str(root)
        )
        notebook.cells = [
            cell for cell in notebook.cells if TAG not in cell.metadata.get("tags", [])
        ] + executed.cells
        write_frame(
            output(root) / "comparison.csv",
            data["comparison"].set_index(
                ["stage", "method_source", "candidate", "scope"]
            ),
        )
        write_frame(
            output(root) / "allocation_diagnostics.csv",
            data["diagnostics"].set_index(["origin", "run"]),
        )
        atomic_bytes(path, nbformat.writes(notebook).encode())
        print(
            "Appended executed hybrid sections; existing notebook cells and charts preserved."
        )


def build_combined_report(root=ROOT):
    build_report(root)
