"""Generate a report from saved results; never fit models or select on September."""

import pandas as pd

from .pipeline import ROOT, read_json


def percent(value):
    return "undefined" if pd.isna(value) else f"{value:.2%}"


def limitations(root=ROOT):
    selection = read_json(root / "artifacts/selection.json")
    name = selection["selected_candidate"]
    metrics = pd.read_csv(root / "artifacts/final/metrics.csv")
    scores = metrics.loc[metrics.candidate == name].set_index("scope")
    full, tail = scores.loc["all"], scores.loc["last_7_days"]
    hourly = pd.read_csv(
        root / f"artifacts/final/2026-09-01_{name}_hourly.csv",
        parse_dates=["hour_start"],
    ).set_index("hour_start")
    daily = pd.read_csv(
        root / f"artifacts/final/2026-09-01_{name}_daily.csv", parse_dates=["date"]
    ).set_index("date")
    daily["absolute_error"] = (daily.forecast - daily.actual).abs()
    worst = daily.nlargest(3, "absolute_error")
    examples = "\n".join(
        f"- {date.date()}: {row.actual:,.0f} recorded studies, {row.forecast:,.1f} forecast "
        f"({row.forecast - row.actual:+,.1f}); daily range {row.p10:,.1f}–{row.p90:,.1f}."
        for date, row in worst.iterrows()
    )
    positive = hourly.loc[hourly.actual > 0]
    positive_coverage = (
        (positive.actual >= positive.p10) & (positive.actual <= positive.p90)
    ).mean()
    bad_hours = (
        hourly.assign(error=(hourly.forecast - hourly.actual).abs())
        .groupby(hourly.index.hour)
        .error.mean()
        .nlargest(3)
    )
    hours_text = ", ".join(
        f"{hour:02d}:00 ({value:.1f} studies MAE)" for hour, value in bad_hours.items()
    )
    dev = pd.read_csv(root / "artifacts/development/ranking.csv").set_index("candidate")
    return f"""## Drawbacks and limitations

The submitted method was frozen as **{name}** using development windows only.
September is an evaluation result, not a reason to change that selection.

1. **Recorded arrivals are not established demand.** In June–August there are no
   rows at stored hour 00:00 or 18:00–23:00. July 8 and 9 contain only 36 and 1
   studies. We preserve zero recorded counts and do not infer true closures,
   outages, timezone corrections, or unobserved patients. Future capture changes
   could invalidate the learned clock pattern.
2. **Unusual-day treatment matters.** Mean development hourly WAPE is
   {percent(dev.loc["quantile_raw", "hourly_wape"])} for raw quantile training and
   {percent(dev.loc["quantile_filtered", "hourly_wape"])} with the specified
   near-empty-day exclusion. Validation counts are never removed. Excluding
   plausible capture failures can also remove genuinely low-demand days.
   On September, raw quantile training has {percent(metrics.loc[(metrics.candidate == "quantile_raw") & (metrics.scope == "all"), "hourly_wape"].iloc[0])}
   hourly WAPE versus {percent(full.hourly_wape)} for the frozen filtered choice;
   this holdout difference does not change the previously selected method.
3. **Long-horizon performance is measured, not assumed.** September hourly WAPE
   is {percent(full.hourly_wape)} overall and {percent(tail.hourly_wape)} in the
   last seven days; bias is {percent(full.bias)} and {percent(tail.bias)}.
   The three development windows overlap, so they are not independent trials.
4. **An 80% interval is a target, not a guarantee.** September hourly coverage is
   {percent(full.coverage)} overall and {percent(tail.coverage)} in the final week.
   Coverage on hours with positive recorded arrivals is {percent(positive_coverage)};
   predictable zero hours can make overall coverage look stronger. Daily coverage
   is {percent(full.daily_coverage)} overall and {percent(tail.daily_coverage)} in
   the final week. Only 21 complete prior days calibrate daily ranges, using
   seven-day horizons rather than a full month. No post-September recalibration
   or widening is performed to improve the reported holdout.
5. **The feature set has limited information.** Trees can fit historical level
   changes but cannot extrapolate a continuing growth trend beyond training.
   Four months cannot establish annual seasonality or October-specific holidays.
   No future hospital staffing, onboarding, hospital volumes, or outages are
   known. Hourly quantiles are marginal estimates, not joint sample paths.
   A sum of hourly medians is a daily point forecast, not a mathematical
   expectation or necessarily the daily median.
6. **Errors are concentrated.** The largest September daily absolute errors are:

{examples}

   The largest hourly MAEs by stored hour are {hours_text}.

Next steps would be a documented timezone, source-system completeness and outage
logs, more historical months, hospital onboarding schedules, and additional
rolling month-long tests of calibration. These data were not invented for this
submission. October actuals are unavailable; no October accuracy is claimed.
"""


def build_report(root=ROOT):
    import nbformat
    from nbclient import NotebookClient

    notes = limitations(root)
    markdown = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    cells = [
        markdown("""# Part 1 — recorded teleradiology arrivals

**Forecast origin: October 1, 2026; horizon: 744 source-clock hours.**
Only the supplied synthetic data is used. These are recorded arrivals, not
verified underlying demand. Timezone is unspecified. No LLM produces these forecasts.
This notebook reads saved experiment results and never trains or selects models.
"""),
        code("""import json
from io import BytesIO
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import Image, display

ROOT = Path.cwd()


def show_chart(figure):
    figure.tight_layout()
    with BytesIO() as image_data:
        figure.savefig(image_data, format="png")
        display(Image(data=image_data.getvalue()))
    plt.close(figure)


selection = json.loads((ROOT / "artifacts/selection.json").read_text())
metadata = json.loads((ROOT / "artifacts/forecast_metadata.json").read_text())
name = selection["selected_candidate"]
dev = pd.read_csv(ROOT / "artifacts/development/metrics.csv")
final = pd.read_csv(ROOT / "artifacts/final/metrics.csv")
hourly = pd.read_csv(ROOT / "forecast.csv", parse_dates=["hour_start"]).set_index(
    "hour_start"
)
daily = pd.read_csv(
    ROOT / "artifacts/daily_forecast.csv", parse_dates=["date"]
).set_index("date")
backtest = pd.read_csv(
    ROOT / f"artifacts/final/2026-09-01_{name}_hourly.csv", parse_dates=["hour_start"]
).set_index("hour_start")
backtest_daily = pd.read_csv(
    ROOT / f"artifacts/final/2026-09-01_{name}_daily.csv", parse_dates=["date"]
).set_index("date")
print("Selected before September:", name)
print("October daily point definition:", metadata["daily_point_estimate"])"""),
        markdown("""## Data and input validity

Count study rows per hour on a complete June–September grid (2,928 hours).
No-row hours become zero recorded arrivals. Near-empty days are identified
within each training fold using less than 10% of its weekday median.
Only training rows are optionally excluded; all validation actuals remain.
The repeated `day_total_studies` field is a retrospective consistency check,
never a future feature, label replacement, or missing-hour repair.

| Feature | Why available at origin |
|---|---|
| Hour, weekday, weekend | Known future calendar |
| Hour sine/cosine, weekday sine/cosine | Deterministic encodings of that calendar |
| Days since June 1 | Known elapsed calendar days |

There are no future actual lags, unknown holiday locations, or study-level
attributes in the predictors. Predict the whole horizon directly.
"""),
        code(
            """quality = json.loads((ROOT / "artifacts/development/data_quality.json").read_text())
display(pd.Series(quality, name="June–August diagnostics"))
fig, ax = plt.subplots(figsize=(10, 3))
profile = pd.Series(quality["arrivals_by_stored_hour"], dtype=float)
profile.index = profile.index.astype(int)
profile.sort_index().plot.bar(ax=ax, color="#3b82f6")
ax.set(
    title="Recorded arrival pattern before September",
    xlabel="Stored hour (timezone unspecified)",
    ylabel="June–August studies",
)
show_chart(fig)"""
        ),
        markdown("""## Methods and evaluation protocol

Seasonal naive repeats the final 168 observed hours. Histogram gradient boosting
uses Poisson loss for the point model and three separate quantile losses for
p10/p50/p90. Each learned family has raw and sparse-day-filtered variants.
Parameters are fixed at 200 iterations, learning rate 0.05, 15 leaves, minimum
20 samples per leaf, L2=1, seed=42, and no automatic early-stopping split.

Histogram gradient boosting learns nonlinear hour/weekday interactions from the
small tabular calendar dataset without feature scaling and runs on a CPU. One
installed estimator supports both count-oriented Poisson and quantile losses.
Direct quantile regression learns asymmetric, time-varying hourly ranges rather
than adding a fixed band to a point forecast.

Development origins are July 7, July 21, and August 4, with 28-day horizons.
Selection minimizes mean development hourly WAPE, then daily WAPE, then simplicity.
September 1–30 is evaluated only after committing the selection and modelling code.
All five candidates are reported; the September winner cannot change submission selection.

WAPE = sum absolute errors / sum actuals; bias = sum(forecast − actual) / sum actuals.
Total error is signed, forecast minus actual. Percentages with zero actual totals
are undefined. Last-seven-day metrics use 168 hours independently.
"""),
        code("""columns = [
    "origin",
    "candidate",
    "scope",
    "hourly_wape",
    "daily_wape",
    "bias",
    "actual_total",
    "forecast_total",
    "total_error",
    "absolute_total_error",
    "absolute_total_error_fraction",
    "coverage",
    "mean_interval_width",
    "daily_coverage",
    "quantile_crossing_rate",
    "pinball_10",
    "pinball_50",
    "pinball_90",
]
print("ALL DEVELOPMENT RESULTS — fractions, not percentages")
display(dev[columns])
print("Selection ranking (mean development full-window errors)")
display(pd.read_csv(ROOT / "artifacts/development/ranking.csv"))
print("FROZEN SEPTEMBER RESULTS — fractions, not percentages")
display(final[columns])"""),
        markdown("""## Uncertainty and September errors

Hourly quantile bounds are learned directly. Nonnegative sorting repairs crossing
quantiles before scoring/export; crossing rates above refer to the original values.
The point model and baseline use hour-conditioned residual percentiles from three
prior seven-day forecasts, each fit only on earlier history.

Daily ranges use percentiles of **complete-day residuals** from those 21 prior days,
added to daily point totals. They preserve within-day error aggregation and are
not sums of hourly quantiles. Bands are clipped/bracketed to contain nonnegative
point forecasts. Coverage is measured rather than asserted to be 80%.
"""),
        code("""fig, axes = plt.subplots(2, 1, figsize=(12, 7))
backtest_daily[["actual", "forecast"]].plot(ax=axes[0])
axes[0].fill_between(
    backtest_daily.index,
    backtest_daily.p10,
    backtest_daily.p90,
    alpha=0.2,
    label="Daily residual-based p10–p90",
)
axes[0].set(title=f"September daily evaluation: {name}", ylabel="Recorded studies")
axes[0].legend()
week = backtest.loc["2026-09-24":"2026-09-30"]
axes[1].plot(week.index, week.actual, label="Actual", color="black", linewidth=1)
axes[1].plot(
    week.index, week.forecast, label="Point forecast", color="#2563eb", linewidth=1
)
axes[1].fill_between(week.index, week.p10, week.p90, alpha=0.2, label="Hourly p10–p90")
axes[1].set(
    title="September final seven days",
    ylabel="Recorded studies per hour",
    xlabel="Source clock; timezone unspecified",
)
axes[1].legend()
show_chart(fig)"""),
        markdown("""## October operations forecast

All available history through September 30 is used to refit the frozen selected
method. Display daily point totals and ranges for staffing discussions; these
are recorded-arrival forecasts. If p50 is selected, the daily point is a sum of
hourly medians, not an expected value or necessarily a daily median.
"""),
        code("""print(f"October point total: {hourly.forecast.sum():,.1f}")
display(daily.round(1))
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(daily.index, daily.forecast, marker="o", label="Daily point total")
ax.fill_between(
    daily.index, daily.p10, daily.p90, alpha=0.2, label="Daily residual-based p10–p90"
)
ax.set(
    title="October 2026: daily workload and uncertainty",
    xlabel="Source calendar date",
    ylabel="Recorded studies per day",
)
ax.legend()
show_chart(fig)
print("Ten busiest forecast dates/hours (source clock)")
display(hourly.nlargest(10, "forecast").round(1))
fig, ax = plt.subplots(figsize=(10, 3))
hourly.forecast.groupby(hourly.index.hour).mean().plot.bar(ax=ax, color="#2563eb")
ax.set(
    title="October: typical workload by stored hour",
    xlabel="Stored hour",
    ylabel="Mean point forecast across October dates",
)
show_chart(fig)"""),
        markdown(notes),
        markdown("""## Reproducibility and handoff

`forecast.csv` is the 744-row submission. API endpoints serve this same saved
forecast and the separately calibrated daily artifact. The permanent pytest
suite covers feature values, input validation, origin-safe training/calibration,
forecast contracts, API responses, date filters, and error handling. Tests use
synthetic temporary artifacts and do not overwrite submitted forecasts.

Run the service with `docker compose up --build` and the tests with
`docker compose run --build --rm tests`. The test service is profile-isolated
from normal API startup; notebook tooling is excluded from the service image.
See README for local testing and pipeline stage commands. Charts are embedded
in this notebook. The original 45-test run is historical; current verification
is recorded in README.
The selection, checksums, and pre-September commit are recorded under `artifacts/`.
AI coding assistance and concrete corrections are disclosed in `AI_USE_LOG.md`.
Part 1 was committed before Part 2. The Part 2 section below reads saved results
for the LLM hybrid comparison and explanation demonstrations without LLM calls.
It reports pending stages if those artifacts are incomplete.
"""),
    ]
    from llm_forecasting.reporting import notebook_cells

    cells.extend(notebook_cells(root))
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "name": "main_env",
                "display_name": "Python (main_env)",
                "language": "python",
            },
            "language_info": {"name": "python", "version": "3.13.18"},
        },
    )
    executed = NotebookClient(notebook, timeout=120, kernel_name="main_env").execute(
        cwd=str(root)
    )
    nbformat.write(executed, root / "report.ipynb")
    print("Saved executed report.ipynb with embedded charts", flush=True)
