# Teleradiology forecasting — classical and LLM hybrid flows

Part 1 forecasts hourly recorded arrivals using classical ML. Part 2 uses local
Qwen to predict daily volume, then Python allocates it to hours and calibrates
empirical ranges. AI assistance is disclosed in [AI_USE_LOG.md](AI_USE_LOG.md).

## Run the service

On macOS, start Docker only for an API session:

```bash
python docker_tasks.py up
```

The helper starts Docker Desktop if it is closed, waits up to 120 seconds for
the engine, builds the API, and displays its logs. Press Ctrl+C to stop the API.
It then closes Docker Desktop if it started Desktop and no other containers
are running. A previously running Desktop stays open. If Desktop is paused or
its initial status is uncertain, resume/check it manually before retrying.
Training, notebooks, local tests, and hosted LLM calls do not need Docker.
Automatic startup at login can remain disabled.

The helper uses only Python's standard library. From another directory, pass
its full path; it always uses this project's Compose file and directory.

The direct assignment command also works with Docker Desktop already running:

```bash
docker compose up --build
```

API documentation: <http://localhost:8000/docs>.

- `GET /forecast/hourly`: all 744 hourly point forecasts and p10/p90 ranges.
- `GET /forecast/daily`: 31 daily point totals and separately calibrated ranges.
- `GET /health`: readiness and artifact row counts.
- `GET /explain`: a checked LLM explanation of Part 1 numbers, or a fixed-template
  fallback if the provider or response is invalid. It accepts the same dates.

Both forecast endpoints accept inclusive `start_date` and `end_date`, for example
`/forecast/hourly?start_date=2026-10-25&end_date=2026-10-31`. Malformed/reversed dates
return 422; ranges extending outside October return 404. Response metadata names
the model, estimate definition, and unspecified timezone. Missing, modified, or
invalid artifacts fail startup rather than silently serving inconsistent data.
The service reads saved artifacts and does not train during requests.

## Run the tests

To start Docker only for the container test run:

```bash
python docker_tasks.py test
```

This has the same startup/shutdown rules as the API helper and returns the
container test exit code. Ctrl+C also cleans up the helper's test container.
Cleanup problems are reported without replacing the task's exit code. If another
container prevents Desktop shutdown, stop Docker manually when finished:
`docker desktop stop`. Images, volumes, and submitted forecasts are preserved.

The single Docker test command builds an isolated test target and exits with
pytest's result:

```bash
docker compose run --build --rm tests
```

The test service uses a Compose profile, so `docker compose up --build` starts
only the API. The service image excludes pytest and notebook tooling; the test
image adds pytest, Git for isolated protocol-freeze tests, nbformat for report-cell
checks, and the permanent `tests/` folder. Tests never call Ollama or hosted APIs.

For local testing after installing the development requirements:

```bash
python -B -m pytest -q -p no:cacheprovider tests
```

Tests cover all eight features, input validation, hourly aggregation, sparse-day
handling, post-origin leakage, saved forecast contracts, and both API endpoints.
They use synthetic fixtures and pytest-managed temporary files without external
services or LLM calls. Small model fits use eight iterations within isolated
tests; production remains at 200. Submitted forecast files are read-only.

## Project structure

- `forecasting/`: calendar features, models, metrics, pipeline, validation, API,
  and notebook generator.
- `llm_forecasting/`: separate LLM CLI, provider adapters, prompts and schemas,
  experiment stages, explanations, and saved-result report sections.
- `tests/`: shared synthetic fixtures, Part 1 tests, and Part 2 forecasting, provider and explanation tests.
- `docker_tasks.py`: macOS Docker startup and cleanup for API sessions and tests.
- `data/`: unchanged source CSV, supplied dictionary, and checksums.
- `artifacts/development/`: all three development backtests, metrics, and ranking.
- `artifacts/final/`: September results, manifest, and training audits.
- `artifacts/selection.json`: frozen model selection, specification, and hashes.
- `artifacts/daily_forecast.csv` and `artifacts/forecast_metadata.json`: service data.
- [forecast.csv](forecast.csv): exactly 744 October rows in the required format.
- [report.ipynb](report.ipynb): executed report, limitations, and embedded charts.
- `artifacts/llm/`: contains the retained daily-volume development results,
  the frozen protocol, request evidence, results and demonstrations.
- `llm_forecast.csv`: created only after the manual October LLM stage succeeds.

The notebook is the report and presentation. It embeds charts directly without
creating a separate Markdown report or image folder. Apart from the Docker task
helper, optional verification scripts, environment snapshots, editor configuration,
and caches are not part of this layout.

## Local environment and pipeline

Use Python **3.13.18** and the existing `main_env` environment:

```bash
conda activate main_env
python -m pip install --only-binary=:all: -r requirements-dev.txt
python -m pip check
```

Runtime/service dependencies are in `requirements.txt`; `requirements-dev.txt`
adds pytest, Matplotlib, JupyterLab, and ipykernel. The service image resolves its own Linux
dependencies and excludes notebook tooling. Part 1 and the local Ollama defaults
require no API key; optional provider variables are documented below. Docker sets
`PYTHONDONTWRITEBYTECODE`, `PYTHONUNBUFFERED`,
`OMP_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` to 1.

The completed checkout can regenerate October artifacts and the notebook:

```bash
python -m forecasting forecast
python -m forecasting report
```

The notebook generator uses kernel `main_env`. If needed, register it with:

```bash
python -m ipykernel install --sys-prefix --name main_env --display-name "Python (main_env)"
```

Development originally ran with `python -m forecasting develop`; code, selection,
and development results were committed at `a929ab4` before the first
`python -m forecasting evaluate-final`. The final manifest records that commit.
Reproduce original development from a separate checkout of the pre-evaluation
commit. In this completed checkout, `develop` refuses to overwrite the selection
and `evaluate-final` preserves the existing September result. Do not remove those
guards to tune on September.

The freeze validates the selected settings, data, dependencies, development
results, and the original pre-evaluation Git snapshot. Saved source hashes refer
to that archived snapshot, not later working-tree formatting or layout edits.
Later source changes are tracked through Git. No separate maintenance record is
required. Original selection and evaluation files are unchanged.

## Method and assumptions

The eight calendar features are hour, weekday, weekend, sine/cosine encodings of
hour and weekday, and days since June 1. All are available at forecast time.
The whole horizon is predicted directly without actual counts inside the window.
`day_total_studies` is only a retrospective consistency check, not a feature or
replacement target. Timestamps stay on the unspecified source clock, without
invented timezone/DST corrections or regional holidays.

The baseline repeats the final 168 observed hours. Histogram gradient boosting
uses Poisson loss for point predictions and three quantile objectives (0.1/0.5/0.9).
Shared parameters: 200 iterations, learning rate 0.05, 15 leaves, minimum 20 samples
per leaf, L2=1, seed=42, and `early_stopping=False`. No parameter search or random
validation split is used.

Histogram gradient boosting keeps this small calendar-feature problem in one
CPU-based library and learns nonlinear hour/weekday interactions without manual
feature scaling. The same estimator supports count-oriented Poisson loss and
direct quantile loss. Quantile regression allows asymmetric, time-varying hourly
ranges rather than imposing a fixed band around a point forecast.

Each learned family is trained raw and with days below 10% of their training
weekday median excluded, producing five candidates including baseline. Thresholds
use only each origin's history. Validation actuals are never removed. Missing rows
represent zero recorded arrivals, not established zero underlying demand.

Origins July 7, July 21, and August 4 forecast 28 days each. September 1 forecasts
all 30 September days from earlier history. Selection minimizes mean development
hourly WAPE, then daily WAPE, then simplicity. September cannot change selection.
WAPE and bias are fractions in result CSVs; total error is forecast minus actual.
Full-window and last-seven-day metrics are reported, including interval coverage.
Percentage metrics with zero actual totals are undefined.

Quantile hourly ranges are learned directly, clipped nonnegative, and sorted to
repair crossings before scoring. Baseline/point hourly ranges use residual
percentiles grouped by stored hour. Daily ranges use percentiles of whole-day
residual sums from three prior weekly forecasts, each trained only on earlier
history. They are not sums of hourly quantiles. A sum of hourly medians is not
necessarily the daily median or expectation. The notebook documents calibration,
missing-data, trend-extrapolation, and long-horizon limitations.

## Selected model and results

The frozen selection is **quantile_filtered**, with mean development hourly WAPE
**19.95%**, versus **24.11%** for seasonal naive.

| September metric | Selected quantile model | Seasonal naive |
|---|---:|---:|
| Hourly WAPE | 15.85% | 19.16% |
| Daily-total WAPE | 13.12% | 15.44% |
| Hourly p10–p90 coverage | 91.25% | 86.25% |

Selected-model final-week hourly WAPE is **15.32%** and hourly coverage **92.86%**.
Full-month daily coverage is **76.67%**; ranges do not guarantee 80% coverage.
Raw quantile training scores slightly better on September (15.71% hourly WAPE),
but the previously frozen selection is retained. October's point total is
**43,974.8** recorded studies. October actuals are unavailable.

The permanent pytest suite and Docker test target are included in this checkout.
The earlier 45-test results belong to the original Part 1 snapshot; verification
of the current suite is recorded below. Part 2 remains a separate phase.

## Preserved Part 1 verification

- Local Part 1 test subset: **58 passed**.
- Linux Docker Part 1 test subset: **58 passed**.
- Notebook: five executed code cells, four embedded charts, no execution errors.
- Original selection, September predictions/metrics, and October forecast files
  remain unchanged. Hourly coverage remains the measured 91.25%, not an asserted
  80% guarantee.

The installed FastAPI/Starlette release emits an HTTPX TestClient deprecation
warning; tests pass without suppressing it. The sandboxed macOS run also reports
a physical-core detection warning and uses logical-core fallback. Neither warning
changes the submitted forecasts.


## Local Qwen configuration

Use the existing Python 3.13.18 `main_env`, Ollama 0.40.15 and installed
`qwen3.5:4b-q4_K_M`. No new installation is required. Start the native server
in a separate terminal, with Docker closed:

```bash
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NO_CLOUD=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve
```

The process environment uses fixed names: `LLM_PROVIDER`, `LLM_BASE_URL`,
`LLM_MODEL`, `LLM_API_KEY`. Defaults select local Ollama, its loopback URL and
the installed model, with no key. `.env.example` documents them; Python does not
load `.env` automatically. To load a trusted local file, use `set -a`,
`source .env`, then `set +a`. Never commit `.env` or API keys.
Compose defaults to `http://host.docker.internal:11434` for host Ollama;
leave `LLM_BASE_URL` unset when switching between native and Docker defaults.
OpenAI/Gemini HTTPX adapters are mock-tested; experiment stages require local
Qwen and do not make paid calls.

Generation remains temperature 0.2, a 16,384-token context, an 8,192-token output
ceiling, thinking disabled, no fixed seed, sequential requests and immediate
unloading. Input budget is context minus the requested output allowance; there
is no extra output-token cap after generation. Truncated responses are rejected.
Weights and Ollama remain outside Docker and the Python environment.

## Part 2 — daily-volume LLM hybrid

The main `llm_forecasting/` flow now predicts two daily point totals per call,
then allocates each total to 24 hours in Python. Its only output directory is
`artifacts/llm/`, and its October export is `llm_forecast.csv`.

### What Qwen predicts and what Python calculates

| Component | Responsibility |
|---|---|
| Qwen | Returns one daily point total per requested date, normally two dates per call, using only pre-origin summaries and the known future calendar. |
| Python hourly allocation | Calculates 24 historical volume proportions for each weekday and multiplies each predicted daily total by that weekday's proportions. |
| Python hourly uncertainty | Uses errors from three prior seven-day hybrid forecasts to calculate empirical residual p10/p90 offsets grouped by clock hour. |
| Python daily uncertainty | Calculates residual p10/p90 offsets from the 21 complete daily errors, separately from the hourly bounds. |

For weekday `w` and hour `h`, the allocation is:

```text
weight[w,h] = historical volume at weekday w, hour h / historical volume at weekday w
hourly_forecast[date,h] = Qwen_daily_total[date] * weight[weekday(date),h]
```

Each weekday's 24 weights sum to one, so the hourly points sum to Qwen's daily
total. Historical proportions and calibration forecasts use only information
available before their respective origins. The fallback rules for zero-volume
history are described below.

Hourly and daily bounds add their respective residual percentiles to the point,
keep lower bounds nonnegative and expand bounds when needed to contain the
point. Hourly calibration captures both daily-volume and hourly allocation
errors; daily calibration captures errors in the total daily volume.
These are empirical uncertainty estimates, not Qwen-generated quantiles;
`forecast` in the CSV is an allocated point, not a directly predicted hourly p50.
Daily intervals are calculated independently, never by summing hourly p10/p90.

Run `report.ipynb` with the repository root as the notebook working directory.
Both report sections resolve their artifact paths from that directory, so the
notebook can run from another clone without editing a personal filesystem path.

**Assignment limitation:** L1 asks an LLM to generate hourly p10/p50/p90 directly.
This chosen hybrid instead generates daily totals and uses Python for hourly
allocation and empirical ranges, so it does not strictly satisfy that direct
LLM-output requirement. It is labelled hybrid throughout. The superseded direct
trial and its working-tree results were deleted at the user's request; the
report does not depend on or present those removed results.

Hybrid development was retained unchanged during promotion: mean hourly WAPE
**19.56%**, versus Part 1's **19.95%**, across the same three development windows.
September outcomes had already been inspected when this hybrid was designed;
its completed September evaluation is explicitly **exploratory**, not a fresh
untouched holdout. Across three runs, mean September hourly WAPE is **15.03%**
and daily-total WAPE is **12.52%**. Mean hourly interval coverage is **76.39%**;
daily coverage is only **56.67%**, a limitation rather than an 80% guarantee.
October forecasts are generated, but October actuals are unavailable.

### Execution sequence (completed)

The promoted code, protocol and development evidence were committed before the
hybrid September evaluation. All three September runs, October forecasting,
the live/fallback explanation demonstrations and the combined report are now
complete. The commands below document the workflow; development is frozen in
this checkout after September evaluation.

```bash
conda activate main_env
# Development step, before September; now frozen in the completed checkout.
python -m llm_forecasting develop

git add -A llm_forecasting tests artifacts/llm Dockerfile .gitignore README.md AI_USE_LOG.md
git commit -m "Promote daily-volume LLM hybrid and retain development evidence"

python -m llm_forecasting evaluate-final
python -m llm_forecasting forecast
python -m llm_forecasting explain-demo
python -m llm_forecasting report
```

The experiment stages and notebook regeneration were executed manually.
`evaluate-final` runs three independent September forecasts and freezes
further development at its first attempt. `forecast` produces October without
selecting a best September run. Completed stages are verified and preserved.
`explain-demo` saves a genuine explanation of verified Part 1 facts and a labelled
simulated-outage fallback. `report` reads saved results only, compares all Part 1
methods with the hybrid, displays measured forecast errors and explanations,
and appends/replaces its tagged notebook cells while preserving existing charts.
The combined notebook includes these sections and the original Part 1 charts.
Part 1's notebook generator can also include the completed Part 2 sections.

Explanation replies may use named placeholders or literal figures in the supplied
template. Literal totals, dates and bounds must match their specific template
fields exactly; merely appearing elsewhere in the verified facts is insufficient.
Other wording must use placeholders. Unknown or extra numeric claims, changed
figures and reversed ranges still trigger retries and the template fallback.
The demo reports the last failure if every attempt is rejected; rerunning it
after an explanation-only fix does not regenerate any forecasting results.

### Inputs, response and allocation

Each main batch uses the same history ending before its window origin:
latest 28 daily totals, weekday daily mean/median/p10/p90/sample counts,
empty/sparse-day diagnostics and requested dates/weekdays. All zeros and unusual
days remain recorded observations; sparse days are not assumed outages.
No study attributes, `day_total_studies`, earlier predictions or observations
inside the forecast horizon enter later prompts. Timezone stays unspecified.
Odd-length horizons end with a one-day request. Strict JSON is:

```json
{"days":{"2026-07-07":1200.0,"2026-07-08":1350.5}}
```

Date keys must match the request; totals must be finite nonnegative numbers,
excluding booleans. Valid dates are kept immediately. Retries request only
missing/invalid dates with the same historical and calendar context. Each batch
invocation allows the initial request plus three retries, delayed 1/2/4 seconds.
Invalid JSON, duplicates, extra fields, wrong dates and truncation yield no new
accepted values. Exhausted isolated batches do not prevent later batches/windows
being tried. Reruns validate raw evidence and request only missing dates; an
incomplete stage publishes no forecast CSVs, metrics or frozen protocol.
Configuration errors, incompatible evidence and exhausted budgets stop clearly.

Python calculates each weekday/hour's recorded volume divided by total volume
for that weekday, using only history before the relevant origin. A weekday with
zero volume uses the overall hourly profile; all-zero history uses uniform 1/24
weights. Multiplying normalized proportions by each daily total preserves it
within `1e-6`. Fractional hourly points are not directly predicted hourly medians.
Daily/hourly coherence is enforced by Python, not a measure of LLM arithmetic.

### Calibration, evidence and budget

Three seven-day calibration forecasts start 21, 14 and 7 days before each outer
origin. Each builds summaries/proportions from its own earlier history. Hourly
residual p10/p90 are grouped by clock hour (21 samples each); daily bands use
whole-day residual percentiles separately. Bounds are nonnegative and contain
the point. Daily intervals never sum hourly bounds. These empirical ranges have
no guaranteed 80% coverage, and seven-day calibration may miss late-month risk.
Fixed historical shares, sparse days and limited history are additional limits.

| Stage | Main batches | Calibration batches | Total |
|---|---:|---:|---:|
| Three development windows | 42 | 36 | 78 |
| Three September runs | 45 | 36 | 81 |
| October | 16 | 12 | 28 |
| Total | 103 | 84 | 187 |

The global maximum is **748 attempts**, including retries and resumes. Retained
development calls count toward this budget. Each September run has independent
calibration draws but identical corresponding prompts/settings. Explanation
calls are separate. A stable ignored `.pipeline.lock` excludes concurrent stages.

Protocol version 5 freezes the main hybrid sources, data, settings, model identity,
allocation/calibration, budgets, development files and request-evidence prefix.
Its `promotion` field embeds the original daily-volume protocol and records the
move. The original 348,962-byte development journal and all 14 development files
were preserved byte for byte. Only that verified legacy development prefix is
accepted; later requests use version 5 and the promoted experiment hash. Other
legacy evidence is rejected. All cached requests were replayed without inference
and the retained hourly/daily forecasts reproduced within `1e-9` before removal
of the duplicate package. A new commit is required before September evaluation.

Saved results include hourly/daily forecasts, seasonal proportions, calibration
forecasts/residuals, metrics, reliability and repeatability. Every backtest reports
hourly/daily WAPE, bias, totals/errors, coverage and width for the full window and
last seven days. Pinball scores describe empirical bounds rather than direct LLM
quantiles. Reliability separates main, calibration and combined calls, partial
recovery, retries, tokens, loading/unloading latency and complete-forecast timing.
Resumed wall times include manual pauses; active time and last-invocation time
are separate. API charges are US$0, excluding electricity and hardware.
Three September runs measure pairwise hourly differences and monthly-total
variation. The report's oracle daily-volume diagnostic uses actual daily totals
only to explain allocation weakness; it is not a valid forecast or additive
error decomposition.

`llm_forecast.csv` is published atomically only after validating exactly 744
consecutive October rows with `hour_start,forecast,p10,p90`, finite nonnegative
values and ordered bounds. Existing submissions without a verified manifest are
preserved and rejected rather than overwritten. `/explain` continues to use only
Part 1 artifacts, validates its numbers and falls back when Ollama is unavailable.

## Current implementation verification

- Local and Linux Docker suites: **165 passed**, including verified literal
  explanation figures, promotion-prefix, partial-date recovery, provider, API
  and full 187-batch workflow tests.
- Both submitted forecast CSVs validate with exactly **744 October hours**.
- All three hybrid September runs and the October forecast have complete,
  verified manifests; comparison and allocation-diagnostic CSVs are saved.
- The live explanation is validated with source `llm`; the simulated-outage
  demonstration returns source `template` using the same verified Part 1 facts.
- The combined `report.ipynb` has **eight executed code cells, five embedded
  charts and no execution errors**. Original Part 1 cells and charts are retained.
- Part 1 models, input data, selection and forecast results are preserved.
- The original hybrid development journal prefix and 14 development files were
  preserved; later manual runs appended their request evidence. Old direct
  results and duplicate experimental folders were removed as requested.
- Implementation checks used mocks; real experiment stages and report execution
  were performed manually afterward. Docker was stopped after container tests.
