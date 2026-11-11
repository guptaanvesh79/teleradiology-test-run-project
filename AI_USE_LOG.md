# AI-use log

ChatGPT/Codex helped plan calendar features, data preparation, forecasting, API,
Docker, tests and reporting. Classical algorithms produce Part 1's numbers.

An AI-written test assumed pytest's `tmp_path` was empty despite existing fixture
files. Pytest caught the mistaken startup-failure expectation. The fixture was
corrected to use a separate nonexistent directory, preserving service behaviour.

Checks include features, leakage and API consistency. Part 1 selection was
committed before September evaluation.

For Part 2, AI helped implement schemas, backtests, retries and explanations.
Installation exposed Homebrew's missing MLX grammar library, resolved with
Q4_K_M. A timestamp failure led to Python-generated dates.
Pytest caught a numeric checker misreading date punctuation. Live development
exposed a stale 4,096-token input limit rejecting a 5,518-token prompt; the guard
now uses the configured budget. Positional replies also crossed quantiles.
The hourly-label trial underwent development and September runs. A live explanation exposed
rejection of correct literal figures; field-specific template matching fixed it
while retaining rejection of incorrect figures.

AI helped build daily-volume/hourly-allocation forecasting and rolling calibration.
Pytest corrected a retry-count expectation that ignored one-day calibration tails.
After development, I requested promotion into the main LLM package and deletion
of the direct trial and temporary folders. Its daily development CSVs and raw
journal were retained unchanged with migration provenance. This is an exploratory
hybrid, differing from L1's direct hourly-quantile requirement. I manually completed
three September runs, October forecasting, live/fallback explanation demonstrations
and the combined executed notebook. Local and Docker suites pass 165 tests.
Part 1 is preserved; no paid calls were made.
