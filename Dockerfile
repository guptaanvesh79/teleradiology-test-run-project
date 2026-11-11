FROM python:3.13-slim AS service
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --only-binary=:all: -r requirements.txt
COPY forecasting ./forecasting
COPY llm_forecasting ./llm_forecasting
COPY forecast.csv .
COPY artifacts/daily_forecast.csv artifacts/forecast_metadata.json ./artifacts/
USER 10001:10001
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "forecasting.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM service AS tests
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir --only-binary=:all: pytest==9.2.0 nbformat==5.11.1
COPY tests ./tests
USER 10001:10001
CMD ["python", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"]
