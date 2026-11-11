"""Read-only service for validated, precomputed October artifacts."""

import json
from contextlib import asynccontextmanager
from datetime import date as Date
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field, model_validator

from .artifacts import validate_daily, validate_hourly
from .config import CANDIDATES
from .data import sha256_file

ROOT = Path(__file__).resolve().parents[1]
FIRST_DAY, LAST_DAY = Date(2026, 10, 1), Date(2026, 10, 31)


class Bounds(BaseModel):
    forecast: float = Field(ge=0, allow_inf_nan=False)
    p10: float = Field(ge=0, allow_inf_nan=False)
    p90: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def check_order(self):
        if not self.p10 <= self.forecast <= self.p90:
            raise ValueError("Require p10 <= forecast <= p90")
        return self


class HourlyRow(Bounds):
    hour_start: datetime


class DailyRow(Bounds):
    date: Date


class Metadata(BaseModel):
    model: str
    forecast_origin: Date
    timezone: Literal["unspecified"]
    point_estimate: str
    daily_point_estimate: str
    hourly_interval: str
    daily_interval: str


class HourlyResponse(BaseModel):
    metadata: Metadata
    data: list[HourlyRow]


class DailyResponse(BaseModel):
    metadata: Metadata
    data: list[DailyRow]


class ExplanationResponse(BaseModel):
    metadata: Metadata
    explanation: str
    source: Literal["llm", "template"]
    model: str | None


def load_artifacts(root):
    try:
        hourly_path = root / "forecast.csv"
        daily_path = root / "artifacts/daily_forecast.csv"
        metadata = json.loads((root / "artifacts/forecast_metadata.json").read_text())
        hourly = validate_hourly(pd.read_csv(hourly_path))
        daily = validate_daily(pd.read_csv(daily_path), hourly)
        if metadata["forecast_csv_sha256"] != sha256_file(hourly_path):
            raise ValueError("Hourly artifact checksum mismatch")
        if metadata["daily_csv_sha256"] != sha256_file(daily_path):
            raise ValueError("Daily artifact checksum mismatch")
        if metadata["model"] not in {c.name for c in CANDIDATES}:
            raise ValueError("Unknown forecast model")
        validated_metadata = Metadata.model_validate(metadata)
        if validated_metadata.forecast_origin != FIRST_DAY:
            raise ValueError("Incorrect forecast origin")
        return hourly, daily, validated_metadata
    except (OSError, ValueError, KeyError) as error:
        raise RuntimeError(
            "Forecast artifacts are missing or invalid. Generate and validate them with "
            "'python -m forecasting forecast' before starting the service. "
            + str(error)
        ) from error


def date_range(start_date, end_date):
    start, end = start_date or FIRST_DAY, end_date or LAST_DAY
    if start > end:
        raise HTTPException(
            status_code=422, detail="start_date must be on or before end_date"
        )
    if start < FIRST_DAY or end > LAST_DAY:
        raise HTTPException(
            status_code=404, detail="Forecast available only for October 1–31, 2026"
        )
    return pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)


def create_app(artifact_root=ROOT, *, llm_client=None):
    artifact_root = Path(artifact_root)

    @asynccontextmanager
    async def lifespan(app):
        app.state.hourly, app.state.daily, app.state.metadata = load_artifacts(
            artifact_root
        )
        yield

    app = FastAPI(
        title="Recorded study arrivals — October 2026",
        description="Classical forecasts on the source clock; timezone unspecified. Daily ranges use whole-day calibration residuals.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request):
        return {
            "status": "ready",
            "hours": len(request.app.state.hourly),
            "days": len(request.app.state.daily),
        }

    @app.get("/forecast/hourly", response_model=HourlyResponse)
    def hourly(
        request: Request,
        start_date: Annotated[Date | None, Query()] = None,
        end_date: Annotated[Date | None, Query()] = None,
    ):
        start, end = date_range(start_date, end_date)
        frame = request.app.state.hourly
        selected = frame.loc[(frame.index >= start) & (frame.index < end)]
        return {
            "metadata": request.app.state.metadata,
            "data": selected.reset_index().to_dict("records"),
        }

    @app.get("/forecast/daily", response_model=DailyResponse)
    def daily(
        request: Request,
        start_date: Annotated[Date | None, Query()] = None,
        end_date: Annotated[Date | None, Query()] = None,
    ):
        start, end = date_range(start_date, end_date)
        frame = request.app.state.daily
        selected = frame.loc[(frame.index >= start) & (frame.index < end)]
        return {
            "metadata": request.app.state.metadata,
            "data": selected.reset_index().to_dict("records"),
        }

    @app.get("/explain", response_model=ExplanationResponse)
    def explain(
        request: Request,
        start_date: Annotated[Date | None, Query()] = None,
        end_date: Annotated[Date | None, Query()] = None,
    ):
        from llm_forecasting.explain import explain_forecast

        start, end = date_range(start_date, end_date)
        hourly = request.app.state.hourly
        daily = request.app.state.daily
        result = explain_forecast(
            hourly.loc[(hourly.index >= start) & (hourly.index < end)],
            daily.loc[(daily.index >= start) & (daily.index < end)],
            llm_client,
        )
        return {
            "metadata": request.app.state.metadata,
            **{key: result[key] for key in ("explanation", "source", "model")},
        }

    return app


app = create_app()
