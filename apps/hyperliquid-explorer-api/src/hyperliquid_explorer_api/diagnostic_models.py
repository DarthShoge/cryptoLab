"""Public, finite, read-only saved-run diagnostic contract."""

from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field, JsonValue


class DiagnosticMetric(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    value: float | None = None
    reason: str | None = None
    unit: str = "ratio"


class DiagnosticPoint(BaseModel):
    time: datetime
    equity: float
    growth: float
    drawdown: float
    gross_exposure: float | None = None
    net_exposure: float | None = None


class DiagnosticPeriod(BaseModel):
    start: datetime
    end: datetime
    return_value: float
    partial: bool


class DiagnosticSeries(BaseModel):
    points: list[DiagnosticPoint] = Field(default_factory=list)
    weeks: list[DiagnosticPeriod] = Field(default_factory=list)
    months: list[DiagnosticPeriod] = Field(default_factory=list)
    samples: int = 0
    max_drawdown: float | None = None
    reason: str | None = None


class AccountingCheck(BaseModel):
    name: str
    difference_usd: float | None = None
    passed: bool | None = None
    reason: str | None = None


class DiagnosticCohort(BaseModel):
    decision_time: datetime
    coin: str | None = None
    candidate_count: int | None = None
    eligible_count: int | None = None
    selected_count: int | None = None
    membership_turnover: float | None = None


class DiagnosticContext(BaseModel):
    start: datetime
    end: datetime
    return_value: float
    fills: int | None = None
    fees_usd: float | None = None
    funding_usd: float | None = None
    assets: list[str] = Field(default_factory=list)
    max_gross_usd: float | None = None
    mean_net_usd: float | None = None
    exposure_reason: str | None = None


class Diagnostics(BaseModel):
    schema_version: str = "saved_diagnostics_v1"
    experiment_id: str
    run_id: str
    name: str
    dataset_id: str
    config_hash: str
    config: dict[str, JsonValue]
    provenance: dict[str, JsonValue]
    research_eligible: bool | None
    synthetic: bool = False
    reconciliation: dict[str, JsonValue]
    warnings: list[str]
    series: dict[str, DiagnosticSeries]
    stored_metrics: dict[str, DiagnosticMetric]
    benchmark_metrics: dict[str, DiagnosticMetric]
    statistics: dict[str, DiagnosticMetric]
    accounting: dict[str, DiagnosticMetric]
    checks: list[AccountingCheck]
    positions: dict[str, JsonValue]
    cohorts: list[DiagnosticCohort]
    cohort_reason: str | None = None
    worst_weeks: list[DiagnosticContext]
    methods: list[str]
