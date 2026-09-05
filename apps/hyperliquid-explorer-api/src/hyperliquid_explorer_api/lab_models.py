"""Typed lab HTTP contracts generated from the domain configuration."""

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from pydantic.dataclasses import dataclass
from arblab.hyperliquid_copy.lab_config import LabConfig as DomainConfig
from .models import Analytics, EquityRow, Page, Model

LabConfig = dataclass(DomainConfig, frozen=True, config=ConfigDict(extra="forbid"))


class Submission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(default="Untitled hypothesis", min_length=1, max_length=120)
    dataset_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    config: LabConfig
    parent_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")

    @model_validator(mode="before")
    @classmethod
    def strict_domain(cls, value):
        if isinstance(value, dict) and isinstance(value.get("config"), dict):
            DomainConfig.from_dict(value["config"])
        return value


class Preview(Submission):
    decision_date: str
    scope: str | None = None


class PublicValidationIssue(Model):
    code: str
    field: str
    message: str
    required: str | None = None
    available: str | None = None


class Preflight(Model):
    ready: bool
    issues: list[PublicValidationIssue]
    config_hash: str
    required_start: str
    required_end: str
    estimates: dict[str, int]


class Annotation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    notes: str = Field(max_length=4000)


class Experiment(Model):
    id: str
    name: str
    notes: str
    dataset_id: str
    config: LabConfig
    config_hash: str
    provenance: dict[str, JsonValue]
    created_at: datetime
    updated_at: datetime
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    kind: Literal["backtest", "cohort_preview"]
    error: str | None = None
    run_id: str | None = None
    needs_resume: bool = False
    artifact_hashes: dict[str, str]
    parent_id: str | None = None
    preview_date: str | None = None
    preview_scope: str | None = None


class Dataset(Model):
    id: str
    name: str
    available: bool
    synthetic: bool
    coverage_note: str
    coverage_start: str | None = None
    coverage_end: str | None = None
    coins: list[str] = Field(default_factory=list)
    rows: int = 0
    dataset_hash: str | None = None
    fee_semantics: str | None = None
    default_config: LabConfig | None = None


class Bootstrap(Model):
    token: str
    enabled: bool
    defaults: LabConfig
    metrics: list[str]
    limits: dict[str, int]
    restrictions: list[str]


class UniverseRow(Model):
    decision_time: datetime | None = None
    time: datetime | None = None
    coin: str | None = None
    user: str | None = None
    score: float | None = None
    rank: int | None = None
    selected: bool | None = None
    eligible: bool | None = None
    reasons: list[str] = Field(default_factory=list)
    metrics: dict[str, float | None] | None = None
    percentiles: dict[str, float | None] | None = None
    weight: float | None = None
    members: list[str] = Field(default_factory=list)
    entries: list[str] = Field(default_factory=list)
    exits: list[str] = Field(default_factory=list)
    candidate_count: int | None = None
    eligible_count: int | None = None
    selected_count: int | None = None
    requested_count: int | None = None
    retention: float | None = None
    membership_turnover: float | None = None
    position_qty: float | None = None
    signal_input: float | None = None
    known: bool | None = None
    nominal_weight: float | None = None
    effective_weight: float | None = None
    target_contribution: float | None = None
    aggregate_signal: float | None = None
    portfolio_target: float | None = None
    reason: str | None = None


class ComparisonSeries(Model):
    id: str
    name: str
    config: dict[str, JsonValue]
    analytics: Analytics
    curve: Page[EquityRow]
    membership_turnover: float | None = None
    synthetic: bool


class Comparison(Model):
    series: list[ComparisonSeries]
    differences: list[str]
    warnings: list[str]
    units: Literal["usd", "growth"]
