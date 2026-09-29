"""Typed lab HTTP contracts generated from the domain configuration."""

from datetime import datetime
from typing import Literal, Annotated
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from pydantic.dataclasses import dataclass
from arblab.hyperliquid_copy.lab_config import LabConfig as DomainConfig
from arblab.hyperliquid_copy.lab_config_v2 import LabConfigV2 as DomainConfigV2
from arblab.hyperliquid_copy.lab_config_proxy import LabConfigProxy as DomainConfigProxy
from arblab.hyperliquid_copy.lab_config_proxy import (
    LabConfigProxyScheduled as DomainConfigProxyScheduled,
)
from arblab.hyperliquid_copy.lab_config_codec import parse_lab_config
from .models import Analytics, EquityRow, Page, Model

LabConfig = dataclass(DomainConfig, frozen=True, config=ConfigDict(extra="forbid"))
LabConfigV2 = dataclass(DomainConfigV2, frozen=True, config=ConfigDict(extra="forbid"))
LabConfigProxy = dataclass(
    DomainConfigProxy, frozen=True, config=ConfigDict(extra="forbid")
)
LabConfigProxyScheduled = dataclass(
    DomainConfigProxyScheduled, frozen=True, config=ConfigDict(extra="forbid")
)
AnyLabConfig = Annotated[
    LabConfig | LabConfigV2 | LabConfigProxy | LabConfigProxyScheduled,
    Field(discriminator="schema_version"),
]


class Submission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(default="Untitled hypothesis", min_length=1, max_length=120)
    dataset_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    config: AnyLabConfig
    parent_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")

    @model_validator(mode="before")
    @classmethod
    def strict_domain(cls, value):
        if isinstance(value, dict) and isinstance(value.get("config"), dict):
            parse_lab_config(value["config"])
        return value


class Preview(Submission):
    decision_date: str
    scope: str | None = None


class PreviewInfo(Model):
    hypothetical: bool | None = None


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
    estimate_notes: list[str] = Field(default_factory=list)


class Annotation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    notes: str = Field(max_length=4000)


class Experiment(Model):
    id: str
    name: str
    notes: str
    dataset_id: str
    config: AnyLabConfig
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
    default_config: AnyLabConfig | None = None
    supported_classes: list[str] = Field(default_factory=list)
    liquidity_available: bool = False
    catalogue_hash: str | None = None
    pricing_mode: Literal["order_book", "hourly_proxy"] = "order_book"
    proxy_mappings: list[dict[str, JsonValue]] = Field(default_factory=list)


class Bootstrap(Model):
    token: str
    enabled: bool
    defaults: LabConfig
    metrics: list[str]
    limits: dict[str, int]
    restrictions: list[str]


class UniverseRow(Model):
    decision_time: datetime | None = None
    market_decision_time: datetime | None = None
    decision_trigger: str | None = None
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
    market_membership_turnover: float | None = None
    mean_selected_assets: float | None = None
    synthetic: bool


class InstrumentRow(Model):
    instrument_id: str
    display_name: str
    venue: str
    asset_class: str
    supported: bool
    listed_at: datetime | None = None
    delisted_at: datetime | None = None
    known_at: datetime | None = None
    effective_from: datetime
    effective_to: datetime | None = None
    availability_basis: str | None = None
    proxy_ticker: str | None = None
    proxy_unit: str | None = None
    calendar: str | None = None


class MarketRow(Model):
    decision_time: datetime
    effective_at: datetime
    instrument_id: str | None = None
    display_name: str | None = None
    venue: str | None = None
    asset_class: str | None = None
    availability_basis: str | None = None
    proxy_ticker: str | None = None
    volume_usd: float | None = None
    rank: int | None = None
    eligible: bool | None = None
    selected: bool | None = None
    reasons: list[str] | None = None
    budget: float | None = None
    window_start: datetime | None = None
    window_end: datetime | None = None
    members: list[str] | None = None
    entries: list[str] | None = None
    exits: list[str] | None = None
    candidate_count: int | None = None
    eligible_count: int | None = None
    selected_count: int | None = None
    requested_count: int | None = None
    retention: float | None = None
    membership_turnover: float | None = None


class Comparison(Model):
    series: list[ComparisonSeries]
    differences: list[str]
    warnings: list[str]
    units: Literal["usd", "growth"]
