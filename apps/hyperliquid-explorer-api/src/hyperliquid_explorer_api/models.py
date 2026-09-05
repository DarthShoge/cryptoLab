"""HTTP contracts; generated TypeScript consumes these models."""

from datetime import datetime
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class Model(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)


class Position(Model):
    coin: str
    qty: float
    entry: float


class Scenario(Model):
    scenario_type: Literal["strategy", "control"]
    name: str
    latency_seconds: int | None
    metrics: dict[str, float | None]
    warnings: list[str] = Field(default_factory=list)
    positions: list[Position] = Field(default_factory=list)


class Run(Model):
    id: str
    available: bool = True
    synthetic: bool = False
    mode: str = "unavailable"
    start: datetime | None = None
    end: datetime | None = None
    initial_equity: float | None = None
    scenario_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class RunDetail(Run):
    scenarios: list[Scenario]
    config: dict[str, JsonValue]
    provenance: dict[str, JsonValue]
    reconciliation: dict[str, JsonValue]
    artifacts: list[str]


class EquityRow(Model):
    time: datetime
    equity: float
    cash: float
    unrealized_pnl: float
    gross_exposure: float
    net_exposure: float
    drawdown: float


class Record(Model):
    """Typed union of available ledger columns, rather than arbitrary JSON rows."""

    coin: str | None = None
    user: str | None = None
    decision_time: datetime | None = None
    score: float | None = None
    selected: bool | None = None
    exclusions: list[str] = Field(default_factory=list)
    metrics: dict[str, float] | None = None
    members: list[str] = Field(default_factory=list)
    cutoff_address: str | None = None
    signal_time: datetime | None = None
    book_time: datetime | None = None
    time: datetime | None = None
    requested_qty: float | None = None
    filled_qty: float | None = None
    unfilled_qty: float | None = None
    vwap: float | None = None
    arrival_mid: float | None = None
    signal_mid: float | None = None
    requested_notional: float | None = None
    fee: float | None = None
    spread_cost: float | None = None
    reason: str | None = None
    qty: float | None = None
    mark: float | None = None
    rate: float | None = None
    cash_delta: float | None = None


T = TypeVar("T")


class Page(Model, Generic[T]):
    rows: list[T]
    total: int
    available: bool = True
    reason: str | None = None
    downsampled: bool = False


class Metric(Model):
    value: float | None
    unit: Literal["usd", "percent", "ratio", "minutes", "count"]
    reason: str | None = None
    source: Literal["stored summary", "Python-derived"] = "stored summary"
    start: datetime | None
    end: datetime | None
    samples: int
    description: str
    group: str


class Analytics(Model):
    metrics: dict[str, Metric]
    benchmark: dict[str, Metric] | None = None
    warnings: list[str]
    conventions: str


class Health(Model):
    status: str = "ok"
    schema_version: str = "1"
