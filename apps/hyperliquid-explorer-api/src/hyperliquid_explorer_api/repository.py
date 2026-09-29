"""Local artifact discovery and containment. No caller-selected filesystem paths."""

import json
import math
from pathlib import Path
import re
import duckdb

from .models import Position, Run, RunDetail, Scenario

ANNUALIZED = {
    "sharpe",
    "sortino",
    "calmar",
    "annualized_return",
    "annualized_volatility",
}

ARTIFACTS = frozenset(
    {
        "config.json",
        "summary.json",
        "data_manifest.json",
        "trial.json",
        "reconciliation.json",
        "report.md",
        "trader_scores.parquet",
        "cohort_history.parquet",
        "signals.parquet",
        "simulated_fills.parquet",
        "equity_curve.parquet",
        "funding_ledger.parquet",
        "control_funding_ledger.parquet",
        "proxy_requests.parquet",
        "control_fills.parquet",
        "control_equity_curve.parquet",
    }
)


class ReportError(Exception):
    def __init__(self, detail="Report data is unavailable or malformed", status=422):
        self.detail, self.status = detail, status
        super().__init__(detail)


def number(value):
    if value is None or isinstance(value, (dict, list, bool)):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def public_json(value):
    """Redact sensitive metadata keys recursively; preserve identifiers as strings."""
    if isinstance(value, dict):
        return {
            k: public_json(v)
            for k, v in value.items()
            if not any(
                word in k.lower()
                for word in (
                    "password",
                    "secret",
                    "token",
                    "credential",
                    "api_key",
                    "path",
                    "directory",
                    "root",
                )
            )
        }
    if isinstance(value, list):
        return [public_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str) and (
        value.startswith(("/", "file://")) or re.match(r"^[A-Za-z]:[\\/]", value)
    ):
        return "[local path omitted]"
    return value


class Repository:
    def __init__(self, root, lab_root=None):
        self.root = Path(root).resolve()
        self.lab_root = Path(lab_root).resolve() if lab_root else None

    def directory(self, run_id):
        if not re.fullmatch(r"hyperliquid_trader_ensemble_[A-Za-z0-9_-]+", run_id):
            raise ReportError("Unknown report", 404)
        path = (self.root / run_id).resolve()
        if not path.is_dir() and self.lab_root:
            import sqlite3

            identifier = run_id.removeprefix("hyperliquid_trader_ensemble_")
            database = self.lab_root / "experiments.sqlite3"
            if re.fullmatch(r"[a-f0-9]{32}", identifier) and database.is_file():
                db = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
                try:
                    completed = db.execute(
                        "SELECT 1 FROM experiments WHERE id=? AND status='completed' AND run_id=?",
                        [identifier, run_id],
                    ).fetchone()
                finally:
                    db.close()
                candidate = (self.lab_root / "reports" / run_id).resolve()
                if (
                    completed
                    and candidate.is_relative_to(self.lab_root / "reports")
                    and candidate.is_dir()
                ):
                    return candidate
        if not path.is_relative_to(self.root) or path == self.root or not path.is_dir():
            raise ReportError("Unknown report", 404)
        return path

    def artifact(self, run_id, name, *, optional=False):
        if name not in ARTIFACTS:
            raise ReportError("Unknown artifact", 404)
        directory = self.directory(run_id)
        path = (directory / name).resolve()
        if not path.is_relative_to(directory):
            raise ReportError("Unknown artifact", 404)
        if not path.is_file():
            if optional:
                return None
            raise ReportError("Artifact unavailable", 404)
        return path

    def json(self, run_id, name, *, optional=False):
        path = self.artifact(run_id, name, optional=optional)
        if path is None:
            return {}
        if path.stat().st_size > 10 * 1024 * 1024:
            raise ReportError("Report metadata exceeds supported size")
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError()
            return value
        except (ValueError, UnicodeError, OSError):
            raise ReportError() from None

    def detail(self, run_id):
        from .queries import Curve

        summary = self.json(run_id, "summary.json")
        if summary.get("schema") != "hyperliquid_copy_report_v1":
            raise ReportError("Unsupported report schema")
        config = self.json(run_id, "config.json", optional=True)
        manifest = self.json(run_id, "data_manifest.json", optional=True)
        scenarios = []
        for row in summary["scenarios"]:
            scenarios.append(
                Scenario(
                    scenario_type=row["scenario_type"],
                    name=row.get("signal_name") or row["control_name"],
                    latency_seconds=row["latency_seconds"],
                    metrics={
                        k: number(
                            int(v)
                            if k == "terminal_open_drawdown" and isinstance(v, bool)
                            else v
                        )
                        for k, v in row.items()
                        if k
                        not in {
                            "scenario_type",
                            "signal_name",
                            "control_name",
                            "latency_seconds",
                        }
                        and not isinstance(v, (list, dict))
                    },
                    warnings=row.get("warnings", []),
                    positions=[
                        Position(coin=c, **p)
                        for c, p in row.get("residual_positions", {}).items()
                    ],
                )
            )
        if not scenarios:
            raise ReportError("Report contains no scenarios")
        stats = Curve(self, run_id, scenarios[0]).stats()
        warnings = summary.get("warnings", [])
        synthetic = manifest.get("synthetic") is True or any(
            "synthetic" in w.lower() for w in warnings
        )
        # Apply the same policy to comparison-table values as to headline cards.
        # Raw downloadable artifacts are deliberately left unchanged.
        for scenario in scenarios:
            scenario_stats = {} if synthetic else Curve(self, run_id, scenario).stats()
            start, end = scenario_stats.get("start"), scenario_stats.get("end")
            if (
                synthetic
                or not start
                or not end
                or (end - start).total_seconds() < 86400
            ):
                scenario.metrics.update(
                    {key: None for key in ANNUALIZED if key in scenario.metrics}
                )
        artifacts = [
            name
            for name in sorted(ARTIFACTS)
            if self.artifact(run_id, name, optional=True)
        ]
        return RunDetail(
            id=run_id,
            synthetic=synthetic,
            mode="synthetic demo" if synthetic else config.get("mode", "unknown"),
            start=stats.get("start"),
            end=stats.get("end"),
            initial_equity=stats.get("initial"),
            scenario_count=len(scenarios),
            warnings=warnings,
            scenarios=scenarios,
            artifacts=artifacts,
            config=public_json(config),
            provenance=public_json(manifest),
            reconciliation=public_json(
                self.json(run_id, "reconciliation.json", optional=True)
            ),
        )

    def list(self):
        if not self.root.is_dir():
            return []
        rows = []
        for path in sorted(
            self.root.glob("hyperliquid_trader_ensemble_*"), reverse=True
        ):
            try:
                self.directory(path.name)
                rows.append(Run(**self.detail(path.name).model_dump()))
            except (
                ReportError,
                ValueError,
                KeyError,
                TypeError,
                OSError,
                duckdb.Error,
            ):
                rows.append(
                    Run(
                        id=path.name,
                        available=False,
                        warnings=["Report unavailable or malformed"],
                    )
                )
        return rows

    def scenario(self, run_id, kind, name, latency):
        detail = self.detail(run_id)
        for scenario in detail.scenarios:
            if (scenario.scenario_type, scenario.name, scenario.latency_seconds) == (
                kind,
                name,
                None if name == "cash" and kind == "control" else latency,
            ):
                return scenario
        raise ReportError("Unknown scenario", 404)
