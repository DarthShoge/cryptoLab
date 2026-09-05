"""Loopback read-only HTTP interface. No trading, downloads or job submission."""

from datetime import date
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError
import duckdb

from arblab.paths import reports_dir
from .analytics import analyze
from .models import Analytics, EquityRow, Health, Page, Record, Run, RunDetail
from .queries import Curve, records
from .repository import ReportError, Repository

Kind = Literal["strategy", "control"]


def create_app(root=None, web_root=None):
    repo = Repository(
        root or os.environ.get("HYPERLIQUID_REPORTS_ROOT") or reports_dir()
    )
    web = web_root or os.environ.get("HYPERLIQUID_WEB_ROOT")
    web = Path(web).resolve() if web else None
    app = FastAPI(
        title="Hyperliquid Read-only Explorer",
        version="1.0.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )

    @app.exception_handler(ReportError)
    async def report_error(request: Request, exc: ReportError):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status)

    async def malformed(request: Request, exc: Exception):
        return JSONResponse(
            {"detail": "Report data is unavailable or malformed"}, status_code=422
        )

    for error in (
        duckdb.Error,
        ValidationError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ):
        app.add_exception_handler(error, malformed)

    @app.get("/api/health", response_model=Health)
    def health():
        return Health()

    @app.get("/api/runs", response_model=list[Run])
    def runs():
        return repo.list()

    @app.get("/api/runs/{run_id}", response_model=RunDetail)
    def detail(run_id: str):
        return repo.detail(run_id)

    @app.get("/api/runs/{run_id}/equity", response_model=Page[EquityRow])
    def equity(
        run_id: str,
        scenario_type: Kind = "strategy",
        name: str = "direction_equal",
        latency_seconds: int | None = Query(5, ge=0),
        max_points: int = Query(2000, ge=10, le=2000),
    ):
        return Curve(
            repo, run_id, repo.scenario(run_id, scenario_type, name, latency_seconds)
        ).page(max_points)

    @app.get("/api/runs/{run_id}/analytics", response_model=Analytics)
    def analytics(
        run_id: str,
        scenario_type: Kind = "strategy",
        name: str = "direction_equal",
        latency_seconds: int | None = Query(5, ge=0),
        benchmark_type: Kind = "control",
        benchmark_name: str | None = None,
        benchmark_latency: int | None = Query(5, ge=0),
    ):
        scenario = repo.scenario(run_id, scenario_type, name, latency_seconds)
        benchmark = (
            repo.scenario(run_id, benchmark_type, benchmark_name, benchmark_latency)
            if benchmark_name
            else None
        )
        return analyze(repo, run_id, scenario, benchmark)

    @app.get("/api/runs/{run_id}/traders", response_model=Page[Record])
    def traders(
        run_id: str,
        page: int = Query(1, ge=1),
        page_size: int = Query(50, ge=1, le=200),
        wallet: str | None = Query(None, max_length=42),
        decision_date: date | None = None,
        scope: str | None = Query(None, max_length=20),
    ):
        return records(
            repo,
            run_id,
            "traders",
            page=page,
            page_size=page_size,
            wallet=wallet,
            decision_date=decision_date,
            scope=scope,
        )

    @app.get("/api/runs/{run_id}/cohorts", response_model=Page[Record])
    def cohorts(
        run_id: str,
        page: int = Query(1, ge=1),
        page_size: int = Query(50, ge=1, le=200),
        decision_date: date | None = None,
        scope: str | None = Query(None, max_length=20),
    ):
        return records(
            repo,
            run_id,
            "cohorts",
            page=page,
            page_size=page_size,
            decision_date=decision_date,
            scope=scope,
        )

    @app.get("/api/runs/{run_id}/fills", response_model=Page[Record])
    def fills(
        run_id: str,
        scenario_type: Kind = "strategy",
        name: str = "direction_equal",
        latency_seconds: int | None = Query(5, ge=0),
        page: int = Query(1, ge=1),
        page_size: int = Query(50, ge=1, le=200),
        coin: str | None = Query(None, max_length=20),
        reason: str | None = Query(None, max_length=40),
    ):
        return records(
            repo,
            run_id,
            "fills",
            scenario=repo.scenario(run_id, scenario_type, name, latency_seconds),
            page=page,
            page_size=page_size,
            coin=coin,
            reason=reason,
        )

    @app.get("/api/runs/{run_id}/funding", response_model=Page[Record])
    def funding(
        run_id: str,
        scenario_type: Kind = "strategy",
        name: str = "direction_equal",
        latency_seconds: int | None = Query(5, ge=0),
        page: int = Query(1, ge=1),
        page_size: int = Query(50, ge=1, le=200),
        coin: str | None = Query(None, max_length=20),
    ):
        return records(
            repo,
            run_id,
            "funding",
            scenario=repo.scenario(run_id, scenario_type, name, latency_seconds),
            page=page,
            page_size=page_size,
            coin=coin,
        )

    @app.get("/api/runs/{run_id}/artifacts/{name}")
    def artifact(run_id: str, name: str):
        path = repo.artifact(run_id, name)
        return FileResponse(
            path,
            filename=name,
            media_type="application/octet-stream",
            headers={"X-Content-Type-Options": "nosniff"},
        )

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        if path == "api" or path.startswith("api/") or web is None:
            return JSONResponse({"detail": "Not found"}, status_code=404)
        target = (web / path).resolve()
        if not target.is_relative_to(web):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        if not target.is_file():
            target = web / "index.html" if not Path(path).suffix else target
        if not target.is_file() or not target.resolve().is_relative_to(web):
            return JSONResponse(
                {"detail": "Frontend build unavailable"}, status_code=404
            )
        return FileResponse(target)

    return app


app = create_app()
