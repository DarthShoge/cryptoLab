"""Typed local strategy-lab API. Mutation is local simulation only, never trading."""

from contextlib import asynccontextmanager
from datetime import date, datetime
from functools import wraps
import secrets
from typing import Literal

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from arblab.hyperliquid_copy.lab_config import LabConfig as DomainConfig, METRICS
from .lab_jobs import LabJobs
from .lab_models import (
    Annotation,
    Bootstrap,
    Comparison,
    Dataset,
    Experiment,
    LabConfig,
    Preview,
    Submission,
    UniverseRow,
)
from .lab_queries import compare, universe
from .models import Page
from .repository import ReportError

HOSTS = {
    "testserver",
    "127.0.0.1:8010",
    "localhost:8010",
    "127.0.0.1:8011",
    "localhost:8011",
    "127.0.0.1:5174",
    "localhost:5174",
}
ORIGINS = {"http://" + host for host in HOSTS}


def install_lab(app, root, repo):
    token = secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(app):
        jobs = LabJobs(root)
        jobs.start()
        app.state.lab = jobs
        try:
            yield
        finally:
            jobs.close()

    app.router.lifespan_context = lifespan

    @app.middleware("http")
    async def local_guard(request: Request, call_next):
        if request.url.path.startswith("/api/lab/"):
            if request.headers.get("host") not in HOSTS or request.headers.get(
                "origin"
            ) not in ORIGINS | {None}:
                return JSONResponse(
                    {"detail": "Local host/origin required"}, status_code=403
                )
            if request.method not in {"GET", "HEAD", "OPTIONS"}:
                if not secrets.compare_digest(
                    request.headers.get("x-lab-token", ""), token
                ):
                    return JSONResponse(
                        {"detail": "Local UI request token required"}, status_code=403
                    )
                if (
                    request.headers.get("content-type", "").split(";")[0]
                    != "application/json"
                ):
                    return JSONResponse(
                        {"detail": "JSON requests required"}, status_code=415
                    )
                if len(await request.body()) > 65536:
                    return JSONResponse(
                        {"detail": "Request too large"}, status_code=413
                    )
        response = await call_next(request)
        if request.url.path.startswith("/api/lab/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    router = APIRouter(prefix="/api/lab", tags=["Copy strategy lab"])

    def safe(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ValueError as exc:
            # Application validation messages only; parsers/OS errors retain the
            # app's generic handler. Avoid reflecting dataset-controlled paths.
            message = str(exc)
            if "/" in message or "\\" in message:
                message = "Dataset or experiment validation failed"
            raise ReportError(message) from None

    @router.get("/bootstrap", response_model=Bootstrap)
    def bootstrap():
        return Bootstrap(
            token=token,
            enabled=True,
            defaults=LabConfig(),
            metrics=list(METRICS),
            limits={"queue": 32, "comparison": 6, "input_rows": 1000000},
            restrictions=[
                "Local development backtests only; no exchange orders or automatic downloads",
                "Wallet ROI ranking unavailable without verified cash-flow-adjusted equity history; PnL efficiency is not ROI",
                "Real research qualification, validation/test launches and parameter sweeps are not enabled in this increment",
            ],
        )

    @router.get("/datasets", response_model=list[Dataset])
    def datasets():
        return app.state.lab.catalog.list()

    @router.get("/experiments", response_model=list[Experiment])
    def experiments():
        return app.state.lab.store.list()

    @router.post("/experiments", response_model=Experiment, status_code=202)
    def submit(body: Submission):
        return safe(app.state.lab.submit, body)

    @router.post("/previews", response_model=Experiment, status_code=202)
    def preview(body: Preview):
        return safe(app.state.lab.submit, body, kind="cohort_preview")

    @router.get("/experiments/{identifier}", response_model=Experiment)
    def detail(identifier: str):
        return safe(app.state.lab.store.get, identifier)

    @router.patch("/experiments/{identifier}/metadata", response_model=Experiment)
    def annotate(identifier: str, body: Annotation):
        return safe(app.state.lab.store.annotate, identifier, body.name, body.notes)

    @router.post("/experiments/{identifier}/clone", response_model=Submission)
    def clone(identifier: str):
        item = safe(app.state.lab.store.get, identifier)
        return Submission(
            name=(item["name"] + " copy")[:120],
            dataset_id=item["dataset_id"],
            config=item["config"],
            parent_id=identifier,
        )

    @router.post("/experiments/{identifier}/cancel", response_model=Experiment)
    def cancel(identifier: str):
        return safe(app.state.lab.cancel, identifier)

    @router.post("/experiments/{identifier}/resume", response_model=Experiment)
    def resume(identifier: str):
        return safe(app.state.lab.store.resume, identifier)

    @router.get("/experiments/{identifier}/universe", response_model=Page[UniverseRow])
    def history(
        identifier: str,
        table: Literal["rankings", "cohorts", "contributions"] = "cohorts",
        page: int = Query(1, ge=1),
        page_size: int = Query(50, ge=1, le=200),
        decision_date: date | None = None,
        scope: str | None = Query(None, max_length=20),
        wallet: str | None = Query(None, max_length=42),
        at: datetime | None = None,
        selection: Literal["selected", "eligible", "excluded"] | None = None,
    ):
        return safe(
            universe,
            app.state.lab,
            identifier,
            table,
            page=page,
            page_size=page_size,
            decision_date=decision_date,
            scope=scope,
            wallet=wallet,
            at=at,
            selection=selection,
        )

    @router.get("/compare", response_model=Comparison)
    def comparison(
        ids: str = Query(max_length=250), units: Literal["usd", "growth"] = "growth"
    ):
        return safe(compare, app.state.lab, repo, ids.split(","), units)

    app.include_router(router)
