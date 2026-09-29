"""Reject misspelled/ignored trial settings before accessing the dataset."""
from dataclasses import fields
from importlib.metadata import version

from .contracts import canonical_json
from .ranking import RankingConfig
from .simulator import SimulationConfig

KEYS = {"schema","mode","research_eligible","coins","closed_pnl_fee_semantics","asset_specialist",
        "scale_lookback_days","scale_quantile","min_known","min_known_weight","trim","l2_coverage",
        "funding_coverage","ranking","simulation","dependency_versions"}


def validate_configuration(config):
    if set(config) != KEYS:
        raise ValueError(f"unknown or missing configuration keys: {sorted(set(config)^KEYS)}")
    canonical_json(config)  # Also rejects non-finite values.
    if config["schema"] != "hyperliquid_ensemble_v1" or config["mode"] not in ("smoke_only","historical_research"):
        raise ValueError("unsupported schema/mode")
    if type(config["research_eligible"]) is not bool or config["research_eligible"] != (config["mode"] == "historical_research"):
        raise ValueError("smoke/research eligibility mismatch")
    if config["coins"] != ["BTC","ETH","SOL"]:
        raise ValueError("v1 universe is fixed to BTC ETH SOL")
    if config["closed_pnl_fee_semantics"] not in ("unknown","gross_excludes_fee","net_includes_fee"):
        raise ValueError("unknown fee semantic setting")
    if type(config["asset_specialist"]) is not bool:
        raise ValueError("asset_specialist must be boolean")
    for key in ("scale_lookback_days","min_known"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"positive integer required: {key}")
    for key in ("scale_quantile","min_known_weight","l2_coverage","funding_coverage"):
        if not 0 < config[key] <= 1:
            raise ValueError(f"invalid fraction: {key}")
    if not 0 <= config["trim"] < .5:
        raise ValueError("invalid trim")
    for key,kind in [("ranking",RankingConfig),("simulation",SimulationConfig)]:
        if set(config[key]) != {f.name for f in fields(kind)}:
            raise ValueError(f"unknown or missing {key} setting")
        instance = kind(**config[key])
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or v < 0 for v in config[key].values()):
            raise ValueError(f"invalid numeric {key} setting")
    ranking = RankingConfig(**config["ranking"])
    if not ranking.lookback_days or ranking.min_cohort < 1 or ranking.max_cohort < ranking.min_cohort or not 0 < ranking.top_fraction <= 1:
        raise ValueError("invalid ranking window/cohort")
    if min(ranking.drawdown_floor,ranking.duration_cap_minutes,ranking.fragmentation_floor) <= 0:
        raise ValueError("zero ranking divisor")
    sim = SimulationConfig(**config["simulation"])
    if not 0 < sim.gross_cap <= 1 or not 0 < sim.asset_cap <= 1 or sim.initial_equity <= 0:
        raise ValueError("invalid capital/exposure cap")
    if sim.latency_seconds != 5 or sim.max_mark_age != 60 or sim.max_book_lag != 2:
        raise ValueError("v1 uses fixed latency sweep and 60s/2s data boundaries")
    if config["l2_coverage"] < .95 or config["funding_coverage"] < (.95 if config["mode"] == "smoke_only" else 1):
        raise ValueError("cannot weaken v1 coverage gates")
    for package,pinned in config["dependency_versions"].items():
        if version(package) != pinned:
            raise ValueError(f"dependency mismatch: {package}")
