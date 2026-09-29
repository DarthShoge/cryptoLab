"""Shared causal signal artifact followed by independent latency simulations.

This reference pipeline accepts already normalized, bounded datasets. The CLI
enforces resource limits before loading; it never silently samples wallets.
"""
from bisect import bisect_left, insort
from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import timedelta

from .contracts import semantic_hash, utc
from .data_quality import validate_fills, validate_market
from .positions import PositionReplay
from .ranking import RankingConfig, rank_traders
from .signals import SIGNALS, aggregate, conviction_value
from .simulator import SimulationConfig, simulate


class RollingScale:
    def __init__(self,lookback_days,quantile):
        self.window = timedelta(days=lookback_days)
        self.quantile = quantile
        self.queue, self.ordered = deque(), []

    def add(self,at,value):
        self.queue.append((at,value))
        insort(self.ordered,value)

    def value(self,at):
        while self.queue and self.queue[0][0] < at-self.window:
            _, value = self.queue.popleft()
            self.ordered.pop(bisect_left(self.ordered,value))
        if not self.ordered:
            return None
        index = (len(self.ordered)-1)*self.quantile
        low = int(index)
        high = min(low+1,len(self.ordered)-1)
        return self.ordered[low]+(self.ordered[high]-self.ordered[low])*(index-low)


@dataclass
class RunResult:
    scores: list
    cohorts: list
    signals: list
    strategies: dict
    controls: dict
    warnings: list


def _shuffled(rows,ranking,config_hash,dataset_hash,decision,scope):
    eligible = sorted([r for r in rows if r.score is not None],key=lambda r:r.user)
    recipients = sorted(eligible,key=lambda r:semantic_hash(dict(config_hash=config_hash,dataset_hash=dataset_hash,
                       decision_time=decision,scope=scope,address=r.user,purpose="score_shuffle_v1")))
    scores = {recipient.user:donor.score for recipient,donor in zip(recipients,eligible)}
    size = sum(r.selected for r in rows)
    return dict(sorted(scores.items(),key=lambda p:(-p[1],p[0]))[:size])


def run_offline(fills, market, start, end, warmup_start, config, *, dataset_sha256):
    start, end, warmup_start = utc(start), utc(end), utc(warmup_start)
    coins = tuple(config["coins"])
    smoke = config["mode"] == "smoke_only"
    if config["research_eligible"] and smoke:
        raise ValueError("smoke is never research eligible")
    semantics = config["closed_pnl_fee_semantics"]
    if semantics == "unknown" and not smoke:
        raise ValueError("unresolved fee semantics")
    if not warmup_start < start < end or any(t.second or t.microsecond for t in (start,end,warmup_start)):
        raise ValueError("minute-aligned warmup and simulation required")
    fills = sorted(fills,key=lambda f:f.order_key)
    validate_fills(fills,min([f.exchange_time for f in fills],default=warmup_start),end).assert_accepted()
    quality = validate_market(market,coins,start,end,research=config["research_eligible"],warmup_start=warmup_start,
                              l2_threshold=config.get("l2_coverage",.95),funding_threshold=config.get("funding_coverage",.95))
    quality.assert_accepted()
    # A missing settlement is diagnostic evidence, never permission to set rate=0.
    if any(i.code == "funding_coverage" for i in quality.issues):
        raise ValueError("missing funding settlement prevents simulation")
    replay = PositionReplay(fills)
    ranking = RankingConfig(**config.get("ranking",{}))
    score_rows, cohorts, signals, controls = [], [], [], {"all_eligible":[],"score_shuffled":[]}
    scales = {}
    at, last_day = start, None
    config_hash = semantic_hash(config)
    while at < end:
        if at.date() != last_day:
            selection, eligible, shuffled = {}, {}, {}
            decision = at.replace(hour=0,minute=0,second=0,microsecond=0)
            scopes = coins if config.get("asset_specialist",False) else (None,)
            for scope in scopes:
                ranked = rank_traders(fills,decision,ranking,semantics,coin=scope,smoke=smoke)
                score_rows.extend(asdict(r) for r in ranked)
                cohort = {r.user:r.score for r in ranked if r.selected}
                all_scores = {r.user:r.score for r in ranked if r.score is not None}
                shuffle = _shuffled(ranked,ranking,config_hash,dataset_sha256,decision,scope)
                cohorts.append(dict(decision_time=decision,coin=scope,members=sorted(cohort),
                                    cutoff_address=next(reversed(cohort),None)))
                for c in (coins if scope is None else (scope,)):
                    selection[c],eligible[c],shuffled[c] = cohort,all_scores,shuffle
            required = {(u,c) for c in coins for u in selection[c]}
            scales = {k:v for k,v in scales.items() if k in required}
            for key in sorted(required-set(scales)):
                scale = RollingScale(config.get("scale_lookback_days",30),config.get("scale_quantile",.95))
                moment = max(warmup_start,at-scale.window)
                while moment < at:
                    qty = replay.position(key,moment,inclusive=False)
                    if qty is not None:
                        scale.add(moment,abs(qty)*market.mark(key[1],moment))
                    moment += timedelta(minutes=1)
                scales[key] = scale
            last_day = at.date()
        for coin in coins:
            mid = market.mark(coin,at)
            positions = {u:replay.position((u,coin),at,inclusive=False) for u in eligible[coin]}
            values = {u:positions.get(u) for u in selection[coin]}
            conviction = {u:conviction_value(position_qty=q,mid=mid,scale=scales[u,coin].value(at)) if q is not None else None
                          for u,q in values.items()}
            options = dict(min_known=config.get("min_known",5),min_weight=config.get("min_known_weight",.6),trim=config.get("trim",.1))
            for name in SIGNALS:
                signal = aggregate(conviction if name == "conviction_trimmed" else values,selection[coin],name,**options)
                signals.append(dict(time=at,coin=coin,signal_name=name,**asdict(signal)))
            for name,scores,method in [("all_eligible",eligible[coin],"direction_equal"),
                                       ("score_shuffled",shuffled[coin],"direction_score_weighted")]:
                signal = aggregate({u:positions.get(u) for u in scores},scores,method,**options)
                controls[name].append(dict(time=at,coin=coin,value=signal.value))
            for user,qty in values.items():
                if qty is not None:
                    scales[user,coin].add(at,abs(qty)*mid)
        at += timedelta(minutes=1)
    strategies, control_results = {}, {}
    base = SimulationConfig(**config.get("simulation",{}))
    for latency in (1,5,15,60):
        simulation = replace(base,latency_seconds=latency)
        for name in SIGNALS:
            strategies[name,latency] = simulate([r for r in signals if r["signal_name"] == name],market,start,end,coins,simulation)
        for name,rows in controls.items():
            control_results[name,latency] = simulate(rows,market,start,end,coins,simulation)
        for name,targets in [("btc_buy_hold",{"BTC":1}), ("equal_universe_buy_hold",{c:1/len(coins) for c in coins})]:
            if not set(targets) <= set(coins):
                raise ValueError("BTC benchmark requires BTC in universe")
            rows = [dict(time=start,coin=c,value=v) for c,v in targets.items()]
            control_results[name,latency] = simulate(rows,market,start,end,coins,replace(simulation,asset_cap=1,deadband=0,min_trade_usd=0))
    control_results["cash",None] = simulate([],market,start,end,coins,base)
    warnings = ["INTEGRATION ONLY: smoke is not research evidence", "unresolved_fee_semantics"] if smoke else []
    warnings.append("funding_uses_settlement_mid_approximation")
    if any(r.insolvent for r in list(strategies.values())+list(control_results.values())):
        raise ValueError("insolvent scenario; no invented liquidation fill")
    return RunResult(score_rows,cohorts,signals,strategies,control_results,warnings)
