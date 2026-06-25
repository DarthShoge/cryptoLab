"""Curated-universe dynamic long/short traffic-light strategy."""

from __future__ import annotations

from typing import Any, Dict, List

from arblab.backtest.asset_universe import curated_kamino_universe, research_symbols
from arblab.backtest.market import MarketParams
from arblab.backtest.strategy import BarData, Strategy
from arblab.backtest.traffic_lights import asset_green_counts_by_bar, latest_asset_rankings
from arblab.kamino_risk import AccountSnapshot, CollateralPosition, DebtPosition


class MultiAssetTrafficLightStrategy(Strategy):
    """Research strategy that rotates long collateral and short debt by signal rank."""

    def setup(
        self,
        snapshot: AccountSnapshot,
        config: Dict[str, Any],
    ) -> AccountSnapshot:
        self._config = dict(config)
        self.event_log: list[dict[str, Any]] = []
        self._history_fields: dict[str, Any] = {
            "selected_long": "",
            "selected_short": "",
            "long_green": 0,
            "short_green": 0,
            "target_long_fraction": 0.0,
            "target_short_fraction": 0.0,
            "equity_drawdown_pct": 0.0,
        }
        self._peak_equity = 0.0
        market_params = MarketParams.kamino_defaults()
        universe = curated_kamino_universe()
        symbols = list(
            config.get(
                "directional_symbols",
                research_symbols(universe, long_short_only=True),
            )
        )
        initial_prices = dict(config.get("initial_prices", {}))

        initial_collateral_symbol = config.get("initial_collateral_symbol")
        initial_collateral_amount = float(config.get("initial_collateral_amount", 0.0))
        initial_usdc = float(config.get("initial_usdc", 10_000.0))
        if initial_collateral_symbol:
            initial_usdc = 0.0

        collateral = [
            CollateralPosition(
                symbol="USDC",
                amount=initial_usdc,
                price=1.0,
                ltv=market_params.assets["USDC"].ltv,
                liquidation_threshold=market_params.assets["USDC"].liquidation_threshold,
            )
        ]
        debt: list[DebtPosition] = [
            DebtPosition(
                symbol="USDC",
                amount=0.0,
                price=1.0,
                borrow_factor=market_params.assets["USDC"].borrow_factor,
            )
        ]

        for symbol in symbols:
            asset = market_params.assets[symbol]
            price = float(initial_prices.get(symbol, 1.0))
            collateral.append(
                CollateralPosition(
                    symbol=symbol,
                    amount=(
                        initial_collateral_amount
                        if symbol == initial_collateral_symbol
                        else 0.0
                    ),
                    price=price,
                    ltv=asset.ltv,
                    liquidation_threshold=asset.liquidation_threshold,
                )
            )
            debt.append(
                DebtPosition(
                    symbol=symbol,
                    amount=0.0,
                    price=price,
                    borrow_factor=asset.borrow_factor,
                )
            )

        return AccountSnapshot(collateral=collateral, debt=debt)

    def on_bar(
        self,
        snapshot: AccountSnapshot,
        bar: BarData,
    ) -> List[Dict[str, Any]]:
        config = self._config
        symbols = list(config.get("directional_symbols", []))
        if not symbols:
            symbols = [
                symbol
                for symbol in research_symbols(long_short_only=True)
                if symbol in bar.prices
            ]

        scores = self._scores_for_bar(config, bar, symbols)
        ranking = latest_asset_rankings(scores)
        min_long_green = int(config.get("min_long_green", len(config.get("timeframes", (1,)))))
        max_short_green = int(config.get("max_short_green", 1))

        selected_long = (
            ranking.strongest
            if ranking.green_counts[ranking.strongest] >= min_long_green
            else ""
        )
        selected_short = (
            ranking.weakest
            if ranking.green_counts[ranking.weakest] <= max_short_green
            and ranking.weakest != selected_long
            else ""
        )
        protective_short = self._protective_short_symbol(
            config,
            selected_long,
            symbols,
            ranking.green_counts,
        )
        hedge_basis_symbol = self._protective_long_basis_symbol(
            snapshot=snapshot,
            selected_long=selected_long,
            short_symbol=protective_short,
            symbols=symbols,
        )

        actions: list[dict[str, Any]] = []
        equity = max(snapshot.total_collateral_value() - snapshot.total_debt_value(), 0.0)
        if equity <= 0.0:
            return []

        threshold = float(config.get("rebalance_threshold", 0.05))
        target_long_fraction, equity_drawdown = self._target_long_fraction(
            equity=equity,
            base_target=float(config.get("target_long_fraction", 0.50)),
            config=config,
            long_green=ranking.green_counts.get(selected_long, 0),
        )
        target_short_fraction = self._target_short_fraction(
            config=config,
            long_green=ranking.green_counts.get(
                hedge_basis_symbol,
                ranking.green_counts.get(selected_long, 0),
            ),
        )
        fee = float(config.get("swap_fee_bps", bar.market_params.swap_fee_bps)) / 10_000.0

        self._history_fields = {
            "selected_long": selected_long,
            "selected_short": selected_short,
            "long_green": ranking.green_counts.get(selected_long, 0),
            "short_green": ranking.green_counts.get(selected_short, 0),
            "target_long_fraction": target_long_fraction,
            "target_short_fraction": target_short_fraction,
            "equity_drawdown_pct": equity_drawdown * 100.0,
        }

        actions.extend(
            self._cover_excess_protective_short(
                snapshot=snapshot,
                selected_long=selected_long,
                hedge_basis_symbol=hedge_basis_symbol,
                protective_short_symbol=protective_short,
                symbols=symbols,
                target_short_fraction=target_short_fraction,
                threshold_usd=equity * threshold,
                fee=fee,
            )
        )
        actions.extend(
            self._rebalance_collateral(
                snapshot=snapshot,
                selected_long=selected_long,
                symbols=symbols,
                target_long_usd=equity * target_long_fraction if selected_long else 0.0,
                threshold_usd=equity * threshold,
                fee=fee,
            )
        )
        actions.extend(
            self._rebalance_debt(
                snapshot=snapshot,
                selected_short=selected_short,
                hedge_basis_symbol=hedge_basis_symbol,
                protective_short_symbol=protective_short,
                symbols=symbols,
                target_short_usd=(
                    target_short_fraction
                    if bool(config.get("enable_protective_short_hedge", False))
                    else equity * target_short_fraction if selected_short else 0.0
                ),
                threshold_usd=equity * threshold,
                fee=fee,
            )
        )

        if actions:
            self.event_log.append(
                {
                    "timestamp": bar.timestamp,
                    "selected_long": selected_long,
                    "selected_short": selected_short,
                    "green_counts": ranking.green_counts,
                    "target_long_fraction": target_long_fraction,
                    "target_short_fraction": target_short_fraction,
                    "equity_drawdown_pct": equity_drawdown * 100.0,
                    "action_count": len(actions),
                }
            )
        return actions

    def history_fields(self) -> dict[str, Any]:
        return dict(self._history_fields)

    def _scores_for_bar(
        self,
        config: dict[str, Any],
        bar: BarData,
        symbols: list[str],
    ):
        precomputed = config.get("green_scores")
        if precomputed is not None:
            if 0 <= bar.bar_index < len(precomputed):
                return precomputed.iloc[[bar.bar_index]][symbols]
            scores = precomputed.loc[[bar.timestamp], symbols]
            if not scores.empty:
                return scores
        return asset_green_counts_by_bar(
            bar.history,
            symbols=symbols,
            atr_period=int(config.get("atr_period", 10)),
            multiplier=float(config.get("supertrend_multiplier", 3.0)),
            timeframes=tuple(config.get("timeframes", ("1h", "4h", "8h", "1d"))),
        )

    def _target_long_fraction(
        self,
        equity: float,
        base_target: float,
        config: dict[str, Any],
        long_green: int,
    ) -> tuple[float, float]:
        self._peak_equity = max(self._peak_equity, equity)
        drawdown = (
            (self._peak_equity - equity) / self._peak_equity
            if self._peak_equity > 0.0
            else 0.0
        )
        if not bool(config.get("enable_drawdown_governor", False)):
            return self._signal_governed_target(base_target, config, long_green), drawdown

        selected_target = base_target
        tiers = list(config.get("drawdown_exposure_tiers", []))
        for tier in sorted(tiers, key=lambda row: float(row["drawdown"])):
            if drawdown >= float(tier["drawdown"]):
                selected_target = min(
                    base_target,
                    float(tier["target_long_fraction"]),
                )
        return self._signal_governed_target(selected_target, config, long_green), drawdown

    def _signal_governed_target(
        self,
        target: float,
        config: dict[str, Any],
        long_green: int,
    ) -> float:
        if not bool(config.get("enable_signal_governor", False)):
            return target

        selected_target = 0.0
        tiers = list(config.get("signal_exposure_tiers", []))
        for tier in sorted(tiers, key=lambda row: int(row["green"])):
            if long_green >= int(tier["green"]):
                selected_target = float(tier["target_long_fraction"])
        return min(target, selected_target)

    def _target_short_fraction(
        self,
        config: dict[str, Any],
        long_green: int,
    ) -> float:
        target = float(config.get("target_short_fraction", 0.20))
        if not bool(config.get("enable_protective_short_hedge", False)):
            return target

        floors = dict(config.get("protective_hedge_floors", {}))
        floor = float(floors.get(long_green, 0.0))
        return max(target, floor)

    def _protective_short_symbol(
        self,
        config: dict[str, Any],
        selected_long: str,
        symbols: list[str],
        green_counts: dict[str, int] | None = None,
    ) -> str:
        configured_candidates = config.get("protective_short_symbols")
        if configured_candidates:
            candidates = [
                str(symbol)
                for symbol in configured_candidates
                if str(symbol) in symbols and str(symbol) != selected_long
            ]
            if not candidates:
                return ""
            if green_counts is None:
                return candidates[0]
            return min(candidates, key=lambda symbol: (green_counts.get(symbol, 0), symbol))

        configured = config.get("protective_short_symbol")
        if configured:
            return str(configured)
        return next((symbol for symbol in symbols if symbol != selected_long), "")

    def _protective_long_basis_symbol(
        self,
        snapshot: AccountSnapshot,
        selected_long: str,
        short_symbol: str,
        symbols: list[str],
    ) -> str:
        if selected_long:
            position = self._collateral(snapshot, selected_long)
            if position is not None and position.value() > 0.0:
                return selected_long

        candidates = [
            position
            for position in snapshot.collateral
            if position.symbol in symbols
            and position.symbol != short_symbol
            and position.price > 0.0
            and position.value() > 0.0
        ]
        if not candidates:
            return selected_long
        return max(candidates, key=lambda position: position.value()).symbol

    def _cover_excess_protective_short(
        self,
        snapshot: AccountSnapshot,
        selected_long: str,
        hedge_basis_symbol: str,
        protective_short_symbol: str,
        symbols: list[str],
        target_short_fraction: float,
        threshold_usd: float,
        fee: float,
    ) -> list[dict[str, Any]]:
        config = self._config
        if not bool(config.get("enable_protective_short_hedge", False)):
            return []
        short_symbol = protective_short_symbol or self._protective_short_symbol(config, selected_long, symbols)
        if not short_symbol:
            return []
        long_position = self._collateral(snapshot, hedge_basis_symbol)
        debt_position = self._debt(snapshot, short_symbol)
        usdc = self._collateral(snapshot, "USDC")
        if (
            long_position is None
            or debt_position is None
            or usdc is None
            or debt_position.price <= 0
        ):
            return []

        target_short_usd = long_position.value() * target_short_fraction
        current_short_usd = debt_position.value()
        excess_usd = current_short_usd - target_short_usd
        if excess_usd < threshold_usd or usdc.amount <= 0:
            return []

        repay_cost = min(excess_usd * (1.0 + fee), usdc.amount)
        repay_tokens = min(
            repay_cost / (1.0 + fee) / debt_position.price,
            debt_position.amount,
        )
        if repay_tokens <= 0:
            return []
        return [
            {
                "type": "withdraw_collateral",
                "symbol": "USDC",
                "amount": repay_tokens * debt_position.price * (1.0 + fee),
            },
            {
                "type": "repay",
                "symbol": short_symbol,
                "amount": repay_tokens,
            },
        ]

    def _rebalance_collateral(
        self,
        snapshot: AccountSnapshot,
        selected_long: str,
        symbols: list[str],
        target_long_usd: float,
        threshold_usd: float,
        fee: float,
    ) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        usdc = self._collateral(snapshot, "USDC")
        usdc_debt = self._debt(snapshot, "USDC")
        if usdc is None:
            return actions

        for position in snapshot.collateral:
            if position.symbol not in symbols or position.price <= 0:
                continue
            current_usd = position.value()
            target_usd = target_long_usd if position.symbol == selected_long else 0.0
            diff = target_usd - current_usd
            if abs(diff) < threshold_usd:
                continue
            if diff > 0 and usdc.amount > 0:
                spend_usdc = min(diff / max(1.0 - fee, 1e-9), usdc.amount)
                deposit_tokens = spend_usdc * (1.0 - fee) / position.price
                if spend_usdc > 0 and deposit_tokens > 0:
                    actions.extend(
                        [
                            {
                                "type": "withdraw_collateral",
                                "symbol": "USDC",
                                "amount": spend_usdc,
                            },
                            {
                                "type": "deposit_collateral",
                                "symbol": position.symbol,
                                "amount": deposit_tokens,
                            },
                        ]
                    )
                    usdc = CollateralPosition(
                        symbol=usdc.symbol,
                        amount=usdc.amount - spend_usdc,
                        price=usdc.price,
                        ltv=usdc.ltv,
                        liquidation_threshold=usdc.liquidation_threshold,
                    )
                remaining_usd = diff - (spend_usdc * (1.0 - fee))
                if remaining_usd > threshold_usd and usdc_debt is not None:
                    borrow_usdc = remaining_usd / max(1.0 - fee, 1e-9)
                    deposit_tokens = borrow_usdc * (1.0 - fee) / position.price
                    if borrow_usdc > 0 and deposit_tokens > 0:
                        actions.extend(
                            [
                                {
                                    "type": "borrow",
                                    "symbol": "USDC",
                                    "amount": borrow_usdc,
                                },
                                {
                                    "type": "deposit_collateral",
                                    "symbol": position.symbol,
                                    "amount": deposit_tokens,
                                },
                            ]
                        )
            elif diff > 0 and usdc_debt is not None:
                borrow_usdc = diff / max(1.0 - fee, 1e-9)
                deposit_tokens = borrow_usdc * (1.0 - fee) / position.price
                if borrow_usdc > 0 and deposit_tokens > 0:
                    actions.extend(
                        [
                            {
                                "type": "borrow",
                                "symbol": "USDC",
                                "amount": borrow_usdc,
                            },
                            {
                                "type": "deposit_collateral",
                                "symbol": position.symbol,
                                "amount": deposit_tokens,
                            },
                        ]
                    )
            elif diff < 0 and current_usd > 0:
                sell_usd = min(-diff, current_usd)
                sell_tokens = sell_usd / position.price
                usdc_proceeds = sell_usd * (1.0 - fee)
                if sell_tokens > 0 and usdc_proceeds > 0:
                    actions.append(
                        {
                            "type": "withdraw_collateral",
                            "symbol": position.symbol,
                            "amount": sell_tokens,
                        }
                    )
                    if usdc_debt is not None and usdc_debt.amount > 0:
                        repay_tokens = min(usdc_proceeds, usdc_debt.amount)
                        if repay_tokens > 0:
                            actions.append(
                                {
                                    "type": "repay",
                                    "symbol": "USDC",
                                    "amount": repay_tokens,
                                }
                            )
                        remainder = usdc_proceeds - repay_tokens
                    else:
                        remainder = usdc_proceeds
                    if remainder > 0:
                        actions.append(
                            {
                                "type": "deposit_collateral",
                                "symbol": "USDC",
                                "amount": remainder,
                            }
                        )
        return actions

    def _rebalance_debt(
        self,
        snapshot: AccountSnapshot,
        selected_short: str,
        hedge_basis_symbol: str,
        protective_short_symbol: str,
        symbols: list[str],
        target_short_usd: float,
        threshold_usd: float,
        fee: float,
    ) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        usdc = self._collateral(snapshot, "USDC")
        if usdc is None:
            return actions
        if bool(self._config.get("enable_protective_short_hedge", False)):
            short_symbol = protective_short_symbol or self._protective_short_symbol(
                self._config,
                selected_long=self._history_fields.get("selected_long", ""),
                symbols=symbols,
            )
            selected_short = short_symbol or selected_short

        for position in snapshot.debt:
            if position.symbol not in symbols or position.price <= 0:
                continue
            if (
                bool(self._config.get("enable_protective_short_hedge", False))
                and position.symbol != selected_short
            ):
                continue
            current_usd = position.value()
            if bool(self._config.get("enable_protective_short_hedge", False)):
                long_position = self._collateral(snapshot, hedge_basis_symbol)
                basis_usd = long_position.value() if long_position is not None else 0.0
                target_usd = basis_usd * target_short_usd
            else:
                target_usd = target_short_usd if position.symbol == selected_short else 0.0
            diff = target_usd - current_usd
            if abs(diff) < threshold_usd:
                continue
            if diff > 0:
                borrow_tokens = diff / position.price
                usdc_proceeds = diff * (1.0 - fee)
                if borrow_tokens > 0 and usdc_proceeds > 0:
                    actions.extend(
                        [
                            {
                                "type": "borrow",
                                "symbol": position.symbol,
                                "amount": borrow_tokens,
                            },
                            {
                                "type": "deposit_collateral",
                                "symbol": "USDC",
                                "amount": usdc_proceeds,
                            },
                        ]
                    )
            elif bool(self._config.get("enable_protective_short_hedge", False)):
                continue
            elif diff < 0 and current_usd > 0:
                repay_usd = min(-diff, current_usd)
                repay_cost = repay_usd * (1.0 + fee)
                if usdc.amount <= 0:
                    continue
                repay_cost = min(repay_cost, usdc.amount)
                repay_tokens = min(repay_cost / (1.0 + fee) / position.price, position.amount)
                if repay_tokens > 0:
                    actions.extend(
                        [
                            {
                                "type": "withdraw_collateral",
                                "symbol": "USDC",
                                "amount": repay_tokens * position.price * (1.0 + fee),
                            },
                            {
                                "type": "repay",
                                "symbol": position.symbol,
                                "amount": repay_tokens,
                            },
                        ]
                    )
        return actions

    @staticmethod
    def _collateral(snapshot: AccountSnapshot, symbol: str) -> CollateralPosition | None:
        return next((position for position in snapshot.collateral if position.symbol == symbol), None)

    @staticmethod
    def _debt(snapshot: AccountSnapshot, symbol: str) -> DebtPosition | None:
        return next((position for position in snapshot.debt if position.symbol == symbol), None)
