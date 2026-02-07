"""Kamino Lending liquidation risk simulator - Streamlit UI."""

import os
import streamlit as st

from arblab.kamino_onchain import (
    _fetch_jupiter_symbols,
    find_wallet_obligations,
    get_obligation_market,
    load_market_reserves,
    load_onchain_snapshot,
    ReserveConfig,
)
from arblab.kamino_risk import (
    AccountSnapshot,
    CollateralPosition,
    DebtPosition,
    apply_actions,
    liquidation_prices,
    scenario_report,
)

PROGRAM_ID = "KLend2g3cP87fffoy8q1mQqGKjrxjC8boSyAYavgmjD"
RPC_URL = "https://api.mainnet-beta.solana.com"
IDL_PATH = os.path.join(os.path.dirname(__file__), "kamino_idl.json")

st.set_page_config(page_title="Kamino Risk Simulator", layout="wide")

# Pre-warm the Jupiter token symbol cache so reserve names are resolved quickly.
_fetch_jupiter_symbols()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _health_color(hf: float) -> str:
    if hf == float("inf"):
        return "green"
    if hf >= 1.5:
        return "green"
    if hf >= 1.1:
        return "orange"
    return "red"


def _fmt(value: float, decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}"


def _fmt_usd(value: float) -> str:
    return f"${value:,.2f}"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _load_snapshot(obligation_address: str) -> AccountSnapshot:
    return load_onchain_snapshot(
        obligation_address=obligation_address,
        program_id=PROGRAM_ID,
        rpc_url=RPC_URL,
        idl_path=IDL_PATH,
    )


def _load_market_reserves(obligation_address: str) -> list[ReserveConfig]:
    market = get_obligation_market(obligation_address, PROGRAM_ID, RPC_URL, IDL_PATH)
    return load_market_reserves(market, PROGRAM_ID, RPC_URL, IDL_PATH)


# ---------------------------------------------------------------------------
# Sidebar - Position Loading
# ---------------------------------------------------------------------------

st.sidebar.title("Kamino Risk Sim")

load_mode = st.sidebar.radio(
    "Load by",
    ["Wallet address", "Obligation address"],
    horizontal=True,
)

if load_mode == "Wallet address":
    wallet = st.sidebar.text_input(
        "Wallet address",
        value="F8ir9FxMgi17DpnLDbkM6mxy5GmS1o8ynmtP73HuHzQL",
    )
    if st.sidebar.button("Load positions"):
        with st.sidebar.status("Fetching obligations...", expanded=True) as status:
            try:
                obligations = find_wallet_obligations(wallet, PROGRAM_ID, RPC_URL)
                if not obligations:
                    st.sidebar.error("No Kamino obligations found for this wallet.")
                else:
                    st.session_state["obligations"] = obligations
                    # Auto-load the first obligation
                    st.session_state["selected_obligation"] = obligations[0]
                    status.update(label="Loading obligation data...", state="running")
                    st.session_state["snapshot"] = _load_snapshot(obligations[0])
                    status.update(label="Loading market reserves...", state="running")
                    st.session_state["market_reserves"] = _load_market_reserves(obligations[0])
                    status.update(label=f"Found {len(obligations)} obligation(s)", state="complete")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

    obligations = st.session_state.get("obligations", [])
    if len(obligations) > 1:
        selected = st.sidebar.selectbox(
            "Select obligation",
            obligations,
            index=obligations.index(st.session_state.get("selected_obligation", obligations[0])),
        )
        if selected != st.session_state.get("selected_obligation"):
            with st.sidebar.status("Loading obligation..."):
                st.session_state["selected_obligation"] = selected
                st.session_state["snapshot"] = _load_snapshot(selected)
                st.session_state["market_reserves"] = _load_market_reserves(selected)

else:
    obligation_addr = st.sidebar.text_input("Obligation address", value="")
    if st.sidebar.button("Load obligation"):
        if obligation_addr:
            with st.sidebar.status("Loading obligation data...") as status:
                try:
                    st.session_state["snapshot"] = _load_snapshot(obligation_addr)
                    st.session_state["selected_obligation"] = obligation_addr
                    st.session_state["obligations"] = [obligation_addr]
                    status.update(label="Loading market reserves...", state="running")
                    st.session_state["market_reserves"] = _load_market_reserves(obligation_addr)
                    status.update(label="Loaded", state="complete")
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")

# ---------------------------------------------------------------------------
# Main content - require a loaded snapshot
# ---------------------------------------------------------------------------

snapshot: AccountSnapshot | None = st.session_state.get("snapshot")

if snapshot is None:
    st.title("Kamino Liquidation Risk Simulator")
    st.info("Enter a wallet or obligation address in the sidebar to get started.")
    st.stop()

st.title("Kamino Liquidation Risk Simulator")

obligation_label = st.session_state.get("selected_obligation", "")
if obligation_label:
    st.caption(f"Obligation: `{obligation_label}`")


# ---------------------------------------------------------------------------
# Section 1 - Current Position Overview
# ---------------------------------------------------------------------------

st.header("Position Overview")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Collateral")
    rows = []
    for p in snapshot.collateral:
        rows.append({
            "Asset": p.symbol,
            "Amount": _fmt(p.amount, 4),
            "Price": _fmt_usd(p.price),
            "Value": _fmt_usd(p.value()),
            "LTV": _pct(p.ltv),
            "Liq. Threshold": _pct(p.liquidation_threshold),
        })
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.markdown(f"**Total collateral: {_fmt_usd(snapshot.total_collateral_value())}**")
    else:
        st.write("No collateral positions.")

with col_right:
    st.subheader("Debt")
    rows = []
    for p in snapshot.debt:
        rows.append({
            "Asset": p.symbol,
            "Amount": _fmt(p.amount, 4),
            "Price": _fmt_usd(p.price),
            "Value": _fmt_usd(p.value()),
        })
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.markdown(f"**Total debt: {_fmt_usd(snapshot.total_debt_value())}**")
    else:
        st.write("No debt positions.")

# ---------------------------------------------------------------------------
# Section 2 - Risk Metrics
# ---------------------------------------------------------------------------

st.header("Risk Metrics")

report = scenario_report(snapshot)
hf = report["health_factor"]
hf_color = _health_color(hf)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Health Factor", _fmt(hf) if hf != float("inf") else "Safe (no debt)")
m2.metric("Current LTV", _pct(report["current_ltv"]) if report["current_ltv"] != float("inf") else "N/A")
m3.metric("Borrow Limit", _fmt_usd(report["borrow_limit"]))
m4.metric("Liquidation Buffer", _fmt_usd(report["liquidation_buffer"]))

if hf != float("inf"):
    if hf < 1.1:
        st.error(f"Health factor is critically low ({_fmt(hf)}). Liquidation risk is high.")
    elif hf < 1.5:
        st.warning(f"Health factor is moderate ({_fmt(hf)}). Monitor closely.")
    else:
        st.success(f"Health factor is healthy ({_fmt(hf)}).")

# Liquidation prices
liq_prices = liquidation_prices(snapshot)
col_lp1, col_lp2 = st.columns(2)
with col_lp1:
    st.subheader("Collateral Liquidation Prices")
    for symbol, price in liq_prices["collateral"].items():
        if price is not None:
            current = next((p.price for p in snapshot.collateral if p.symbol == symbol), 0)
            drop = ((current - price) / current * 100) if current > 0 else 0
            st.markdown(f"**{symbol}**: {_fmt_usd(price)} (current: {_fmt_usd(current)}, -{_fmt(drop, 1)}% to liq)")
with col_lp2:
    st.subheader("Debt Liquidation Prices")
    for symbol, price in liq_prices["debt"].items():
        if price is not None:
            current = next((p.price for p in snapshot.debt if p.symbol == symbol), 0)
            rise = ((price - current) / current * 100) if current > 0 else 0
            st.markdown(f"**{symbol}**: {_fmt_usd(price)} (current: {_fmt_usd(current)}, +{_fmt(rise, 1)}% to liq)")


# ---------------------------------------------------------------------------
# Section 3 - Scenario Simulator
# ---------------------------------------------------------------------------

st.header("Scenario Simulator")

actions = []
new_collateral: list[CollateralPosition] = []
new_debt: list[DebtPosition] = []

# Available market reserves (loaded alongside the snapshot)
market_reserves: list[ReserveConfig] = st.session_state.get("market_reserves", [])

# Unique asset symbols across collateral and debt
all_symbols = list({p.symbol for p in snapshot.collateral} | {p.symbol for p in snapshot.debt})
all_symbols.sort()

# Price adjustments
with st.expander("Price Changes", expanded=True):
    price_cols = st.columns(min(len(all_symbols), 4))
    for i, symbol in enumerate(all_symbols):
        col = price_cols[i % len(price_cols)]
        current = next(
            (p.price for p in snapshot.collateral if p.symbol == symbol),
            next((p.price for p in snapshot.debt if p.symbol == symbol), 0),
        )
        if current > 0:
            new_price = col.slider(
                f"{symbol} price",
                min_value=current * 0.01,
                max_value=current * 3.0,
                value=current,
                step=current * 0.01,
                format=f"$%.{'2' if current >= 1 else '6'}f",
                key=f"price_{symbol}",
            )
            if abs(new_price - current) > current * 0.001:
                actions.append({"type": "set_price", "symbol": symbol, "price": new_price})

# Collateral adjustments
with st.expander("Collateral Adjustments"):
    for p in snapshot.collateral:
        col_a, col_b = st.columns([3, 1])
        delta = col_a.slider(
            f"{p.symbol} collateral change",
            min_value=-p.amount,
            max_value=p.amount * 2.0,
            value=0.0,
            step=max(p.amount * 0.01, 0.001),
            key=f"coll_{p.symbol}",
        )
        col_b.caption(f"Current: {_fmt(p.amount, 4)}")
        if delta > 0:
            actions.append({"type": "deposit_collateral", "symbol": p.symbol, "amount": delta})
        elif delta < 0:
            actions.append({"type": "withdraw_collateral", "symbol": p.symbol, "amount": abs(delta)})

# Debt adjustments
with st.expander("Debt Adjustments"):
    for p in snapshot.debt:
        col_a, col_b = st.columns([3, 1])
        delta = col_a.slider(
            f"{p.symbol} debt change",
            min_value=-p.amount,
            max_value=p.amount * 2.0,
            value=0.0,
            step=max(p.amount * 0.01, 0.01),
            key=f"debt_{p.symbol}",
        )
        col_b.caption(f"Current: {_fmt(p.amount, 4)}")
        if delta > 0:
            actions.append({"type": "borrow", "symbol": p.symbol, "amount": delta})
        elif delta < 0:
            actions.append({"type": "repay", "symbol": p.symbol, "amount": abs(delta)})

# Add new collateral / debt from available Kamino reserves
if market_reserves:
    existing_collateral_symbols = {p.symbol for p in snapshot.collateral}
    existing_debt_symbols = {p.symbol for p in snapshot.debt}

    # Filter to reserves with LTV > 0 for collateral (usable as collateral)
    available_collateral = [
        r for r in market_reserves
        if r.symbol not in existing_collateral_symbols and float(r.ltv) > 0
    ]
    # All reserves with a price can potentially be borrowed
    available_debt = [
        r for r in market_reserves
        if r.symbol not in existing_debt_symbols and float(r.price) > 0
    ]

    collateral_by_symbol = {r.symbol: r for r in available_collateral}
    debt_by_symbol = {r.symbol: r for r in available_debt}

    with st.expander("Add New Collateral"):
        if not available_collateral:
            st.caption("No additional collateral assets available.")
        else:
            selected_coll = st.multiselect(
                "Select assets to deposit as collateral",
                options=[r.symbol for r in available_collateral],
                format_func=lambda s: f"{s} (${float(collateral_by_symbol[s].price):,.4f}, LTV {float(collateral_by_symbol[s].ltv):.0%})",
                key="new_coll_select",
            )
            for symbol in selected_coll:
                r = collateral_by_symbol[symbol]
                col_info, col_amt = st.columns([2, 2])
                col_info.caption(
                    f"**{r.symbol}** -- Price: ${float(r.price):,.4f} | "
                    f"LTV: {float(r.ltv):.0%} | Liq: {float(r.liquidation_threshold):.0%}"
                )
                amount = col_amt.number_input(
                    f"Amount of {r.symbol}",
                    min_value=0.0,
                    value=0.0,
                    step=10.0 ** (-r.decimals) * 1000,
                    format=f"%.{min(r.decimals, 6)}f",
                    key=f"new_coll_{r.symbol}",
                )
                if amount > 0:
                    new_collateral.append(CollateralPosition(
                        symbol=r.symbol,
                        amount=amount,
                        price=float(r.price),
                        ltv=float(r.ltv),
                        liquidation_threshold=float(r.liquidation_threshold),
                    ))

    with st.expander("Take New Debt"):
        if not available_debt:
            st.caption("No additional debt assets available.")
        else:
            selected_debt = st.multiselect(
                "Select assets to borrow",
                options=[r.symbol for r in available_debt],
                format_func=lambda s: f"{s} (${float(debt_by_symbol[s].price):,.4f})",
                key="new_debt_select",
            )
            for symbol in selected_debt:
                r = debt_by_symbol[symbol]
                col_info, col_amt = st.columns([2, 2])
                col_info.caption(f"**{r.symbol}** -- Price: ${float(r.price):,.4f}")
                amount = col_amt.number_input(
                    f"Borrow amount of {r.symbol}",
                    min_value=0.0,
                    value=0.0,
                    step=10.0 ** (-r.decimals) * 1000,
                    format=f"%.{min(r.decimals, 6)}f",
                    key=f"new_debt_{r.symbol}",
                )
                if amount > 0:
                    new_debt.append(DebtPosition(
                        symbol=r.symbol,
                        amount=amount,
                        price=float(r.price),
                    ))

# Build the simulation snapshot: start from baseline, add new positions, then apply actions
has_changes = bool(actions) or bool(new_collateral) or bool(new_debt)

if has_changes:
    try:
        # Start from baseline snapshot, add new positions, then apply actions
        sim_collateral = list(snapshot.collateral) + new_collateral
        sim_debt = list(snapshot.debt) + new_debt
        augmented = AccountSnapshot(collateral=sim_collateral, debt=sim_debt)
        sim_snapshot = apply_actions(augmented, actions) if actions else augmented
    except ValueError as e:
        st.error(f"Invalid scenario: {e}")
        st.stop()

    sim_report = scenario_report(sim_snapshot)
    sim_liq = liquidation_prices(sim_snapshot)
    sim_hf = sim_report["health_factor"]

    st.subheader("Scenario Results")

    s1, s2, s3, s4 = st.columns(4)

    def _delta_str(new: float, old: float) -> str:
        diff = new - old
        if abs(diff) < 0.0001:
            return ""
        return f"{diff:+.2f}"

    hf_delta = _delta_str(sim_hf, hf) if hf != float("inf") and sim_hf != float("inf") else ""
    s1.metric(
        "Health Factor",
        _fmt(sim_hf) if sim_hf != float("inf") else "Safe",
        delta=hf_delta or None,
        delta_color="normal",
    )
    ltv_delta = _delta_str(sim_report["current_ltv"], report["current_ltv"]) if report["current_ltv"] != float("inf") else ""
    s2.metric(
        "Current LTV",
        _pct(sim_report["current_ltv"]) if sim_report["current_ltv"] != float("inf") else "N/A",
        delta=ltv_delta or None,
        delta_color="inverse",
    )
    s3.metric(
        "Borrow Limit",
        _fmt_usd(sim_report["borrow_limit"]),
        delta=_delta_str(sim_report["borrow_limit"], report["borrow_limit"]) or None,
    )
    s4.metric(
        "Liq. Buffer",
        _fmt_usd(sim_report["liquidation_buffer"]),
        delta=_delta_str(sim_report["liquidation_buffer"], report["liquidation_buffer"]) or None,
    )

    if sim_hf != float("inf"):
        if sim_hf < 1.0:
            st.error("LIQUIDATION! Health factor is below 1.0 in this scenario.")
        elif sim_hf < 1.1:
            st.error(f"Health factor is critically low ({_fmt(sim_hf)}).")
        elif sim_hf < 1.5:
            st.warning(f"Health factor is moderate ({_fmt(sim_hf)}).")
        else:
            st.success(f"Health factor is healthy ({_fmt(sim_hf)}).")

    # Show updated liquidation prices
    lp1, lp2 = st.columns(2)
    with lp1:
        st.markdown("**Updated Collateral Liquidation Prices**")
        for symbol, price in sim_liq["collateral"].items():
            if price is not None:
                orig = liq_prices["collateral"].get(symbol)
                delta = f" (was {_fmt_usd(orig)})" if orig and abs(price - orig) > 0.01 else ""
                st.markdown(f"- {symbol}: {_fmt_usd(price)}{delta}")
    with lp2:
        st.markdown("**Updated Debt Liquidation Prices**")
        for symbol, price in sim_liq["debt"].items():
            if price is not None:
                orig = liq_prices["debt"].get(symbol)
                delta = f" (was {_fmt_usd(orig)})" if orig and abs(price - orig) > 0.01 else ""
                st.markdown(f"- {symbol}: {_fmt_usd(price)}{delta}")
else:
    st.info("Adjust the sliders above to simulate scenarios.")
