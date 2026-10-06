"""Kamino adapter: existing holdings loader plus protocol-recorded risk totals."""

from arblab.kamino_onchain import (
    _get_account,
    _load_idl,
    _decode_account,
    _sf_to_decimal,
)
from arblab.paths import fixture_path


def recorded_risk(obligation):
    collateral = float(_sf_to_decimal(obligation.deposited_value_sf))
    adjusted_debt = float(
        _sf_to_decimal(obligation.borrow_factor_adjusted_debt_value_sf)
    )
    debt = float(_sf_to_decimal(obligation.borrowed_assets_market_value_sf))
    capacity = float(_sf_to_decimal(obligation.allowed_borrow_value_sf))
    threshold = float(_sf_to_decimal(obligation.unhealthy_borrow_value_sf))
    return {
        "supplied": collateral,
        "debt": debt,
        "ltv": adjusted_debt / collateral if collateral else None,
        "liquidationLtv": threshold / collateral if collateral else 0,
        "health": threshold / adjusted_debt if adjusted_debt else None,
        "borrowHealth": capacity / adjusted_debt if adjusted_debt else None,
        "liquidationBuffer": threshold - adjusted_debt,
        "elevationGroup": int(obligation.elevation_group),
    }


def load_recorded_risk(address, program, rpc_url):
    account = _get_account(rpc_url, address)
    if account["owner"] != program:
        raise ValueError("Unexpected obligation owner.")
    idl = _load_idl(str(fixture_path("kamino_idl.json")))
    return recorded_risk(_decode_account(idl, "Obligation", account["data"]))
