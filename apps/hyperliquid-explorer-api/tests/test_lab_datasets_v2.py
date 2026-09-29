def test_cross_class_dataset_registration_and_bounded_catalogue(tmp_path):
    from arblab.hyperliquid_copy.lab_fixture_v2 import write_fixture
    from arblab.hyperliquid_copy.lab_config_codec import parse_lab_config
    from hyperliquid_explorer_api.lab_datasets import DatasetCatalog

    write_fixture(tmp_path / "datasets" / "cross")
    catalog = DatasetCatalog(tmp_path)
    info = catalog.list()[0]
    assert set(info["supported_classes"]) == {"crypto", "commodity", "equity", "index"}
    assert info["liquidity_available"] is True
    config = parse_lab_config(info["default_config"])
    assert catalog.inspect("cross", config)["ready"]
    frozen = catalog.preflight("cross", config)
    loaded = catalog.load("cross", config, frozen)
    assert len(loaded.catalogue.by_id) == 7
    assert loaded.volume is not None
    assert (
        loaded.market.mark(
            "demo:GOLD",
            __import__("arblab.hyperliquid_copy.lab_config", fromlist=["day"]).day(
                config.start
            ),
        )
        > 0
    )


def test_liquidity_history_estimate_includes_excluded_classes(tmp_path):
    from dataclasses import replace
    from arblab.hyperliquid_copy.lab_fixture_v2 import write_fixture, fixture_rows
    from hyperliquid_explorer_api.lab_datasets import DatasetCatalog

    write_fixture(tmp_path / "datasets" / "cross")
    config = fixture_rows()[-2]
    config = replace(
        config,
        market_universe=replace(
            config.market_universe, general=False, classes=["equity"]
        ),
    )
    result = DatasetCatalog(tmp_path).inspect("cross", config)
    assert (
        result["estimates"]["market_history_rows"] == 14
    )  # all seven IDs, two daily decisions
