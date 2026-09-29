import json


def test_numeric_strings_preserve_metric_values(report_root):
    from hyperliquid_explorer_api.repository import Repository

    root, run = report_root
    path = root / run / "summary.json"
    summary = json.loads(path.read_text())
    summary["scenarios"][0]["final_equity"] = "123.45"
    path.write_text(json.dumps(summary))
    assert Repository(root).detail(run).scenarios[0].metrics["final_equity"] == 123.45


def test_query_dates_are_utc_even_at_midnight_boundary():
    from hyperliquid_explorer_api.queries import connection

    with connection() as db:
        assert (
            str(
                db.execute(
                    "SELECT CAST(TIMESTAMPTZ '2026-07-01 23:30:00+00' AS DATE)"
                ).fetchone()[0]
            )
            == "2026-07-01"
        )
