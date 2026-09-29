"""Bounded queries over validated all-wallet fill Parquet, without wallet sampling.

Input partitions must come from the archive parser. This layer does not establish
archive coverage: acquisition manifests must do that before dataset registration.
DuckDB owns sorting/deduplication under explicit memory and temporary-disk limits;
Python retains only one bounded wallet history plus bounded ranking output.
"""

from dataclasses import fields
from datetime import timedelta
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb

from .contracts import FillEvent, address, symbol, utc, finite
from .lab_ranking import rank_activity


ORDER = "exchange_time, coalesce(block_number, -1), source_key, source_line, event_index, event_id"
IDENTITY = "user, coin, exchange_time, tid, oid, side"
ECONOMICS = "exchange_time, px, sz, start_position, post_position, direction, closed_pnl, fee, fee_token, crossed, liquidation, tx_hash"
REVERSE_ORDER = "exchange_time DESC, coalesce(block_number, -1) DESC, source_key DESC, source_line DESC, event_index DESC, event_id DESC"


class ProxyActivity:
    def __init__(
        self,
        paths,
        *,
        temp_root,
        max_input_bytes=8 * 1024**3,
        memory_mb=256,
        temp_disk_mb=2048,
        max_wallet_fills=100_000,
        max_candidates=100_000,
        query_window=None,
        seed_path=None,
        seed_cutoff=None,
    ):
        self.query_window = None
        if query_window is not None:
            start, end = (utc(value) for value in query_window)
            if start >= end:
                raise ValueError("Increasing query window required")
            self.query_window = (start, end)
        if (seed_path is None) != (seed_cutoff is None):
            raise ValueError("Checkpoint seed path and cutoff are required together")
        if seed_path is not None:
            seed_cutoff = utc(seed_cutoff)
            if self.query_window is None or self.query_window[0] < seed_cutoff:
                raise ValueError("Query window must start at or after seed cutoff")
            seed_path = Path(seed_path).resolve(strict=True)
        limits = (
            max_input_bytes,
            memory_mb,
            temp_disk_mb,
            max_wallet_fills,
            max_candidates,
        )
        if any(type(value) is not int or value <= 0 for value in limits):
            raise ValueError("positive integer activity bounds required")
        paths = [Path(p).resolve(strict=True) for p in paths]
        all_paths = paths + ([seed_path] if seed_path is not None else [])
        if (
            not all_paths
            or len(all_paths) > 5000
            or len(set(all_paths)) != len(all_paths)
        ):
            raise ValueError("require 1..5000 distinct fill partitions")
        if sum(p.stat().st_size for p in all_paths) > max_input_bytes:
            raise ValueError("activity input byte limit exceeded")
        self.max_wallet_fills = max_wallet_fills
        self.max_candidates = max_candidates
        self._temp = TemporaryDirectory(prefix="proxy_activity_", dir=temp_root)
        self.db = None
        try:
            self.db = duckdb.connect(
                config={
                    "memory_limit": f"{memory_mb}MB",
                    "max_temp_directory_size": f"{temp_disk_mb}MB",
                    "temp_directory": self._temp.name,
                    "threads": 1,
                    "TimeZone": "UTC",
                    "preserve_insertion_order": False,
                }
            )
            if seed_path is None:
                self.db.read_parquet(
                    [str(p) for p in paths], union_by_name=True, hive_partitioning=False
                ).create_view("source_fills")
            else:
                self.db.read_parquet(
                    str(seed_path), hive_partitioning=False
                ).create_view("checkpoint_seeds")
                if self.db.execute(
                    "SELECT 1 FROM checkpoint_seeds WHERE exchange_time IS NULL OR exchange_time >= ? LIMIT 1",
                    [seed_cutoff],
                ).fetchone():
                    raise ValueError("Checkpoint seeds must be strictly before cutoff")
                if paths:
                    self.db.read_parquet(
                        [str(p) for p in paths],
                        union_by_name=True,
                        hive_partitioning=False,
                    ).create_view("replay_fills")
                    # Filter before union/dedup: overlapping files must never
                    # resurrect noncanonical provenance for an expired event.
                    self.db.execute(
                        "CREATE TEMP VIEW source_fills AS SELECT * FROM checkpoint_seeds UNION ALL BY NAME "
                        f"SELECT * FROM replay_fills WHERE exchange_time >= TIMESTAMPTZ '{seed_cutoff.isoformat()}'"
                    )
                else:
                    self.db.execute(
                        "CREATE TEMP VIEW source_fills AS SELECT * FROM checkpoint_seeds"
                    )
            names = {
                row[0] for row in self.db.execute("DESCRIBE source_fills").fetchall()
            }
            expected = {field.name for field in fields(FillEvent)}
            if expected - names - {"raw_details_json"}:
                raise ValueError("missing canonical fill columns")
            raw = (
                "*"
                if "raw_details_json" in names
                else "*, NULL::VARCHAR AS raw_details_json"
            )
            self.db.execute(
                f"CREATE TEMP VIEW canonical AS SELECT {raw} FROM source_fills"
            )
            # tid is a hash, not a globally unique trade ID. The raw fill's
            # millisecond time disambiguates trades across both archive formats.
            # Nullable legacy envelope columns may have Arrow's null type.
            if self.db.execute(
                "SELECT 1 FROM (SELECT *, "
                "try_cast(exchange_time AS TIMESTAMPTZ) AS fill_time, "
                "try_cast(block_time AS TIMESTAMPTZ) AS envelope_time FROM canonical) "
                "WHERE fill_time IS NULL "
                "OR fill_time != date_trunc('milliseconds', fill_time) "
                "OR (starts_with(source_key, 'node_fills_by_block/') AND block_time IS NULL) "
                "OR (block_time IS NOT NULL AND (envelope_time IS NULL "
                "OR date_trunc('milliseconds', envelope_time) != fill_time)) LIMIT 1"
            ).fetchone():
                raise ValueError("invalid canonical trade timestamp")
            conflict = self.db.execute(
                f"SELECT 1 FROM canonical GROUP BY {IDENTITY} "
                f"HAVING count(DISTINCT ({ECONOMICS})) > 1 LIMIT 1"
            ).fetchone()
            if conflict:
                raise ValueError("conflicting economic fill duplicates")
            if self.db.execute(
                "SELECT 1 FROM canonical GROUP BY coin, exchange_time, tid "
                "HAVING count(DISTINCT (px, sz)) > 1 LIMIT 1"
            ).fetchone():
                raise ValueError("inconsistent market trade counterparties")
            # Native identity deduplicates archive overlaps even when provenance
            # changes event_id. Preserve both counterparties as separate fills.
            deduplicated = (
                f"SELECT * FROM canonical "
                f"QUALIFY row_number() OVER (PARTITION BY {IDENTITY} ORDER BY {ORDER}) = 1"
            )
            if self.query_window is None:
                self.db.execute(f"CREATE TEMP TABLE fills AS {deduplicated}")
                self.seed_count = 0
            else:
                start, end = self.query_window
                # No synthetic timestamps or position-only fake fills. Old seeds
                # retain their native order/economics but are outside every
                # permitted ranking/volume lookback. Full conflict checks above
                # still cover the entire supplied frozen prefix.
                self.db.execute(
                    f"CREATE TEMP TABLE fills AS WITH deduplicated AS ({deduplicated}), "
                    f"seeds AS (SELECT * FROM deduplicated WHERE exchange_time < ? "
                    f"QUALIFY row_number() OVER (PARTITION BY user, coin ORDER BY {REVERSE_ORDER}) = 1) "
                    f"SELECT * FROM seeds UNION ALL SELECT * FROM deduplicated "
                    f"WHERE exchange_time >= ? AND exchange_time < ?",
                    [start, start, end],
                )
                self.seed_count = self.db.execute(
                    "SELECT count(*) FROM fills WHERE exchange_time < ?", [start]
                ).fetchone()[0]
            self.count = self.db.execute("SELECT count(*) FROM fills").fetchone()[0]
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        self._temp.cleanup()

    def observed(self, decision):
        self._check_window(decision, decision)
        rows = self.db.execute(
            "SELECT DISTINCT coin FROM fills WHERE exchange_time < ? ORDER BY coin",
            [utc(decision)],
        ).fetchall()
        return [symbol(row[0]) for row in rows]

    def validate_registered_scope(self, coins, start, end):
        """Check canonical economic rows on disk before admitting registered data."""
        if self.query_window is not None:
            raise ValueError("Windowed activity cannot qualify full registered scope")
        schema = {
            row[0]: row[1] for row in self.db.execute("DESCRIBE fills").fetchall()
        }
        integers = {
            "TINYINT",
            "SMALLINT",
            "INTEGER",
            "BIGINT",
            "UTINYINT",
            "USMALLINT",
            "UINTEGER",
            "UBIGINT",
        }
        if any(
            schema[name] not in integers
            for name in ("tid", "oid", "source_line", "event_index")
        ):
            raise ValueError("invalid registered fill identity types")
        invalid = self.db.execute(
            """
            SELECT 1 FROM fills WHERE NOT coalesce(
                coin IN (SELECT unnest(?)) AND exchange_time >= ? AND exchange_time < ?
                AND tid IS NOT NULL AND oid IS NOT NULL
                AND source_line >= 0 AND event_index >= 0 AND crossed IS NOT NULL
                AND regexp_full_match(user, '0x[0-9a-f]{40}')
                AND isfinite(px) AND px > 0 AND isfinite(sz) AND sz > 0
                AND isfinite(start_position) AND isfinite(post_position)
                AND isfinite(closed_pnl) AND isfinite(fee) AND side IN ('A', 'B')
                AND abs(post_position - (start_position + CASE WHEN side = 'B' THEN sz ELSE -sz END))
                    <= greatest(1e-8, abs(post_position)*1e-9), false)
            LIMIT 1
        """,
            [list(coins), utc(start), utc(end)],
        ).fetchone()
        if invalid:
            raise ValueError("invalid registered fill scope or economic values")

    def position(self, user, coin, decision):
        self._check_window(decision, decision)
        # Reverse the complete tie-break order, not just the exchange timestamp.
        reverse = ", ".join(
            f"{part} DESC"
            for part in (
                "exchange_time",
                "coalesce(block_number, -1)",
                "source_key",
                "source_line",
                "event_index",
                "event_id",
            )
        )
        row = self.db.execute(
            f"SELECT post_position FROM fills WHERE user = ? AND coin = ? "
            f"AND exchange_time < ? ORDER BY {reverse} LIMIT 1",
            [address(user), symbol(coin), utc(decision)],
        ).fetchone()
        return row[0] if row else None

    def volume(self, coin, start, decision):
        start, decision = utc(start), utc(decision)
        self._check_window(start, decision)
        if start >= decision:
            raise ValueError("positive volume window required")
        # Aggregate unique market trades, not the two wallet fill sides.
        return self.db.execute(
            "SELECT coalesce(sum(notional), 0) FROM ("
            "SELECT exchange_time, tid, max(px * sz) AS notional FROM fills "
            "WHERE coin = ? AND exchange_time >= ? AND exchange_time < ? GROUP BY exchange_time, tid)",
            [symbol(coin), start, decision],
        ).fetchone()[0]

    def hourly_exposure(self, user, coin, start, end, *, max_price_age_seconds):
        """Native quantity × last native trade price, strictly before each sample.

        This is a bounded normalization proxy, not an exchange mark or equity.
        Missing position/price stays unknown; never infer an earlier position.
        """
        start, end = utc(start), utc(end)
        self._check_window(start, end)
        count = (end - start).total_seconds() / 3600
        age = finite(max_price_age_seconds)
        if not 0 < count <= 100_000 or count != int(count) or age <= 0:
            raise ValueError("invalid or oversized hourly exposure request")
        reverse = "coalesce(block_number,-1) DESC, source_key DESC, source_line DESC, event_index DESC, event_id DESC"
        query = self.db.execute(
            f"""
            WITH positions AS (
                SELECT exchange_time, post_position FROM fills
                WHERE user = ? AND coin = ? AND exchange_time < ?
                QUALIFY row_number() OVER (PARTITION BY exchange_time ORDER BY {reverse}) = 1
            ), prices AS (
                SELECT exchange_time, px FROM fills WHERE coin = ? AND exchange_time < ?
                QUALIFY row_number() OVER (PARTITION BY exchange_time ORDER BY {reverse}) = 1
            ), samples AS (
                SELECT generate_series AS time FROM generate_series(?, ?, INTERVAL '1 hour')
            )
            SELECT s.time, p.post_position, m.px, m.exchange_time FROM samples s
            ASOF LEFT JOIN positions p ON s.time > p.exchange_time
            ASOF LEFT JOIN prices m ON s.time > m.exchange_time ORDER BY s.time
        """,
            [
                address(user),
                symbol(coin),
                end,
                coin,
                end,
                start,
                end - timedelta(hours=1),
            ],
        )
        return [
            (
                at,
                None
                if qty is None
                or price_time is None
                or (at - price_time).total_seconds() > age
                else qty * price,
            )
            for at, qty, price, price_time in query.fetchall()
        ]

    def rank(self, decision, config, scope, semantics, *, smoke=False):
        decision = utc(decision)
        self._check_window(decision - timedelta(days=config.lookback_days), decision)
        coins = [coin for coin in config.coins if scope is None or coin == scope]
        users = self.db.execute(
            "SELECT DISTINCT user FROM fills WHERE coin IN (SELECT unnest(?)) "
            "AND exchange_time < ? ORDER BY user LIMIT ?",
            [coins, decision, self.max_candidates + 1],
        ).fetchall()
        if len(users) > self.max_candidates:
            raise ValueError("ranking candidate limit exceeded; no wallet sampling")

        columns = [
            field.name
            for field in fields(FillEvent)
            if field.name != "raw_details_json"
        ]
        query = self.db.execute(
            f"SELECT {', '.join(columns)} FROM fills WHERE coin IN (SELECT unnest(?)) "
            f"AND exchange_time >= ? AND exchange_time < ? ORDER BY user, {ORDER}",
            [coins, decision - timedelta(days=config.lookback_days), decision],
        )

        def stream():
            while batch := query.fetchmany(1024):
                yield from batch

        def groups():
            # One ordered scan, not one scan of the archive per candidate wallet.
            active = iter(groupby(stream(), key=lambda row: row[columns.index("user")]))
            current = next(active, None)
            for (user,) in users:
                history = []
                if current is not None and current[0] == user:
                    for row in current[1]:
                        if len(history) >= self.max_wallet_fills:
                            raise ValueError(
                                "ranking wallet history limit exceeded; no truncation"
                            )
                        history.append(FillEvent(**dict(zip(columns, row))))
                    current = next(active, None)
                yield user, history

        return rank_activity(groups(), decision, config, scope, semantics, smoke=smoke)

    def _check_window(self, start, end):
        if self.query_window is not None:
            low, high = self.query_window
            if not low <= utc(start) <= utc(end) <= high:
                raise ValueError("Query requires history outside the active window")
