"""Bounded capacity artifact decoding shared by staging and retained readers."""

from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

MAX_DECODED_BYTES = 64 * 1024**2
SCHEMA = pa.schema(
    [
        pa.field("day", pa.date32()),
        pa.field("coin", pa.string()),
        pa.field("entrants", pa.int64()),
    ]
)


def read_counts(path, source, *, max_bytes, max_rows):
    if path.stat().st_size > max_bytes:
        raise ValueError("Candidate capacity artifact byte limit")
    with pq.ParquetFile(path, read_dictionary=["coin"]) as reader:
        expected = SCHEMA.set(
            1, pa.field("coin", pa.dictionary(pa.int32(), pa.string()))
        )
        if (
            reader.schema_arrow != expected
            or reader.metadata.num_rows > max_rows
            or reader.metadata.num_row_groups > 732 * 51
        ):
            raise ValueError("Candidate capacity artifact schema/bounds changed")
        for index in range(reader.metadata.num_row_groups):
            group = reader.metadata.row_group(index)
            if group.total_byte_size > MAX_DECODED_BYTES or any(
                group.column(i).total_uncompressed_size > MAX_DECODED_BYTES
                for i in range(group.num_columns)
            ):
                raise ValueError("Candidate capacity decoded metadata limit")
        values, previous = [], None
        for batch in reader.iter_batches(batch_size=4096, use_threads=False):
            if len(values) + batch.num_rows > max_rows:
                raise ValueError("Candidate capacity decoded row limit")
            if batch.nbytes > MAX_DECODED_BYTES:
                raise ValueError("Candidate capacity decoded batch limit")
            # Keep dictionary strings compressed until their scope is checked;
            # otherwise one huge dictionary value could expand4096times.
            dictionary = batch.column(1).dictionary
            if any(value.as_py() not in source.coins for value in dictionary):
                raise ValueError("Invalid candidate capacity dictionary scope")
            for row in batch.to_pylist():
                date, coin, entrants = row["day"], row["coin"], row["entrants"]
                key = (coin or "", date)
                if (
                    date is None
                    or not source.origin.date() <= date < source.finish.date()
                    or coin is not None
                    and coin not in source.coins
                    or type(entrants) is not int
                    or entrants <= 0
                    or previous is not None
                    and key <= previous
                ):
                    raise ValueError("Invalid candidate capacity count row")
                values.append((date, coin, entrants))
                previous = key
        if len(values) != reader.metadata.num_rows:
            raise ValueError("Candidate capacity decoded row count mismatch")
    by_day = defaultdict(dict)
    for date, coin, entrants in values:
        by_day[date][coin] = entrants
    totals = {coin: 0 for coin in (*source.coins, None)}
    for date in sorted(by_day):
        for coin, entrants in by_day[date].items():
            totals[coin] += entrants
        markets = [totals[coin] for coin in source.coins]
        if not max(markets, default=0) <= totals[None] <= sum(markets):
            raise ValueError("Inconsistent pooled candidate capacity")
    return tuple(values)
