"""Byte-capped checkpoint publication, including Parquet metadata/footer writes."""

from io import RawIOBase

import pyarrow.parquet as pq


class _CappedOutput(RawIOBase):
    def __init__(self, output, limit):
        self.output, self.limit = output, limit

    def writable(self):
        return True

    def tell(self):
        return self.output.tell()

    def write(self, value):
        if self.tell() + len(value) > self.limit:
            raise ValueError("Checkpoint seed byte limit exceeded")
        return self.output.write(value)


def write_seed(query, path, max_bytes):
    # Stream fixed-size record batches rather than unrestricted COPY. DuckDB
    # still owns the bounded sort. The stream enforces compressed bytes exactly;
    # the decoded batch guard is checked after each Arrow batch is produced.
    with query.to_arrow_reader(2048) as batches, path.open("xb") as output:
        with pq.ParquetWriter(
            _CappedOutput(output, max_bytes), batches.schema, compression="zstd"
        ) as writer:
            for batch in batches:
                if batch.nbytes > 64 * 1024**2:
                    raise ValueError(
                        "Checkpoint decoded seed batch byte limit exceeded"
                    )
                writer.write_batch(batch)
