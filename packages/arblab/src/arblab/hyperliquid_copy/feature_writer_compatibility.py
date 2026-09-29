"""Exact compatibility declaration for the 2026-09-24 reservation-only fix.

Feature cache keys identify compatible feature semantics. This explicit pair
retains the prior identity because rows, ordering, schema and shard boundaries
are unchanged; only admission of writes changes. Unknown versions get their real
hashes. Experiment provenance independently records every actual Python hash.
"""

from pathlib import Path

from .download import file_hash

LEGACY = {
    "feature_writer.py": "059f867cd804f9adde7b4d87f541d6b10ff61e6bab00fecdd69bc0d1a89260b8",
    "feature_publication.py": "3e02738fc88a4cf65088cae54ad7c4b72c338af06de4703c388304956c988dca",
    "feature_metric_producer.py": "01ae91d0cc8a5d9411bb8ea0772fe5a46e3bc299a260c18515d32474dec9645f",
}
COMPATIBLE = {
    "feature_writer.py": "96246806e179b3f533528844c570db30f6a67a25121a0a80bb5e631cf94cc16d",
    "feature_publication.py": "a6d4367942cac8aaed218d57d8a836e5d37d17a90b228ddd03c6c1c345d3a2b9",
    "feature_writer_budget.py": "abfe57348b9702243fa724f41362fa4016c9bcbcc2a2913fd471282f8413e35d",
    "feature_metric_producer.py": "98a3dc15578b30d5403a01217ac300af0eefb3d01c13133779708165a78d9575",
}


def compatible_code(code):
    actual = {name: file_hash(Path(__file__).with_name(name)) for name in COMPATIBLE}
    if actual == COMPATIBLE and all(
        code[name] == actual[name] for name in LEGACY if name in code
    ):
        return {name: LEGACY.get(name, digest) for name, digest in code.items()}
    return {
        **code,
        **actual,
        "feature_writer_compatibility.py": file_hash(Path(__file__)),
    }
