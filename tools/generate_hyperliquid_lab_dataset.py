"""Write a new fabricated local lab dataset; no downloads or exchange access."""

import argparse
from pathlib import Path
from arblab.hyperliquid_copy.lab_fixture import write_fixture

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output",
    type=Path,
    required=True,
    help="New lab-root/datasets/dataset-id directory",
)
args = parser.parse_args()
print("SYNTHETIC dataset:", write_fixture(args.output).resolve())
