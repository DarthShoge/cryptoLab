"""Create a new explicitly synthetic cross-class lab dataset without downloads."""

import argparse
from pathlib import Path
from arblab.hyperliquid_copy.lab_fixture_v2 import write_fixture

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
print("SYNTHETIC cross-class dataset:", write_fixture(args.output).resolve())
