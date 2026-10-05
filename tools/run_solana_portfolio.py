"""Run the local read-only API used by apps/solana-portfolio."""

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "apps" / "solana-portfolio")
)
from api.server import main

if __name__ == "__main__":
    main()
