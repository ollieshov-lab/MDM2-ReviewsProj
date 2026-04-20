from __future__ import annotations

import argparse
from pathlib import Path
import sys


LIB_DIR = Path(__file__).resolve().parents[1] / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from output_migration import consolidate_legacy_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move legacy RP output trees into the canonical outputs/ layout."
    )
    return parser.parse_args()


def main() -> None:
    parse_args()
    records = consolidate_legacy_outputs()
    if not records:
        print("No legacy outputs required consolidation.")
        return

    print(f"Consolidated {len(records)} legacy output path(s):")
    for record in records:
        print(f"  {record.source} -> {record.destination}")


if __name__ == "__main__":
    main()
