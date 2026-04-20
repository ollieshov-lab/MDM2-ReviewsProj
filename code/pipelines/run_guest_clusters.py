from __future__ import annotations

import argparse
from pathlib import Path
import sys


LIB_DIR = Path(__file__).resolve().parents[1] / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from notebook_runner import execute_notebook
from output_migration import consolidate_legacy_outputs


NOTEBOOK_PATH = "code/notebooks/guest_clusters.ipynb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute the migrated RP guest-clusters notebook via a canonical Python entrypoint."
    )
    parser.add_argument(
        "--executed-notebook-output",
        default="outputs/intermediate/notebook_runs/guest_clusters.executed.ipynb",
        help="Optional path for the executed notebook copy.",
    )
    parser.add_argument(
        "--skip-consolidate",
        action="store_true",
        help="Skip consolidation of any legacy output paths created by the notebook.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    execute_notebook(
        NOTEBOOK_PATH,
        execution_dir="code",
        executed_notebook_relative=args.executed_notebook_output,
    )
    if not args.skip_consolidate:
        consolidate_legacy_outputs()


if __name__ == "__main__":
    main()
