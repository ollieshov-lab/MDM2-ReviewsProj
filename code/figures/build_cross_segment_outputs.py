from __future__ import annotations

import argparse
from pathlib import Path
import sys


LIB_DIR = Path(__file__).resolve().parents[1] / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from rp_paths import run_legacy_script


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cross-segment traveler/season interactive outputs."
    )
    parser.add_argument(
        "--tag-input-dir",
        help="Optional traveler-type results directory. Defaults to RP autodetection.",
    )
    parser.add_argument(
        "--season-input-dir",
        help="Optional season results directory. Defaults to RP autodetection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_updates: dict[str, str] = {}
    if args.tag_input_dir:
        env_updates["RP_TAG_INPUT_DIR"] = args.tag_input_dir
    if args.season_input_dir:
        env_updates["RP_SEASON_INPUT_DIR"] = args.season_input_dir
    run_legacy_script("code/archive/plottimesplit.py", env_updates=env_updates)


if __name__ == "__main__":
    main()
