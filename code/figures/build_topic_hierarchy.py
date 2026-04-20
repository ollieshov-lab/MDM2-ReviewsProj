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
        description="Export BERTopic hierarchy JSON from the canonical RP figure entrypoint."
    )
    parser.add_argument(
        "--model-dir",
        help="Optional model directory. Defaults to RP autodetection.",
    )
    parser.add_argument(
        "--output-file",
        default="outputs/interactive/bertopic_hierarchy.json",
        help="JSON file to generate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_updates = {"RP_TOPIC_HIERARCHY_OUTPUT_FILE": args.output_file}
    if args.model_dir:
        env_updates["RP_TAG_MODEL_DIR"] = args.model_dir
    run_legacy_script("code/archive/plottagsplit.py", env_updates=env_updates)


if __name__ == "__main__":
    main()
