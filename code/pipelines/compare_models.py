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
        description="Run the BERTopic vs NMF comparison via the canonical RP pipeline entrypoint."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/tables/topic_model_comparison",
        help="Directory for model-comparison outputs.",
    )
    parser.add_argument(
        "--dataset",
        help="Optional path to Hotel_Reviews.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_updates = {"RP_TOPIC_COMPARISON_OUTPUT_DIR": args.output_dir}
    if args.dataset:
        env_updates["RP_HOTEL_REVIEWS_CSV"] = args.dataset
    run_legacy_script("code/archive/bertopicvsnnmfcomp.py", env_updates=env_updates)


if __name__ == "__main__":
    main()
