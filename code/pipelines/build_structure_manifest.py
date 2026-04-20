from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


LIB_DIR = Path(__file__).resolve().parents[1] / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from project_manifest import build_manifest, render_manifest_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an RP structure manifest.")
    parser.add_argument("--root", default=".", help="Project root to scan.")
    parser.add_argument("--json-output", required=True, help="Path to the JSON manifest.")
    parser.add_argument("--md-output", required=True, help="Path to the Markdown manifest.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    json_output = Path(args.json_output)
    md_output = Path(args.md_output)

    manifest = build_manifest(root)

    json_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.parent.mkdir(parents=True, exist_ok=True)

    json_output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    md_output.write_text(render_manifest_markdown(manifest), encoding="utf-8")

    print(f"Saved JSON manifest to {json_output}")
    print(f"Saved Markdown manifest to {md_output}")


if __name__ == "__main__":
    main()
