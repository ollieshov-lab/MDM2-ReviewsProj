from __future__ import annotations

import argparse
import html
import os
from pathlib import Path
import sys


LIB_DIR = Path(__file__).resolve().parents[1] / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))


SECTION_PATHS = {
    "Hotel pipeline outputs": ["outputs/interactive/hotel_pipeline"],
    "Hotel cross-tag outputs": ["outputs/interactive/hotel_cross_tags"],
    "Legacy hotel pipeline outputs": ["code/BERTModelRawOutputs/Plots"],
    "Legacy hotel cross-tag outputs": ["results_cross_tags_hotels/Plots"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an HTML index for hotel-level RP interactive outputs."
    )
    parser.add_argument("--root", default=".", help="Project root to scan.")
    parser.add_argument(
        "--output",
        default="figures/hotel_interactive_index.html",
        help="HTML file to generate.",
    )
    return parser.parse_args()


def collect_html_files(root: Path, relative_paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for relative_path in relative_paths:
        path = root / relative_path
        if not path.exists():
            continue
        if path.is_file() and path.suffix.lower() == ".html":
            files.append(path)
            continue
        files.extend(sorted(path.rglob("*.html")))
    return files


def render_section(name: str, files: list[Path], root: Path, output_path: Path) -> str:
    if not files:
        return f"<section><h2>{html.escape(name)}</h2><p>No files found.</p></section>"

    items = []
    for file_path in files:
        rel_href = Path(
            os.path.relpath(file_path.resolve(), output_path.parent.resolve())
        ).as_posix()
        rel_label = file_path.resolve().relative_to(root.resolve())
        items.append(
            "<li>"
            f"<a href=\"{html.escape(rel_href)}\">{html.escape(file_path.name)}</a> "
            f"<code>{html.escape(str(rel_label))}</code>"
            "</li>"
        )

    return (
        f"<section><h2>{html.escape(name)}</h2>"
        f"<p>{len(files)} HTML file(s)</p>"
        f"<ul>{''.join(items)}</ul></section>"
    )


def build_html(root: Path, output_path: Path) -> str:
    sections = []
    for name, relative_paths in SECTION_PATHS.items():
        sections.append(
            render_section(
                name,
                collect_html_files(root, relative_paths),
                root,
                output_path,
            )
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>RP Hotel Interactive Outputs</title>
  <style>
    body {{
      font-family: Georgia, "Times New Roman", serif;
      margin: 2rem auto;
      max-width: 980px;
      line-height: 1.5;
      color: #1f1f1f;
      background: #faf8f2;
      padding: 0 1rem 3rem;
    }}
    h1, h2 {{
      color: #213547;
    }}
    section {{
      background: white;
      border: 1px solid #d6d1c4;
      padding: 1rem 1.25rem;
      margin-bottom: 1rem;
    }}
    code {{
      font-size: 0.95em;
    }}
  </style>
</head>
<body>
  <h1>RP Hotel Interactive Outputs</h1>
  <p>This index groups hotel-level interactive outputs under the migrated RP layout while still surfacing any legacy locations that remain.</p>
  {''.join(sections)}
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_html(root, output_path.resolve()), encoding="utf-8")
    print(f"Saved hotel interactive output index to {output_path}")


if __name__ == "__main__":
    main()
