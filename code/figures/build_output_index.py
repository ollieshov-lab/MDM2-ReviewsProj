from __future__ import annotations

import argparse
import html
import os
from pathlib import Path
import sys


LIB_DIR = Path(__file__).resolve().parents[1] / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from project_manifest import build_manifest, discover_html_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an HTML index for RP interactive outputs.")
    parser.add_argument("--root", default=".", help="Project root to scan.")
    parser.add_argument("--output", required=True, help="HTML file to generate.")
    return parser.parse_args()


def render_section(section: dict[str, object], root: Path, output_path: Path) -> str:
    files = section["files"]
    if not files:
        return (
            f"<section><h2>{html.escape(section['name'])}</h2>"
            "<p>No HTML outputs found.</p></section>"
        )

    items = []
    for file_path in files[:50]:
        rel_href = Path(
            os.path.relpath(file_path.resolve(), output_path.parent.resolve())
        ).as_posix()
        try:
            label_path = file_path.resolve().relative_to(root.resolve())
        except ValueError:
            label_path = file_path.resolve()
        items.append(
            "<li>"
            f"<a href=\"{html.escape(rel_href)}\">{html.escape(file_path.name)}</a>"
            f" <code>{html.escape(str(label_path))}</code>"
            "</li>"
        )

    remainder = section["count"] - min(section["count"], 50)
    if remainder > 0:
        items.append(f"<li>... and {remainder} more file(s)</li>")

    return (
        f"<section><h2>{html.escape(section['name'])}</h2>"
        f"<p>{section['count']} HTML file(s)</p>"
        f"<ul>{''.join(items)}</ul></section>"
    )


def build_html(root: Path, output_path: Path) -> str:
    manifest = build_manifest(root)
    sections = discover_html_outputs(root)

    top_level_rows = []
    for entry in manifest["entries"]:
        if entry["is_dir"]:
            detail = f"{entry.get('file_count', 0)} files"
        else:
            detail = f"{entry.get('size_bytes', 0)} bytes"
        top_level_rows.append(
            "<tr>"
            f"<td><code>{html.escape(entry['relative_path'])}</code></td>"
            f"<td>{html.escape(entry['classification'])}</td>"
            f"<td>{html.escape(detail)}</td>"
            "</tr>"
        )

    section_html = "".join(render_section(section, root, output_path) for section in sections)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>RP Output Index</title>
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
    table {{
      border-collapse: collapse;
      width: 100%;
      margin-bottom: 2rem;
      background: white;
    }}
    th, td {{
      border: 1px solid #d6d1c4;
      padding: 0.5rem 0.75rem;
      text-align: left;
    }}
    th {{
      background: #efe8d8;
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
  <h1>RP Output Index</h1>
  <p>This index is generated to make canonical RP interactive outputs discoverable while still surfacing any legacy locations that remain.</p>
  <h2>Top-Level Inventory</h2>
  <table>
    <thead>
      <tr><th>Path</th><th>Classification</th><th>Detail</th></tr>
    </thead>
    <tbody>
      {''.join(top_level_rows)}
    </tbody>
  </table>
  {section_html}
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_html(root, output_path.resolve()), encoding="utf-8")
    print(f"Saved HTML output index to {output_path}")


if __name__ == "__main__":
    main()
