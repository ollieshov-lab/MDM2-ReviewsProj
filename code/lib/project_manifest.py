from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import os


EXCLUDE_DIRS = {".git", ".venv", "__pycache__", ".ipynb_checkpoints"}

TOP_LEVEL_TYPES = {
    ".git": "vcs",
    ".gitignore": "config",
    ".venv": "environment",
    "README.md": "docs",
    "Makefile": "automation",
    "everything.sh": "automation",
    "requirements.txt": "config",
    "outline.md": "docs",
    "code": "source",
    "figures": "curated_outputs",
    "notes": "notes",
    "report": "report",
    "outputs": "generated_outputs",
    "Datasets": "raw_data",
    "results": "legacy_generated_outputs",
    "results_cross_segment": "legacy_generated_outputs",
    "results_cross_tags_hotels": "legacy_generated_outputs",
    "results_zeroshot_seasons": "legacy_generated_outputs",
    "results_zeroshot_tags": "legacy_generated_outputs",
    "results_by_hotel": "legacy_generated_outputs",
    "results_topic_comparison": "legacy_generated_outputs",
    "results_sentiment_comparison": "legacy_generated_outputs",
    "time_series_findings": "legacy_generated_outputs",
    "interactive_topics.html": "legacy_generated_outputs",
    "interactive_seasons.html": "legacy_generated_outputs",
}

TYPE_NOTES = {
    "source": "Restructure toward lib/pipelines/figures/notebooks/archive.",
    "curated_outputs": "Curated report-facing outputs should land here.",
    "generated_outputs": "Canonical generated-results namespace.",
    "legacy_generated_outputs": "Legacy generated output; migrate or delete after consolidation.",
    "raw_data": "Keep local-only unless a curated subset is required.",
    "notes": "Meeting and planning material.",
    "report": "Report source and compiled report assets.",
    "automation": "Root entrypoints should call real RP scripts only.",
    "config": "Repo configuration or environment specification.",
    "docs": "Project or steering documentation.",
    "environment": "Local environment state; do not track.",
    "vcs": "Repository metadata.",
    "unknown": "Review during migration.",
}

HTML_SECTIONS = {
    "Canonical interactive outputs": [Path("outputs/interactive")],
    "Curated figures": [Path("figures")],
    "Legacy root interactive outputs": [Path("interactive_topics.html"), Path("interactive_seasons.html")],
    "Legacy cross-segment outputs": [Path("results_cross_segment")],
    "Legacy hotel outputs": [Path("results_cross_tags_hotels"), Path("code/BERTModelRawOutputs/Plots")],
}


def classify_top_level(path: Path) -> str:
    return TOP_LEVEL_TYPES.get(path.name, "unknown")


def summarize_directory(path: Path) -> dict[str, object]:
    file_count = 0
    dir_count = 0
    ext_counter: Counter[str] = Counter()

    for current_root, dirnames, filenames in os.walk(path):
        dirnames[:] = [name for name in dirnames if name not in EXCLUDE_DIRS]
        dir_count += len(dirnames)
        for filename in filenames:
            file_count += 1
            ext = Path(filename).suffix.lower() or "[no-ext]"
            ext_counter[ext] += 1

    return {
        "file_count": file_count,
        "dir_count": dir_count,
        "top_extensions": [
            {"extension": extension, "count": count}
            for extension, count in ext_counter.most_common(5)
        ],
    }


def build_manifest(root: Path) -> dict[str, object]:
    entries: list[dict[str, object]] = []

    for path in sorted(root.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        entry_type = classify_top_level(path)
        entry = {
            "name": path.name,
            "relative_path": path.name,
            "is_dir": path.is_dir(),
            "classification": entry_type,
            "note": TYPE_NOTES.get(entry_type, TYPE_NOTES["unknown"]),
        }
        if path.is_dir():
            entry.update(summarize_directory(path))
        else:
            entry["size_bytes"] = path.stat().st_size
        entries.append(entry)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root.resolve()),
        "project": root.name,
        "entries": entries,
    }


def render_manifest_markdown(manifest: dict[str, object]) -> str:
    lines = [
        "# RP Structure Manifest",
        "",
        f"- Project: `{manifest['project']}`",
        f"- Root: `{manifest['root']}`",
        f"- Generated: `{manifest['generated_at_utc']}`",
        "",
        "## Top-Level Entries",
        "",
        "| Path | Kind | Classification | Detail | Note |",
        "| --- | --- | --- | --- | --- |",
    ]

    for entry in manifest["entries"]:
        if entry["is_dir"]:
            detail = (
                f"{entry.get('file_count', 0)} files, "
                f"{entry.get('dir_count', 0)} subdirs"
            )
        else:
            detail = f"{entry.get('size_bytes', 0)} bytes"
        lines.append(
            f"| `{entry['relative_path']}` | "
            f"{'dir' if entry['is_dir'] else 'file'} | "
            f"{entry['classification']} | {detail} | {entry['note']} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This manifest is designed to support the RP restructure and should stay script-generated.",
            "- Canonical generated outputs should live under `outputs/`.",
            "- Any top-level legacy output trees that still appear should be treated as migration leftovers.",
        ]
    )
    return "\n".join(lines) + "\n"


def discover_html_outputs(root: Path) -> list[dict[str, object]]:
    sections: list[dict[str, object]] = []

    for section_name, paths in HTML_SECTIONS.items():
        files: list[Path] = []
        for rel_path in paths:
            path = root / rel_path
            if not path.exists():
                continue
            if path.is_file() and path.suffix.lower() == ".html":
                files.append(path)
            elif path.is_dir():
                files.extend(sorted(path.rglob("*.html")))
        sections.append(
            {
                "name": section_name,
                "count": len(files),
                "files": files,
            }
        )

    return sections
