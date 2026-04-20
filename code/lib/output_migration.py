from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rp_paths import PROJECT_ROOT, resolve_path


@dataclass(frozen=True)
class MoveRecord:
    source: str
    destination: str


DIRECTORY_MOVES: tuple[tuple[str, str], ...] = (
    ("results_zeroshot_tags", "outputs/models/zeroshot_tags"),
    ("results_zeroshot_seasons", "outputs/models/zeroshot_seasons"),
    ("results_by_hotel", "outputs/models/hotel_split"),
    ("results", "outputs/models/legacy_results"),
    ("results_cross_segment/Plots", "outputs/interactive/cross_segment"),
    ("results_cross_tags_hotels/Plots", "outputs/interactive/hotel_cross_tags"),
    ("results_topic_comparison", "outputs/tables/topic_model_comparison"),
    ("results_sentiment_comparison", "outputs/tables/sentiment_model_comparison"),
    ("code/BERTModelRawOutputs/Plots", "outputs/interactive/hotel_pipeline"),
    ("code/BERTModelRawOutputs/stats_tests_outputs", "outputs/tables/stats_tests"),
    ("code/BERTModelRawOutputs/guest_clusters_outputs", "outputs/intermediate/guest_clusters"),
    ("code/LDA/results", "outputs/interactive/lda"),
)

FILE_MOVES: tuple[tuple[str, str], ...] = (
    ("interactive_topics.html", "outputs/interactive/interactive_topics.html"),
    ("interactive_seasons.html", "outputs/interactive/interactive_seasons.html"),
    (
        "results_cross_segment/cross_TravelerType_Season_Topic.csv",
        "outputs/tables/cross_segment/cross_TravelerType_Season_Topic.csv",
    ),
    (
        "time_series_findings/time_series_output.txt",
        "outputs/tables/time_series_findings/time_series_output.txt",
    ),
)


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    counter = 1
    while True:
        candidate = path.with_name(f"{stem}_legacy_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _move_file(source: Path, destination: Path, records: list[MoveRecord]) -> None:
    if not source.exists():
        return

    destination = _unique_destination(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.replace(destination)
    records.append(
        MoveRecord(
            source=str(source.relative_to(PROJECT_ROOT)),
            destination=str(destination.relative_to(PROJECT_ROOT)),
        )
    )


def _move_tree(source: Path, destination: Path, records: list[MoveRecord]) -> None:
    if not source.exists():
        return

    destination.mkdir(parents=True, exist_ok=True)
    for child in sorted(source.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        target = destination / child.name
        if child.is_dir():
            _move_tree(child, target, records)
        else:
            _move_file(child, target, records)

    if source.exists() and not any(source.iterdir()):
        source.rmdir()


def _move_bert_root_files(records: list[MoveRecord]) -> None:
    source_root = resolve_path("code/BERTModelRawOutputs")
    if not source_root.exists():
        return

    for file_path in sorted(source_root.glob("*.csv")):
        _move_file(
            file_path,
            resolve_path("outputs/tables/hotel_pipeline") / file_path.name,
            records,
        )

    for file_path in sorted(source_root.glob("*.npy")):
        _move_file(
            file_path,
            resolve_path("outputs/intermediate/hotel_pipeline") / file_path.name,
            records,
        )


def _remove_empty_parent_chain(path: Path) -> None:
    current = path
    while current != PROJECT_ROOT and current.exists():
        try:
            next(current.iterdir())
            break
        except StopIteration:
            current.rmdir()
            current = current.parent


def consolidate_legacy_outputs() -> list[MoveRecord]:
    records: list[MoveRecord] = []

    for source_relative, destination_relative in DIRECTORY_MOVES:
        _move_tree(resolve_path(source_relative), resolve_path(destination_relative), records)

    for source_relative, destination_relative in FILE_MOVES:
        source_path = resolve_path(source_relative)
        destination_path = resolve_path(destination_relative)
        if source_path.exists():
            _move_file(source_path, destination_path, records)
            _remove_empty_parent_chain(source_path.parent)

    _move_bert_root_files(records)
    _remove_empty_parent_chain(resolve_path("code/BERTModelRawOutputs"))
    _remove_empty_parent_chain(resolve_path("results_cross_segment"))
    _remove_empty_parent_chain(resolve_path("results_cross_tags_hotels"))
    _remove_empty_parent_chain(resolve_path("time_series_findings"))

    return records
