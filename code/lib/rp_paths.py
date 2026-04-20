from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping
import os
import runpy


def find_project_root(start: Path | None = None) -> Path:
    candidate = (start or Path(__file__).resolve()).resolve()
    current = candidate if candidate.is_dir() else candidate.parent

    for parent in [current, *current.parents]:
        if (parent / "code").is_dir() and (parent / "README.md").exists():
            return parent

    raise FileNotFoundError(
        "Could not locate the RP project root from the current path."
    )


PROJECT_ROOT = find_project_root()
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"


def resolve_path(path_like: str | os.PathLike[str]) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def relative_to_project(path_like: str | os.PathLike[str]) -> str:
    return str(resolve_path(path_like).relative_to(PROJECT_ROOT))


def resolve_existing_path(
    *relative_candidates: str,
    env_var: str | None = None,
) -> Path:
    if env_var:
        configured = os.environ.get(env_var)
        if configured:
            path = resolve_path(configured)
            if path.exists():
                return path
            raise FileNotFoundError(
                f"{env_var} points to a missing path: {path}"
            )

    for candidate in relative_candidates:
        path = resolve_path(candidate)
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not resolve any of the expected RP paths: "
        + ", ".join(relative_candidates)
    )


def resolve_output_dir(env_var: str, default_relative: str) -> Path:
    path = resolve_path(os.environ.get(env_var, default_relative))
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_output_file(env_var: str, default_relative: str) -> Path:
    path = resolve_path(os.environ.get(env_var, default_relative))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_stratified_reviews_csv(
    env_var: str = "RP_STRATIFIED_REVIEWS_CSV",
) -> Path:
    return resolve_existing_path(
        "Datasets/Hotel_Reviews_StratSamp_Balanced.csv",
        env_var=env_var,
    )


def resolve_hotel_document_info_csv(
    env_var: str = "RP_HOTEL_DOCUMENT_INFO_CSV",
) -> Path:
    return resolve_existing_path(
        "outputs/tables/hotel_pipeline/Hotel_Document_Info.csv",
        "code/BERTModelRawOutputs/Hotel_Document_Info.csv",
        env_var=env_var,
    )


def resolve_hotel_sentiment_scores_csv(
    env_var: str = "RP_HOTEL_SENTIMENT_SCORES_CSV",
) -> Path:
    return resolve_existing_path(
        "outputs/tables/hotel_pipeline/Hotel_Sentiment_Scores.csv",
        "code/BERTModelRawOutputs/Hotel_Sentiment_Scores.csv",
        env_var=env_var,
    )


def resolve_hotel_reviews_csv(
    env_var: str = "RP_HOTEL_REVIEWS_CSV",
    allow_download: bool = False,
) -> Path:
    configured = os.environ.get(env_var)
    if configured:
        path = resolve_path(configured)
        if path.exists():
            return path
        raise FileNotFoundError(f"{env_var} points to a missing CSV: {path}")

    for candidate in ("Datasets/Hotel_Reviews.csv", "code/Hotel_Reviews.csv"):
        path = resolve_path(candidate)
        if path.exists():
            return path

    if allow_download:
        import kagglehub

        dataset_dir = Path(
            kagglehub.dataset_download("jiashenliu/515k-hotel-reviews-data-in-europe")
        )
        csv_files = sorted(dataset_dir.glob("*.csv"))
        if csv_files:
            return csv_files[0].resolve()

    raise FileNotFoundError(
        "Hotel_Reviews.csv was not found in Datasets/ or code/. "
        f"Set {env_var} to an explicit CSV path if needed."
    )


@contextmanager
def temporary_environment(
    env_updates: Mapping[str, str | None] | None = None,
) -> Iterator[None]:
    env_updates = env_updates or {}
    previous: dict[str, str | None] = {}

    try:
        for key, value in env_updates.items():
            previous[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_legacy_script(
    relative_path: str,
    env_updates: Mapping[str, str | None] | None = None,
) -> None:
    script_path = resolve_path(relative_path)
    if not script_path.exists():
        raise FileNotFoundError(f"Legacy script not found: {script_path}")

    previous_cwd = Path.cwd()
    os.chdir(PROJECT_ROOT)
    try:
        with temporary_environment(env_updates):
            runpy.run_path(str(script_path), run_name="__main__")
    finally:
        os.chdir(previous_cwd)
