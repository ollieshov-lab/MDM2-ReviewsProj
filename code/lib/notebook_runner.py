from __future__ import annotations

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from rp_paths import resolve_path


def execute_notebook(
    notebook_relative: str,
    *,
    execution_dir: str = "code",
    executed_notebook_relative: str | None = None,
    kernel_name: str = "python3",
    timeout_seconds: int = 7200,
) -> Path:
    notebook_path = resolve_path(notebook_relative)
    execution_path = resolve_path(execution_dir)

    with notebook_path.open("r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)

    executor = ExecutePreprocessor(
        timeout=timeout_seconds,
        kernel_name=kernel_name,
    )
    executor.preprocess(notebook, {"metadata": {"path": str(execution_path)}})

    if executed_notebook_relative:
        executed_path = resolve_path(executed_notebook_relative)
        executed_path.parent.mkdir(parents=True, exist_ok=True)
        with executed_path.open("w", encoding="utf-8") as handle:
            nbformat.write(notebook, handle)
        return executed_path

    return notebook_path
