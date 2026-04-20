#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "Neither python nor python3 is available in PATH." >&2
  exit 1
fi
MODE="${1:-surface}"

run_surface() {
  "${PYTHON_BIN}" code/pipelines/build_structure_manifest.py \
    --root . \
    --json-output outputs/manifests/rp_structure_manifest.json \
    --md-output outputs/manifests/rp_structure_manifest.md

  "${PYTHON_BIN}" code/figures/build_output_index.py \
    --root . \
    --output figures/output_index.html

  "${PYTHON_BIN}" code/figures/build_hotel_interactive_outputs.py \
    --root . \
    --output figures/hotel_interactive_index.html
}

run_analysis() {
  "${PYTHON_BIN}" code/pipelines/run_tag_topics.py
  "${PYTHON_BIN}" code/pipelines/run_season_topics.py
  "${PYTHON_BIN}" code/figures/build_cross_segment_outputs.py
  "${PYTHON_BIN}" code/figures/build_topic_hierarchy.py
}

run_compare() {
  "${PYTHON_BIN}" code/pipelines/compare_models.py
  "${PYTHON_BIN}" code/pipelines/compare_sentiment_models.py
}

run_notebooks() {
  "${PYTHON_BIN}" code/pipelines/run_full_pipeline.py
  "${PYTHON_BIN}" code/pipelines/run_stat_tests.py
  "${PYTHON_BIN}" code/pipelines/run_guest_clusters.py
  "${PYTHON_BIN}" code/pipelines/run_topic_nn.py
  "${PYTHON_BIN}" code/pipelines/run_nnmf.py
}

case "${MODE}" in
  surface)
    run_surface
    ;;
  analysis)
    run_surface
    run_analysis
    ;;
  compare)
    run_compare
    ;;
  hotel)
    "${PYTHON_BIN}" code/pipelines/run_hotel_split.py
    ;;
  notebooks)
    run_notebooks
    ;;
  consolidate)
    "${PYTHON_BIN}" code/pipelines/consolidate_outputs.py
    ;;
  *)
    echo "Usage: bash everything.sh [surface|analysis|compare|hotel|notebooks|consolidate]" >&2
    exit 1
    ;;
esac
