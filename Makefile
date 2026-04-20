PYTHON ?= python

.PHONY: all surface structure-manifest output-index hotel-interactive-index consolidate-outputs tag-topics season-topics hotel-split cross-segment-outputs topic-hierarchy compare-models compare-sentiment full-pipeline stat-tests guest-clusters topic-nn nnmf

all: surface

surface: structure-manifest output-index hotel-interactive-index

structure-manifest:
	$(PYTHON) code/pipelines/build_structure_manifest.py --root . --json-output outputs/manifests/rp_structure_manifest.json --md-output outputs/manifests/rp_structure_manifest.md

output-index:
	$(PYTHON) code/figures/build_output_index.py --root . --output figures/output_index.html

hotel-interactive-index:
	$(PYTHON) code/figures/build_hotel_interactive_outputs.py --root . --output figures/hotel_interactive_index.html

consolidate-outputs:
	$(PYTHON) code/pipelines/consolidate_outputs.py

tag-topics:
	$(PYTHON) code/pipelines/run_tag_topics.py

season-topics:
	$(PYTHON) code/pipelines/run_season_topics.py

hotel-split:
	$(PYTHON) code/pipelines/run_hotel_split.py

cross-segment-outputs:
	$(PYTHON) code/figures/build_cross_segment_outputs.py

topic-hierarchy:
	$(PYTHON) code/figures/build_topic_hierarchy.py

compare-models:
	$(PYTHON) code/pipelines/compare_models.py

compare-sentiment:
	$(PYTHON) code/pipelines/compare_sentiment_models.py

full-pipeline:
	$(PYTHON) code/pipelines/run_full_pipeline.py

stat-tests:
	$(PYTHON) code/pipelines/run_stat_tests.py

guest-clusters:
	$(PYTHON) code/pipelines/run_guest_clusters.py

topic-nn:
	$(PYTHON) code/pipelines/run_topic_nn.py

nnmf:
	$(PYTHON) code/pipelines/run_nnmf.py
