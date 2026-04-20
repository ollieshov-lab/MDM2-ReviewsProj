# Pipeline Scripts

This directory is the long-term home for canonical RP analysis entrypoints.

In this restructuring pass it contains:

- `build_structure_manifest.py`: inventories the current RP project surface and writes JSON/Markdown manifests
- `consolidate_outputs.py`: migrates legacy output trees into `outputs/`
- `run_tag_topics.py`: canonical traveler-type topic-modelling entrypoint
- `run_season_topics.py`: canonical season-based topic-modelling entrypoint
- `run_hotel_split.py`: canonical hotel-level topic-modelling entrypoint
- `compare_models.py`: canonical BERTopic vs NMF comparison entrypoint
- `compare_sentiment_models.py`: canonical RoBERTa vs VADER comparison entrypoint
- `run_full_pipeline.py`: canonical Python wrapper for `code/notebooks/Main_Full_Pipeline.ipynb`
- `run_stat_tests.py`: canonical Python wrapper for `code/notebooks/statistical_tests.ipynb`
- `run_guest_clusters.py`: canonical Python wrapper for `code/notebooks/guest_clusters.ipynb`
- `run_topic_nn.py`: canonical Python wrapper for `code/notebooks/topic_nn.ipynb`
- `run_nnmf.py`: canonical Python wrapper for `code/notebooks/NNMF.ipynb`

The canonical entrypoints now target notebooks in `code/notebooks/` or archived legacy implementations in `code/archive/`, keeping the repo surface stable while preserving historical code paths.
