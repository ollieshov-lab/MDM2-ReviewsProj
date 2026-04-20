# Figure Scripts

This directory is the long-term home for curated figure and interactive-output builders.

In this restructuring pass it contains:

- `build_output_index.py`: generates an HTML index of existing interactive outputs while migration is in progress
- `build_cross_segment_outputs.py`: canonical builder for traveler/season comparison outputs
- `build_topic_hierarchy.py`: canonical builder for exported BERTopic hierarchy JSON
- `build_hotel_interactive_outputs.py`: canonical builder for the hotel-level interactive output index

Future report-facing figure scripts should be added here instead of being embedded in notebooks or archived legacy scripts.
