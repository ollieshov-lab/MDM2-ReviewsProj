# MDM2 Group Project 2 — Hotel Reviews Analysis
**Topic-Based Sentiment Analysis and Competitive Benchmarking of Hotel Reviews**  
Group 16: Meet Pandit, Nathan Fairclough, John Stephani, Tom Gillmore, Ollie Shovlin

## What This Repository Contains:

This project applies NLP to 515,738 Booking.com hotel reviews across six European cities (Amsterdam, Barcelona, London, Milan, Paris, Vienna). It uses BERTopic for topic modelling and a multilingual RoBERTa model for sentiment analysis to produce actionable, city-benchmarked performance scores for individual hotels.

The main pipeline (`Main_Full_Pipeline.ipynb`) runs end-to-end and covers:

1. **Stratified sampling** — city-balanced sample of 36,000 reviews (30 hotels × 6 cities × 200 reviews)
2. **Text preprocessing** — fragment splitting, stop word removal, placeholder cleaning
3. **Topic modelling** — BERTopic with zero-shot assignment to 10 predefined business categories
4. **Sentiment analysis** — multilingual RoBERTa classifying each fragment as Positive / Neutral / Negative
5. **City benchmarking** — Delta score and ±0.5σ threshold to classify hotel performance relative to local competitors
6. **Results** — Per-hotel radar charts, peer ranking tables, seasonal analysis, and a single-hotel deep dive

Two browser-based Streamlit dashboards are available for exploring results interactively:
- **`Macro_UI.py`** — City-level overview: cross-city benchmarks, topic correlations, seasonal trends
- **`Owner_UI.py`** — Single hotel deep dive: topic scores, peer ranking, and actionable advice

Both require the pipeline to be run first to generate the output files.

## Setup

**Dependencies:** Install from `requirements.txt`

**Dataset:** Download the hotel reviews dataset from  https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe  and place the CSV file in a folder called `Datasets/` in the parent directory.

## Running the Pipeline

Open and run `Main_Full_Pipeline.ipynb` from top to bottom.  
Outputs (CSVs, HTML plots, model files) are saved to `BERTModelRawOutputs/`.

Sentiment scoring is the most computationally expensive step (~53,000 fragments).
Saved outputs are reloaded automatically on subsequent runs.

## Running the Dashboards

After the pipeline has completed, launch either dashboard with:

```bash
# City-level macro dashboard
streamlit run Macro_UI.py

# Single hotel owner dashboard
streamlit run Owner_UI.py
```

Both open automatically in your browser at `http://localhost:8501`.
