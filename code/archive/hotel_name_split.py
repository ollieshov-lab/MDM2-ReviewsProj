import math
import re
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset as TorchDataset
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from tqdm import tqdm

from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer

CURRENT_DIR = Path(__file__).resolve().parent
LIB_DIR = next(
    candidate
    for candidate in (CURRENT_DIR / "lib", CURRENT_DIR.parent / "lib")
    if candidate.is_dir()
)
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from rp_paths import resolve_hotel_reviews_csv, resolve_output_dir
from topic_modeling import (
    MIN_WORDS,
    SPLITTER_LIST,
    ZERO_SHOT_MAJOR_TOPICS,
    build_multilingual_stop_words,
    make_safe_filename,
    safe_linkage,
    split_and_update_indices,
)


multi_stop_words = build_multilingual_stop_words()
print(f"Total Multilingual Stop Words: {len(multi_stop_words)}")


# ── Load Data ─────────────────────────────────────────────────────────────────
data_path = resolve_hotel_reviews_csv(allow_download=True)
df = pd.read_csv(data_path)
print(f"Loaded {len(df):,} rows from {data_path}")

df['ID'] = df.index


# ── Extract City from Hotel_Address ──────────────────────────────────────────
# The address format is typically: "Street, City, Country"
# We take the second-to-last comma-separated token as the city.
def extract_city(address: str) -> str:
    if not isinstance(address, str):
        return "Unknown"
    parts = [p.strip() for p in address.split(',')]
    # Last token is usually country/postcode; second-to-last is city
    if len(parts) >= 2:
        return parts[-2].strip()
    return parts[0].strip()

df['City'] = df['Hotel_Address'].apply(extract_city)

# Build a lookup: Hotel_Name → City (one city per hotel)
hotel_city_map = (
    df.groupby('Hotel_Name')['City']
    .agg(lambda x: x.mode().iloc[0])   # most common city label for that hotel
    .to_dict()
)


# ── Build Hotel-Level DataFrames ──────────────────────────────────────────────
hotel_groups = {
    name: group.reset_index(drop=True)
    for name, group in df.groupby('Hotel_Name')
}

hotel_names = sorted(hotel_groups.keys())
print(f"\nTotal hotels found: {len(hotel_names):,}")
for name in hotel_names[:10]:                          # preview first 10
    city = hotel_city_map.get(name, "Unknown")
    print(f"  [{city}] {name} — {len(hotel_groups[name]):,} reviews")
if len(hotel_names) > 10:
    print(f"  ... and {len(hotel_names) - 10} more")


# ── Models ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {DEVICE}")

embedding_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2", device=DEVICE
)

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model        = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    device       = 0 if DEVICE == "cuda" else -1,
    truncation   = True,
    max_length   = 64,
    use_fast     = False,
    model_kwargs = {"use_safetensors": True},
)

LABEL_MAP = {"negative": -1, "neutral": 0, "positive": 1}


# ── Output Dirs ───────────────────────────────────────────────────────────────
OUTPUT_DIR = os.fspath(
    resolve_output_dir("RP_HOTEL_OUTPUT_DIR", "outputs/models/hotel_split")
)
EMBEDDINGS_DIR = os.path.join(OUTPUT_DIR, "embeddings")
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
print(f"Using hotel-level output dir: {OUTPUT_DIR}")


# ── Main Loop ─────────────────────────────────────────────────────────────────
for hotel_name in hotel_names:
    city      = hotel_city_map.get(hotel_name, "Unknown")
    safe_name = make_safe_filename(hotel_name)

    print(f"\n{'='*60}")
    print(f"  Hotel : {hotel_name}")
    print(f"  City  : {city}")
    print(f"{'='*60}")

    sub_df = hotel_groups[hotel_name]

    # ── Build fragment-level lists ────────────────────────────────────────────
    records = []
    for _, row in sub_df.iterrows():
        for col in ["Negative_Review", "Positive_Review"]:
            val = row[col]
            if not isinstance(val, str):
                continue
            val = val.strip()
            if val.lower() in ("no negative", "no positive", "nothing", "n/a", ""):
                continue
            records.append({"text": val, "id": row["ID"]})

    if not records:
        print("  No reviews — skipping.")
        continue

    raw_docs = [r["text"] for r in records]
    raw_ids  = [r["id"]   for r in records]

    docs, doc_index = split_and_update_indices(raw_docs, raw_ids, SPLITTER_LIST)

    stop_set = set(multi_stop_words)
    filtered = [
        (d.strip(), i)
        for d, i in zip(docs, doc_index)
        if len(d.split()) >= MIN_WORDS and d.lower().strip() not in stop_set
    ]
    if not filtered:
        print("  No fragments after filtering — skipping.")
        continue

    docs, doc_index = map(list, zip(*filtered))
    print(f"  Fragments after preprocessing: {len(docs):,}")

    # ── Embed ─────────────────────────────────────────────────────────────────
    EMBEDDING_PATH = os.path.join(EMBEDDINGS_DIR, f"{safe_name}_embeddings.npy")
    if os.path.exists(EMBEDDING_PATH):
        print(f"  Loading saved embeddings from {EMBEDDING_PATH}")
        embeddings = np.load(EMBEDDING_PATH)
        assert len(embeddings) == len(docs), (
            "Embedding count mismatch — delete the .npy file and re-run."
        )
    else:
        print("  Embedding fragments...")
        embeddings = embedding_model.encode(
            docs, batch_size=64, show_progress_bar=True, convert_to_numpy=True
        )
        np.save(EMBEDDING_PATH, embeddings)
        print(f"  Saved embeddings to {EMBEDDING_PATH}")
    print(f"  Embeddings shape: {embeddings.shape}")

    # ── BERTopic ──────────────────────────────────────────────────────────────
    print("  Fitting BERTopic...")
    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0,
        metric='cosine', random_state=42
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=30, min_samples=5, metric='euclidean',
        cluster_selection_method='eom', prediction_data=True
    )
    vectorizer_model     = CountVectorizer(stop_words=multi_stop_words, min_df=3, ngram_range=(1, 2))
    ctfidf_model         = ClassTfidfTransformer(reduce_frequent_words=True)
    representation_model = MaximalMarginalRelevance(diversity=0.8)

    model = BERTopic(
        embedding_model         = embedding_model,
        umap_model              = umap_model,
        hdbscan_model           = hdbscan_model,
        vectorizer_model        = vectorizer_model,
        ctfidf_model            = ctfidf_model,
        representation_model    = representation_model,
        zeroshot_topic_list     = ZERO_SHOT_MAJOR_TOPICS,
        zeroshot_min_similarity = 0.45,
        verbose                 = False,
    )

    topics, _ = model.fit_transform(docs, embeddings=embeddings)

    num_outliers = topics.count(-1)
    print(f"  Topics found (excl. outlier -1): {len(model.get_topic_info()) - 1}")
    print(f"  Initial outliers: {num_outliers}")

    # ── Outlier Reduction (SAFE) ──────────────────────────────────────────────
    if num_outliers > 0:
        print("  Reducing outliers...")
        try:
            new_topics = model.reduce_outliers(
                docs,
                topics,
                strategy="embeddings",
                embeddings=embeddings
            )
            model.update_topics(docs, topics=new_topics)
            topics = new_topics

            remaining_outliers = topics.count(-1)
            print(f"  Outliers remaining after reduction: {remaining_outliers:,}")
        except Exception as e:
            print(f"  ⚠ Outlier reduction skipped ({e})")
    else:
        print("  ✓ No outliers found — skipping reduction.")

    # ── Topic Labels ──────────────────────────────────────────────────────────
    topic_info     = model.get_topic_info()
    topic_to_label = {}
    for _, row in topic_info.iterrows():
        t = row['Topic']
        topic_to_label[t] = "Outlier" if t == -1 else row['Name']
    model.set_topic_labels(topic_to_label)

    # ── Build Document Info ───────────────────────────────────────────────────
    doc_inf = model.get_document_info(docs)
    doc_inf["Person_id"]      = doc_index
    doc_inf["Semantic_Label"] = doc_inf["Topic"].map(topic_to_label)
    doc_inf["Hotel_Name"]     = hotel_name          # ← hotel name column
    doc_inf["City"]           = city                # ← city column

    # ── Sentiment (stratified sample of 36,000 fragments) ────────────────────
    SENTIMENT_SAMPLE = 36_000
    RANDOM_SEED      = 42

    doc_inf["Sentiment_Label"] = None
    doc_inf["Sentiment_Score"] = None

    SENTIMENT_PATH = os.path.join(OUTPUT_DIR, f"{safe_name}_Sentiment_Scores.csv")

    if os.path.exists(SENTIMENT_PATH):
        print(f"  Loading saved sentiment scores from {SENTIMENT_PATH}")
        saved = pd.read_csv(SENTIMENT_PATH)
        doc_inf.update(saved[["Sentiment_Label", "Sentiment_Score"]])

    else:
        eligible = doc_inf[doc_inf['Topic'] != -1].copy()
        n_sample = min(SENTIMENT_SAMPLE, len(eligible))

        topic_counts  = eligible['Topic'].value_counts()
        topic_weights = topic_counts / topic_counts.sum()
        topic_alloc   = (topic_weights * n_sample).round().astype(int)

        sampled_indices = []
        rng = np.random.default_rng(RANDOM_SEED)
        for tid, alloc in topic_alloc.items():
            pool   = eligible[eligible['Topic'] == tid].index.tolist()
            chosen = rng.choice(pool, size=min(alloc, len(pool)), replace=False)
            sampled_indices.extend(chosen.tolist())

        print(f"  Running sentiment on {len(sampled_indices):,} sampled fragments "
              f"(of {len(doc_inf):,} total)...")

        sampled_docs = [docs[doc_inf.index.get_loc(i)] for i in sampled_indices]
        batch_size   = 32
        all_results  = []
        for i in tqdm(range(0, len(sampled_docs), batch_size), desc="  Scoring"):
            batch_result = sentiment_pipeline(sampled_docs[i : i + batch_size])
            all_results.extend(batch_result)

        for idx, result in zip(sampled_indices, all_results):
            doc_inf.at[idx, "Sentiment_Label"] = result['label']
            doc_inf.at[idx, "Sentiment_Score"]  = (
                LABEL_MAP.get(result['label'].lower(), 0) * result['score']
            )

        doc_inf[["Sentiment_Label", "Sentiment_Score"]].to_csv(SENTIMENT_PATH, index=False)
        print(f"  Saved sentiment scores to {SENTIMENT_PATH}")

    # ── Save Document Info ────────────────────────────────────────────────────
    doc_inf.to_csv(
        os.path.join(OUTPUT_DIR, f"{safe_name}_Document_Info.csv"),
        index=False,
    )
    print(f"  Saved Document_Info ({len(doc_inf):,} rows)")

    # ── Topic Sentiment Aggregation ───────────────────────────────────────────
    scored = doc_inf[
        (doc_inf['Topic'] != -1) &
        (doc_inf['Sentiment_Score'].notna())
    ]
    hotel_topic_sentiment = (
        scored
        .groupby('Semantic_Label')
        .agg(
            Mean_Sentiment = ('Sentiment_Score', 'mean'),
            Fragment_Count = ('Sentiment_Score', 'count')
        )
        .reset_index()
    )
    hotel_topic_sentiment = hotel_topic_sentiment[
        hotel_topic_sentiment['Semantic_Label'].isin(ZERO_SHOT_MAJOR_TOPICS)
    ]
    hotel_topic_sentiment = hotel_topic_sentiment[
        hotel_topic_sentiment['Fragment_Count'] >= 3
    ].copy()
    hotel_topic_sentiment['Hotel_Name'] = hotel_name   # ← hotel name column
    hotel_topic_sentiment['City']       = city          # ← city column
    hotel_topic_sentiment.to_csv(
        os.path.join(OUTPUT_DIR, f"{safe_name}_Topic_Sentiment.csv"),
        index=False,
    )

    # ── Hierarchical Topics ───────────────────────────────────────────────────
    try:
        model.hierarchical_topics(docs, linkage_function=safe_linkage)
        print("  ✓ Hierarchy computed")
    except Exception as e:
        print(f"  ⚠ Hierarchy skipped ({e})")

    # ── Save Model ────────────────────────────────────────────────────────────
    model_path = os.path.join(OUTPUT_DIR, f"{safe_name}_bertopic_model")
    model.save(
        model_path,
        serialization        = "safetensors",
        save_ctfidf          = True,
        save_embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"
    )
    print(f"  ✓ Saved model to {model_path}")


# ── Optional: Combined Summary Across All Hotels ──────────────────────────────
all_sentiment_files = [
    os.path.join(OUTPUT_DIR, f"{make_safe_filename(h)}_Topic_Sentiment.csv")
    for h in hotel_names
    if os.path.exists(os.path.join(OUTPUT_DIR, f"{make_safe_filename(h)}_Topic_Sentiment.csv"))
]
if all_sentiment_files:
    combined = pd.concat([pd.read_csv(f) for f in all_sentiment_files], ignore_index=True)
    combined_path = os.path.join(OUTPUT_DIR, "_ALL_Hotels_Topic_Sentiment.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\n✓ Combined sentiment summary saved → {combined_path}")

print(f"\n✓ All done — outputs in ./{OUTPUT_DIR}/")
