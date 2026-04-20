"""
Topic Model Comparison: BERTopic vs NMF vs Ground Truth
========================================================
(Updated with multilingual + domain-specific stopwords)
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from tqdm import tqdm

import kagglehub
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    normalized_mutual_info_score,
    accuracy_score,
)
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer

from nltk.corpus import stopwords

warnings.filterwarnings("ignore")


# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_SIZE = 36_000
RANDOM_SEED = 42
RESULTS_DIR = "results_topic_comparison"
CACHE_PATH  = os.path.join(RESULTS_DIR, "topic_assignments.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Multilingual + domain stopwords ───────────────────────────────────────────
print("Building multilingual stopword list …")

languages = ['english', 'french', 'spanish', 'italian', 'german', 'dutch']

multi_stop_words = []
for lang in languages:
    multi_stop_words.extend(stopwords.words(lang))

# Domain-specific noise
domain_stop_words = [
    "hotel", "room", "rooms", "stay", "stayed", "booking", "booked", "check",
    "night", "day", "time", "area", "place", "city", "walk", "minutes",
    "people", "arrival", "asked", "told", "got", "pay", "work", "located",
    "near", "away", "floor", "building", "way", "morning"
]

# Location-specific noise (removes dataset bias)
domain_stop_words_2 = [
    "amsterdam", "barcelona", "london", "milan", "paris", "vienna",
    "netherlands", "spain", "france", "italy", "austria",
    "eiffel", "oxford", "gogh", "montmartre", "milano",
    "centre", "center", "street", "europe", "european"
]

multi_stop_words.extend(domain_stop_words)
multi_stop_words.extend(domain_stop_words_2)

multi_stop_words = list(set(multi_stop_words))

print(f"Total stop words: {len(multi_stop_words)}")


# ── Ground-truth topic keywords ───────────────────────────────────────────────
TOPIC_KEYWORDS = {
    "Location": ["location", "central", "transport", "station", "airport"],
    "Cleanliness": ["clean", "dirty", "smell", "dust", "hygiene"],
    "Staff": ["staff", "friendly", "helpful", "rude", "service"],
    "Room": ["bed", "bathroom", "noise", "comfortable", "small"],
    "Food": ["breakfast", "food", "restaurant", "coffee"],
    "Value": ["price", "value", "expensive", "cheap", "cost"],
}

LABEL_ORDER = list(TOPIC_KEYWORDS.keys()) + ["Other"]


def keyword_label(text: str) -> str:
    tl = text.lower()
    scores = {k: sum(kw in tl for kw in v) for k, v in TOPIC_KEYWORDS.items()}
    best = max(scores.values())
    if best == 0:
        return "Other"
    return max(scores, key=scores.get)


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading dataset …")
path = kagglehub.dataset_download("jiashenliu/515k-hotel-reviews-data-in-europe")
csv_f = [f for f in os.listdir(path) if f.endswith(".csv")][0]
df = pd.read_csv(os.path.join(path, csv_f))

df["Review_Text"] = df.apply(
    lambda r: " ".join(
        str(v).strip()
        for v in [r.get("Negative_Review", ""), r.get("Positive_Review", "")]
        if isinstance(v, str) and v.strip().lower() not in ("no negative", "no positive", "")
    ),
    axis=1,
)

df = df[df["Review_Text"].str.split().str.len() >= 5].reset_index(drop=True)


# ── Labels ────────────────────────────────────────────────────────────────────
df["True_Label"] = df["Review_Text"].apply(keyword_label)


# ── Sample ────────────────────────────────────────────────────────────────────
df = df.sample(SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)
texts = df["Review_Text"].tolist()


# ── Embeddings ────────────────────────────────────────────────────────────────
print("Encoding text …")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(texts, show_progress_bar=True)


# ── BERTopic ──────────────────────────────────────────────────────────────────
print("Running BERTopic …")

umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0)
hdbscan_model = HDBSCAN(min_cluster_size=30)

vectorizer_model = CountVectorizer(
    stop_words=multi_stop_words,
    min_df=3,
    ngram_range=(1, 2)
)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    representation_model=MaximalMarginalRelevance(diversity=0.8),
)

bert_topics, _ = topic_model.fit_transform(texts, embeddings)
df["BERTopic"] = bert_topics


# ── NMF ───────────────────────────────────────────────────────────────────────
print("Running NMF …")

tfidf = TfidfVectorizer(
    stop_words=multi_stop_words,
    max_features=20000,
    min_df=3,
    ngram_range=(1, 2)
)

X = tfidf.fit_transform(texts)

nmf = NMF(n_components=10, random_state=42)
W = nmf.fit_transform(X)
df["NMF"] = W.argmax(axis=1)


# ── Majority vote mapping ─────────────────────────────────────────────────────
def map_topics(topic_ids):
    mapping = {}
    for t in set(topic_ids):
        subset = df[df["BERTopic"] == t]
        if len(subset) == 0:
            continue
        mapping[t] = subset["True_Label"].mode()[0]
    return [mapping[t] for t in topic_ids]


df["BERT_PRED"] = map_topics(df["BERTopic"])
df["NMF_PRED"]  = map_topics(df["NMF"])


# ── Metrics ───────────────────────────────────────────────────────────────────
print("\nBERTopic Accuracy:", accuracy_score(df["True_Label"], df["BERT_PRED"]))
print("NMF Accuracy:", accuracy_score(df["True_Label"], df["NMF_PRED"]))

print("\nBERTopic Report")
print(classification_report(df["True_Label"], df["BERT_PRED"]))

print("\nNMF Report")
print(classification_report(df["True_Label"], df["NMF_PRED"]))


# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(df["True_Label"], df["BERT_PRED"], labels=LABEL_ORDER)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("BERTopic Confusion Matrix")
plt.show()

# ── Full distributions (no label restriction) ────────────────────────────────
print("\nPlotting FULL topic distributions …")

true_dist = pd.Series(df["True_Label"]).value_counts(normalize=True)
bert_dist = pd.Series(df["BERTopic"]).value_counts(normalize=True)
nmf_dist  = pd.Series(df["NMF"]).value_counts(normalize=True)

# Combine into one dataframe (outer join keeps ALL topics)
dist_df = pd.concat(
    [true_dist.rename("Ground Truth"),
     bert_dist.rename("BERTopic"),
     nmf_dist.rename("NMF")],
    axis=1
).fillna(0)

# Sort by Ground Truth for readability (optional)
dist_df = dist_df.sort_values(by="Ground Truth", ascending=False)

# Plot
plt.figure(figsize=(14, 7))
dist_df.plot(kind="bar", width=0.8)

plt.title("Full Topic Distribution Comparison (All Topics)")
plt.ylabel("Proportion")
plt.xlabel("Topic / Label")

plt.xticks(rotation=75)
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()