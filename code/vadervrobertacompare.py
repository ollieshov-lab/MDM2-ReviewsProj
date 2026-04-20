"""
Sentiment Comparison: RoBERTa vs VADER vs Ground Truth (Review Score)
======================================================================
Ground truth labels are derived from the Reviewer_Score column:
  - Median score is computed on the full dataset.
  - A ±NEUTRAL_BAND tolerance window around the median defines "Neutral".
  - Scores above the band → Positive; below the band → Negative.

Both models are run on a stratified sample of 36,000 reviews.
Confusion matrices are produced for each model vs. ground truth.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from tqdm import tqdm

import kagglehub
import torch
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_SIZE   = 36_000
RANDOM_SEED   = 42
NEUTRAL_BAND  = 0.5          # ±0.5 around median is classed "Neutral"
BATCH_SIZE    = 64
RESULTS_DIR   = "results_sentiment_comparison"
CACHE_PATH    = os.path.join(RESULTS_DIR, "scored_sample.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)

LABEL_ORDER   = ["Negative", "Neutral", "Positive"]   # consistent axis order


# ── Load Data ─────────────────────────────────────────────────────────────────
print("Loading dataset …")
path    = kagglehub.dataset_download("jiashenliu/515k-hotel-reviews-data-in-europe")
csv_f   = [f for f in os.listdir(path) if f.endswith(".csv")][0]
df      = pd.read_csv(os.path.join(path, csv_f))
print(f"  Loaded {len(df):,} rows")

# Combine both review columns into a single text field per row
df["Review_Text"] = df.apply(
    lambda r: " ".join(
        str(v).strip()
        for v in [r.get("Negative_Review", ""), r.get("Positive_Review", "")]
        if isinstance(v, str)
        and v.strip().lower() not in ("no negative", "no positive", "nothing", "n/a", "")
    ),
    axis=1,
)
df = df[df["Review_Text"].str.split().str.len() >= 5].reset_index(drop=True)
print(f"  Rows with usable review text: {len(df):,}")


# ── Ground Truth Labels ───────────────────────────────────────────────────────
median_score = df["Reviewer_Score"].median()
print(f"\nMedian Reviewer_Score: {median_score:.2f}")
print(f"  Neutral band: ({median_score - NEUTRAL_BAND:.2f}, {median_score + NEUTRAL_BAND:.2f})")

def score_to_label(score):
    if score > median_score + NEUTRAL_BAND:
        return "Positive"
    elif score < median_score - NEUTRAL_BAND:
        return "Negative"
    else:
        return "Neutral"

df["True_Label"] = df["Reviewer_Score"].apply(score_to_label)
print("\nGround-truth label distribution (full corpus):")
print(df["True_Label"].value_counts().to_string())


# ── Stratified Sample ─────────────────────────────────────────────────────────
print(f"\nDrawing stratified sample of {SAMPLE_SIZE:,} …")
rng     = np.random.default_rng(RANDOM_SEED)
strata  = df["True_Label"].value_counts()
alloc   = (strata / strata.sum() * SAMPLE_SIZE).round().astype(int)

parts = []
for label, n in alloc.items():
    pool  = df[df["True_Label"] == label]
    idx   = rng.choice(pool.index, size=min(n, len(pool)), replace=False)
    parts.append(df.loc[idx])

sample_df = pd.concat(parts).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
print(f"  Sample size: {len(sample_df):,}")
print(sample_df["True_Label"].value_counts().to_string())


# ── Models ────────────────────────────────────────────────────────────────────
if os.path.exists(CACHE_PATH):
    # ── Load from cache ───────────────────────────────────────────────────────
    print(f"\nLoading cached scores from {CACHE_PATH} …")
    sample_df = pd.read_csv(CACHE_PATH)

else:
    # ── RoBERTa ───────────────────────────────────────────────────────────────
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {DEVICE}")

    roberta_pipe = pipeline(
        "sentiment-analysis",
        model        = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        device       = 0 if DEVICE == "cuda" else -1,
        truncation   = True,
        max_length   = 128,
        use_fast     = False,
        model_kwargs = {"use_safetensors": True},
    )

    ROBERTA_MAP = {
        "negative" : "Negative",
        "neutral"  : "Neutral",
        "positive" : "Positive",
    }

    print("Running RoBERTa …")
    roberta_labels = []
    texts = sample_df["Review_Text"].tolist()
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="  RoBERTa"):
        batch   = texts[i : i + BATCH_SIZE]
        results = roberta_pipe(batch)
        roberta_labels.extend(ROBERTA_MAP.get(r["label"].lower(), "Neutral") for r in results)

    sample_df["RoBERTa_Label"] = roberta_labels
    del roberta_pipe   # free GPU memory

    # ── VADER ─────────────────────────────────────────────────────────────────
    print("Running VADER …")
    analyser = SentimentIntensityAnalyzer()

    def vader_label(text):
        compound = analyser.polarity_scores(str(text))["compound"]
        if compound >= 0.05:
            return "Positive"
        elif compound <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    tqdm.pandas(desc="  VADER")
    sample_df["VADER_Label"] = sample_df["Review_Text"].progress_apply(vader_label)

    # ── Cache ─────────────────────────────────────────────────────────────────
    sample_df.to_csv(CACHE_PATH, index=False)
    print(f"  Saved scores to {CACHE_PATH}")


# ── Confusion Matrices ────────────────────────────────────────────────────────
from sklearn.metrics import confusion_matrix, classification_report

def build_cm(y_true, y_pred, labels):
    """Return a DataFrame confusion matrix with readable row/col names."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"True: {l}" for l in labels],
                           columns=[f"Pred: {l}" for l in labels])

true_labels    = sample_df["True_Label"]
roberta_labels = sample_df["RoBERTa_Label"]
vader_labels   = sample_df["VADER_Label"]

cm_roberta = build_cm(true_labels, roberta_labels, LABEL_ORDER)
cm_vader   = build_cm(true_labels, vader_labels,   LABEL_ORDER)

print("\n── RoBERTa Confusion Matrix ──────────────────────────────────")
print(cm_roberta.to_string())
print("\n── VADER Confusion Matrix ────────────────────────────────────")
print(cm_vader.to_string())

print("\n── RoBERTa Classification Report ────────────────────────────")
print(classification_report(true_labels, roberta_labels, target_names=LABEL_ORDER))
print("\n── VADER Classification Report ──────────────────────────────")
print(classification_report(true_labels, vader_labels, target_names=LABEL_ORDER))


# ── Plot ──────────────────────────────────────────────────────────────────────
LABEL_COLORS = {
    "Negative" : "#E05252",
    "Neutral"  : "#C4A84F",
    "Positive" : "#52A882",
}

def plot_confusion_matrix(cm_df, title, ax, annot_fontsize=13):
    """
    Plot a confusion matrix with a custom, styled heatmap.
    Values are shown as counts AND as % of the true class (row %).
    """
    raw      = cm_df.values.astype(float)
    row_sums = raw.sum(axis=1, keepdims=True)
    pct      = np.where(row_sums > 0, raw / row_sums * 100, 0)

    annots = np.empty(raw.shape, dtype=object)
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            annots[i, j] = f"{int(raw[i,j]):,}\n({pct[i,j]:.1f}%)"

    sns.heatmap(
        pct, ax=ax,
        cmap="RdYlGn",
        vmin=0, vmax=100,
        linewidths=0.6,
        linecolor="#1a1a2e",
        annot=annots,
        fmt="",
        annot_kws={"fontsize": annot_fontsize, "fontweight": "bold"},
        cbar_kws={"shrink": 0.75, "format": mticker.PercentFormatter()},
        xticklabels=[l.replace("Pred: ", "") for l in cm_df.columns],
        yticklabels=[l.replace("True: ", "") for l in cm_df.index],
    )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=8)
    ax.set_ylabel("True Label",      fontsize=12, labelpad=8)
    ax.tick_params(axis="both", labelsize=11)


# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor("#0f0f1a")
for ax in axes:
    ax.set_facecolor("#0f0f1a")

plt.rcParams.update({
    "text.color"       : "white",
    "axes.labelcolor"  : "white",
    "xtick.color"      : "white",
    "ytick.color"      : "white",
})

plot_confusion_matrix(cm_roberta, "RoBERTa  (twitter-roberta-base-sentiment-latest)", axes[0])
plot_confusion_matrix(cm_vader,   "VADER  (vaderSentiment)",                           axes[1])

fig.suptitle(
    f"Sentiment Model Comparison — n = {len(sample_df):,} hotel reviews\n"
    f"Ground Truth: Reviewer_Score (median = {sample_df['Reviewer_Score'].median():.1f}, "
    f"neutral band ±{NEUTRAL_BAND})",
    fontsize=14, y=1.02, color="white", fontweight="bold"
)

plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, "confusion_matrices.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\n✓ Saved confusion matrix plot → {out_path}")


# ── Per-class accuracy bar chart ──────────────────────────────────────────────
from sklearn.metrics import accuracy_score

records = []
for label in LABEL_ORDER:
    mask = true_labels == label
    records.append({
        "Label"   : label,
        "RoBERTa" : accuracy_score(true_labels[mask], roberta_labels[mask]),
        "VADER"   : accuracy_score(true_labels[mask], vader_labels[mask]),
    })
acc_df = pd.DataFrame(records)

fig2, ax2 = plt.subplots(figsize=(10, 5))
fig2.patch.set_facecolor("#0f0f1a")
ax2.set_facecolor("#1a1a2e")

x      = np.arange(len(LABEL_ORDER))
width  = 0.35
bars_r = ax2.bar(x - width/2, acc_df["RoBERTa"], width, label="RoBERTa",
                 color="#5B8AF5", edgecolor="#0f0f1a", linewidth=1.2)
bars_v = ax2.bar(x + width/2, acc_df["VADER"],   width, label="VADER",
                 color="#F5A45B", edgecolor="#0f0f1a", linewidth=1.2)

for bars in (bars_r, bars_v):
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                 f"{h:.1%}", ha="center", va="bottom",
                 fontsize=11, color="white", fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels(LABEL_ORDER, fontsize=13, color="white")
ax2.set_ylabel("Per-class Accuracy", fontsize=12, color="white")
ax2.set_ylim(0, 1.12)
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax2.tick_params(colors="white")
ax2.spines[:].set_color("#333355")
ax2.legend(fontsize=12, facecolor="#1a1a2e", labelcolor="white", framealpha=0.8)
ax2.set_title(
    "Per-class Accuracy by Sentiment Label\n(RoBERTa vs VADER)",
    fontsize=14, color="white", fontweight="bold"
)

plt.tight_layout()
bar_path = os.path.join(RESULTS_DIR, "per_class_accuracy.png")
fig2.savefig(bar_path, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
plt.close(fig2)
print(f"✓ Saved per-class accuracy chart → {bar_path}")

print("\n✓ All done — outputs in ./" + RESULTS_DIR)