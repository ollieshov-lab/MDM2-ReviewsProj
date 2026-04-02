import kagglehub
import pandas as pd
import os
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# --- Load Data ---
path = kagglehub.dataset_download("jiashenliu/515k-hotel-reviews-data-in-europe")
csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(path, csv_file))
print(f"Loaded {len(df):,} rows")

def clean_review(pos, neg):
    pos = str(pos).strip()
    neg = str(neg).strip()

    pos_invalid = pos.lower() in ["no positive", "nothing", "n/a", ""]
    neg_invalid = neg.lower() in ["no negative", "nothing", "n/a", ""]

    if pos_invalid and neg_invalid:
        return ""  # drop empty reviews
    elif pos_invalid:
        return neg
    elif neg_invalid:
        return pos
    else:
        return pos + " " + neg

df['review_text'] = df.apply(lambda x: clean_review(x['Positive_Review'], x['Negative_Review']), axis=1)
df['review_text'] = df['review_text'].str.strip()
REVIEW_COL = 'review_text'

df['Review_Date'] = pd.to_datetime(df['Review_Date'], errors='coerce')
df['Month'] = df['Review_Date'].dt.month

def assign_season(month):
    if month in [12, 1, 2]:   return 'Winter'
    elif month in [3, 4, 5]:  return 'Spring'
    elif month in [6, 7, 8]:  return 'Summer'
    else:                      return 'Autumn'

df['Season'] = df['Month'].apply(assign_season)
season_dfs = {s: g.reset_index(drop=True) for s, g in df.groupby('Season')}

embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

os.makedirs('results', exist_ok=True)
os.makedirs('results_zeroshot', exist_ok=True)

# --- Semantic name cleaner (free, no API) ---
EXTRA_STOPWORDS = {
    'hotel', 'room', 'the', 'and', 'was', 'is', 'it', 'no', 'not',
    'very', 'good', 'great', 'nice', 'bad', 'would', 'could', 'also',
    'get', 'got', 'us', 'we', 'our', 'my', 'had', 'has', 'have',
    'been', 'were', 'are', 'con', 'est', 'que', 'les', 'une', 'des',
}
ALL_STOPWORDS = ENGLISH_STOP_WORDS.union(EXTRA_STOPWORDS)

def get_semantic_name(topic_name: str) -> str:
    words = topic_name.split('_')[1:]  # strip topic number
    seen = set()
    cleaned = []
    for w in words:
        if w.lower() not in ALL_STOPWORDS and w.lower() not in seen and len(w) > 2:
            cleaned.append(w)
            seen.add(w.lower())
    return ' '.join(cleaned[:4])

# ════════════════════════════════════════════════════════════
# STAGE 1 — Unsupervised BERTopic, extract top 100 per season
# ════════════════════════════════════════════════════════════
def run_unsupervised(season_df, season_name):
    print(f"\n{'='*50}\nStage 1 — Unsupervised BERTopic: {season_name}\n{'='*50}")
    docs = season_df[REVIEW_COL].dropna().tolist()
    docs = [d for d in docs if d.strip()]
    print(f"  Reviews: {len(docs):,}")
    model = BERTopic(embedding_model=embedding_model, language="multilingual", verbose=True)
    topics, probs = model.fit_transform(docs)
    info = model.get_topic_info()
    return model, info, docs

# collect raw topic names across all seasons before naming
raw_topic_names = set()

for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
    model, info, docs = run_unsupervised(season_dfs[season], season)

    info['Season'] = season
    info.to_csv(f'results/{season}_topics.csv', index=False)

    topics_assigned, _ = model.transform(docs)
    pd.DataFrame({'text': docs, 'topic': topics_assigned}).to_csv(
        f'results/{season}_docs.csv', index=False)
    print(f"  Saved results/{season}_topics.csv")

    top100 = info[info['Topic'] != -1].head(100)
    for _, row in top100.iterrows():
        raw_topic_names.add(row['Name'])

print(f"\n✓ Stage 1 complete — {len(raw_topic_names)} unique raw topics collected")

# deduplicate and clean topic names
discovered_topics = set()
mapping_rows = []
for raw in sorted(raw_topic_names):
    semantic = get_semantic_name(raw)
    if semantic:  # skip empty results
        discovered_topics.add(semantic)
        mapping_rows.append({'raw': raw, 'semantic': semantic})
        print(f"  {raw:55s} → {semantic}")

pd.DataFrame(mapping_rows).to_csv('results/topic_name_mapping.csv', index=False)
print(f"\n✓ {len(discovered_topics)} unique semantic topics after cleaning")
print("Saved results/topic_name_mapping.csv")

# ════════════════════════════════════════════════════════════
# STAGE 2 — Zero-shot BERTopic using semantic topic names
# ════════════════════════════════════════════════════════════
zeroshot_topic_list = sorted(discovered_topics)
print(f"\nUsing {len(zeroshot_topic_list)} topics for zero-shot pass")

pd.DataFrame({'topic': zeroshot_topic_list}).to_csv('results/discovered_zeroshot_topics.csv', index=False)
print("Saved results/discovered_zeroshot_topics.csv")

def run_zeroshot(season_df, season_name, topic_list):
    print(f"\n{'='*50}\nStage 2 — Zero-Shot BERTopic: {season_name}\n{'='*50}")
    docs = season_df[REVIEW_COL].dropna().tolist()
    docs = [d for d in docs if d.strip()]
    print(f"  Reviews: {len(docs):,}, Topics: {len(topic_list)}")

    model = BERTopic(
        embedding_model=embedding_model,
        zeroshot_topic_list=topic_list,
        zeroshot_min_similarity=0.70,
        language="multilingual",
        verbose=True,
    )
    topics, probs = model.fit_transform(docs)
    info = model.get_topic_info()
    return model, info, docs, topics

summary_rows = []
for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
    model, info, docs, topics = run_zeroshot(season_dfs[season], season, zeroshot_topic_list)

    info['Season'] = season
    info.to_csv(f'results_zeroshot/{season}_topics.csv', index=False)

    pd.DataFrame({'text': docs, 'topic': topics}).to_csv(
        f'results_zeroshot/{season}_docs.csv', index=False)
    print(f"  Saved results_zeroshot/{season}_docs.csv")

    noise = int(info[info['Topic'] == -1]['Count'].values[0]) if -1 in info['Topic'].values else 0
    total = int(info['Count'].sum())
    summary_rows.append({
        'Season': season,
        'Total_Reviews': total,
        'Noise_Reviews': noise,
        'Clustered_Reviews': total - noise,
        'Num_Topics': len(info[info['Topic'] != -1]),
    })

pd.DataFrame(summary_rows).to_csv('results_zeroshot/summary.csv', index=False)
print("\n✓ Stage 2 complete — all results saved to ./results_zeroshot/")

# --- Print top 20 per season ---
print("\n\n===== ZERO-SHOT TOPICS PER SEASON (top 20) =====")
for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
    info = pd.read_csv(f'results_zeroshot/{season}_topics.csv')
    print(f"\n{season}:")
    print(info[info['Topic'] != -1][['Topic', 'Count', 'Name']].head(20).to_string(index=False))