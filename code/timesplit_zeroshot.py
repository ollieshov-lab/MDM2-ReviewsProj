import kagglehub
import pandas as pd
import os
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# --- Load Data ---
path = kagglehub.dataset_download("jiashenliu/515k-hotel-reviews-data-in-europe")
csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(path, csv_file))
print(f"Loaded {len(df):,} rows")

df['review_text'] = df['Positive_Review'].fillna('') + ' ' + df['Negative_Review'].fillna('')
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

# --- Zero-shot topic list derived from unsupervised results ---
zeroshot_topics = [
    "wifi internet connection slow free",
    "noise noisy loud traffic street",
    "London location central tube underground",
    "tea coffee kettle machine room",
    "breakfast included expensive buffet choice",
    "staff friendly helpful service front desk",
    "Amsterdam tram centre canal city",
    "parking car park garage valet",
    "pool swimming spa rooftop indoor",
    "shower bathroom bath water pressure",
    "air conditioning heating AC hot cold",
    "Paris metro Eiffel French location",
    "Barcelona beach metro Catalonia",
    "bed comfortable mattress pillow sleep",
    "booking credit card charged overcharged",
    "room size small tiny space",
    "cleanliness dirty clean hygiene",
    "check-in check-out front desk waiting",
    "view balcony sea ocean window",
    "restaurant food dinner lunch meal",
]

def run_bertopic_zeroshot(season_df, season_name):
    print(f"\n{'='*50}\nRunning Zero-Shot BERTopic for: {season_name}\n{'='*50}")
    docs = season_df[REVIEW_COL].dropna().tolist()
    docs = [d for d in docs if d.strip()]
    print(f"  Reviews: {len(docs):,}")

    model = BERTopic(
        embedding_model=embedding_model,
        zeroshot_topic_list=zeroshot_topics,
        zeroshot_min_similarity=0.70,  # lower = more docs matched, higher = stricter
        language="multilingual",
        verbose=True,
    )
    topics, probs = model.fit_transform(docs)
    return model, model.get_topic_info(), docs, topics

os.makedirs('results_zeroshot', exist_ok=True)

summary_rows = []
for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
    model, info, docs, topics = run_bertopic_zeroshot(season_dfs[season], season)

    info['Season'] = season
    info.to_csv(f'results_zeroshot/{season}_topics.csv', index=False)

    # save docs with topic assignments
    docs_df = pd.DataFrame({'text': docs, 'topic': topics})
    docs_df.to_csv(f'results_zeroshot/{season}_docs.csv', index=False)
    print(f"Saved results_zeroshot/{season}_docs.csv")

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
print("\nDone — all results saved to ./results_zeroshot/")

print("\n\n===== ZERO-SHOT TOPICS PER SEASON =====")
for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
    info = pd.read_csv(f'results_zeroshot/{season}_topics.csv')
    print(f"\n{season}:")
    print(info[info['Topic'] != -1][['Topic', 'Count', 'Name']].head(20).to_string(index=False))