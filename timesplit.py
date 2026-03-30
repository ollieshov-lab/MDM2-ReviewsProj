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

def run_bertopic(season_df, season_name):
    print(f"\n{'='*50}\nRunning BERTopic for: {season_name}\n{'='*50}")
    docs = season_df[REVIEW_COL].dropna().tolist()
    docs = [d for d in docs if d.strip()]
    print(f"  Reviews: {len(docs):,}")
    model = BERTopic(embedding_model=embedding_model, language="multilingual", verbose=True)
    topics, probs = model.fit_transform(docs)
    return model, model.get_topic_info()

os.makedirs('results', exist_ok=True)

summary_rows = []
for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
    model, info = run_bertopic(season_dfs[season], season)

    # save full topic info for this season
    info['Season'] = season
    info.to_csv(f'results/{season}_topics.csv', index=False)
    print(f"Saved results/{season}_topics.csv")

    # accumulate summary row
    noise = int(info[info['Topic'] == -1]['Count'].values[0])
    total = int(info['Count'].sum())
    summary_rows.append({
        'Season': season,
        'Total_Reviews': total,
        'Noise_Reviews': noise,
        'Clustered_Reviews': total - noise,
        'Num_Topics': len(info[info['Topic'] != -1]),
    })

# save summary as csv too
pd.DataFrame(summary_rows).to_csv('results/summary.csv', index=False)
print("\nSaved results/summary.csv")
print("Done — all results saved to ./results/")