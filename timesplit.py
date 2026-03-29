
import kagglehub
import pandas as pd
import os
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# --- Load Data ---
path = kagglehub.dataset_download("jiashenliu/515k-hotel-reviews-data-in-europe")
print("Path to dataset files:", path)

csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(path, csv_file))
print(f"Loaded {len(df):,} rows")
print(df.columns.tolist())  # sanity check

# --- Combine Positive + Negative Reviews ---
df['review_text'] = df['Positive_Review'].fillna('') + ' ' + df['Negative_Review'].fillna('')
df['review_text'] = df['review_text'].str.strip()

REVIEW_COL = 'review_text'

# --- Time Split ---
df['Review_Date'] = pd.to_datetime(df['Review_Date'], errors='coerce')
df['Month'] = df['Review_Date'].dt.month

def assign_season(month):
    if month in [12, 1, 2]:
        return 'Winter (Dec-Feb)'
    elif month in [3, 4, 5]:
        return 'Spring (Mar-May)'
    elif month in [6, 7, 8]:
        return 'Summer (Jun-Aug)'
    else:
        return 'Autumn (Sep-Nov)'

df['Season'] = df['Month'].apply(assign_season)

season_dfs = {season: group.reset_index(drop=True) for season, group in df.groupby('Season')}

winter_df = season_dfs.get('Winter (Dec-Feb)')
spring_df = season_dfs.get('Spring (Mar-May)')
summer_df = season_dfs.get('Summer (Jun-Aug)')
autumn_df = season_dfs.get('Autumn (Sep-Nov)')

print(df['Season'].value_counts().sort_index())

# --- GPU Embedding Model ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

# --- BERTopic per Season ---
def run_bertopic(season_df, season_name):
    print(f"\n{'='*50}")
    print(f"Running BERTopic for: {season_name}")
    print(f"{'='*50}")

    docs = season_df[REVIEW_COL].dropna().tolist()
    # Filter out empty/whitespace strings
    docs = [d for d in docs if d.strip()]
    print(f"  Reviews: {len(docs):,}")

    if len(docs) < 10:
        print("  Not enough reviews, skipping.")
        return None, None

    topic_model = BERTopic(embedding_model=embedding_model, language="multilingual", verbose=True)
    topics, probs = topic_model.fit_transform(docs)

    info = topic_model.get_topic_info()
    print(info.head(10))

    return topic_model, info

winter_model,  winter_info  = run_bertopic(winter_df,  'Winter (Dec-Feb)')
spring_model,  spring_info  = run_bertopic(spring_df,  'Spring (Mar-May)')
summer_model,  summer_info  = run_bertopic(summer_df,  'Summer (Jun-Aug)')
autumn_model,  autumn_info  = run_bertopic(autumn_df,  'Autumn (Sep-Nov)')

# --- Compare Top Topics Across Seasons ---
print("\n\n===== TOP TOPICS PER SEASON =====")
for name, info in [('Winter', winter_info), ('Spring', spring_info),
                   ('Summer', summer_info), ('Autumn', autumn_info)]:
    if info is not None:
        print(f"\n{name}:")
        print(info[info['Topic'] != -1][['Topic', 'Count', 'Name']].head(10).to_string(index=False))

