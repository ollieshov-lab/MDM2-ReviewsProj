import kagglehub
import pandas as pd
import os
import numpy as np
import torch
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ── Load Data ─────────────────────────────────────────────────────────────────
path = kagglehub.dataset_download("jiashenliu/515k-hotel-reviews-data-in-europe")
csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(path, csv_file))
print(f"Loaded {len(df):,} rows")

# ── Clean Reviews ─────────────────────────────────────────────────────────────
def clean_review(pos, neg):
    pos, neg = str(pos).strip(), str(neg).strip()
    pos_invalid = pos.lower() in ["no positive", "nothing", "n/a", ""]
    neg_invalid = neg.lower() in ["no negative", "nothing", "n/a", ""]
    if pos_invalid and neg_invalid: return ""
    elif pos_invalid: return neg
    elif neg_invalid: return pos
    return pos + " " + neg

df['review_text'] = df.apply(
    lambda x: clean_review(x['Positive_Review'], x['Negative_Review']), axis=1
).str.strip()

# ── Season Split ──────────────────────────────────────────────────────────────
df['Review_Date'] = pd.to_datetime(df['Review_Date'], errors='coerce')
df['Month'] = df['Review_Date'].dt.month

def assign_season(m):
    if m in [12, 1, 2]: return 'Winter'
    elif m in [3, 4, 5]: return 'Spring'
    elif m in [6, 7, 8]: return 'Summer'
    return 'Autumn'

df['Season'] = df['Month'].apply(assign_season)
season_dfs = {s: g.reset_index(drop=True) for s, g in df.groupby('Season')}

# ── Models ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=0 if DEVICE == "cuda" else -1,
    truncation=True,
    max_length=128,
    model_kwargs={"use_safetensors": True},
)
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}

REPR_DOCS_PER_TOPIC = 50   # sentiment runs on this many docs per topic only

# ── Sentiment on representative docs only ─────────────────────────────────────
def sentiment_from_repr_docs(model: BERTopic, topic_info: pd.DataFrame) -> pd.DataFrame:
    LABELS = ["Negative", "Neutral", "Positive"]
    rows = []

    repr_docs = model.get_representative_docs()   # dict: topic_id -> [doc, doc, ...]

    for tid in topic_info['Topic']:
        if tid == -1:
            rows.append({'Topic': -1, 'Neg': None, 'Neu': None, 'Pos': None,
                         'Sentiment': None, 'Net_Sentiment': None})
            continue

        docs_for_topic = repr_docs.get(tid, [])[:REPR_DOCS_PER_TOPIC]

        if not docs_for_topic:
            rows.append({'Topic': tid, 'Neg': None, 'Neu': None, 'Pos': None,
                         'Sentiment': None, 'Net_Sentiment': None})
            continue

        results = sentiment_pipe(docs_for_topic, batch_size=REPR_DOCS_PER_TOPIC)

        scores = np.zeros((len(results), 3))
        for i, r in enumerate(results):
            idx = LABEL_MAP[r['label'].lower()]
            scores[i, idx] = r['score']
            remainder = (1 - r['score']) / 2
            for j in range(3):
                if j != idx:
                    scores[i, j] = remainder

        mean_scores = scores.mean(axis=0)
        dominant = LABELS[int(np.argmax(mean_scores))]
        net = float(mean_scores[2] - mean_scores[0])

        rows.append({
            'Topic': tid,
            'Neg':          round(float(mean_scores[0]), 4),
            'Neu':          round(float(mean_scores[1]), 4),
            'Pos':          round(float(mean_scores[2]), 4),
            'Sentiment':    dominant,
            'Net_Sentiment': round(net, 4),
        })

    return pd.DataFrame(rows)

# ── Output Dir ────────────────────────────────────────────────────────────────
os.makedirs('results', exist_ok=True)

# ── Main Loop ─────────────────────────────────────────────────────────────────
for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
    print(f"\n{'='*55}\n  {season}\n{'='*55}")

    docs = season_dfs[season]['review_text'].dropna().tolist()
    docs = [d for d in docs if d.strip()]
    print(f"  Reviews: {len(docs):,}")

    # 1. Embed
    print("  Embedding...")
    embeddings = embedding_model.encode(
        docs, show_progress_bar=True, batch_size=512
    )

    # 2. BERTopic — bump nr_repr_docs so sentiment has more signal per topic
    print("  Fitting BERTopic...")
    model = BERTopic(
    embedding_model=embedding_model,
    language="multilingual",
    verbose=False,)
    model.nr_repr_docs = REPR_DOCS_PER_TOPIC
    topics, _ = model.fit_transform(docs, embeddings)
    topic_info = model.get_topic_info()

    # 3. Sentiment on representative docs only (fast)
    print(f"  Running sentiment on {REPR_DOCS_PER_TOPIC} repr docs × {len(topic_info)-1} topics...")
    sent_df = sentiment_from_repr_docs(model, topic_info)

    # 4. Merge and save
    out = topic_info.merge(sent_df, on='Topic', how='left')
    out['Season'] = season

    out_path = f'results/{season}_topics.csv'
    out.to_csv(out_path, index=False)
    print(f"  ✓ Saved {out_path}  ({len(out)} topics)")

    preview = out[out['Topic'] != -1][
        ['Topic', 'Count', 'Name', 'Sentiment', 'Net_Sentiment']
    ].head(15)
    print(preview.to_string(index=False))
    safe_name = season.replace(" ", "_")
    model_path = f'results/{safe_name}_bertopic_model'
    
    # You can save using the "safetensors" format (recommended) 
    # or the standard pickle format.
    model.save(model_path, serialization="safetensors", 
               save_ctfidf=True, save_embedding_model=False)
    
    print(f"  ✓ Saved Model to {model_path}")


print("\n✓ All done — 4 CSVs in ./results/")


# import kagglehub
# import pandas as pd
# import os
# import numpy as np
# import torch
# from bertopic import BERTopic
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline

# # ── Load Data ─────────────────────────────────────────────────────────────────
# path = kagglehub.dataset_download("jiashenliu/515k-hotel-reviews-data-in-europe")
# csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
# df = pd.read_csv(os.path.join(path, csv_file))
# print(f"Loaded {len(df):,} rows")

# # ── Clean Reviews ─────────────────────────────────────────────────────────────
# def clean_review(pos, neg):
#     pos, neg = str(pos).strip(), str(neg).strip()
#     pos_invalid = pos.lower() in ["no positive", "nothing", "n/a", ""]
#     neg_invalid = neg.lower() in ["no negative", "nothing", "n/a", ""]
#     if pos_invalid and neg_invalid: return ""
#     elif pos_invalid: return neg
#     elif neg_invalid: return pos
#     return pos + " " + neg

# df['review_text'] = df.apply(
#     lambda x: clean_review(x['Positive_Review'], x['Negative_Review']), axis=1
# ).str.strip()

# # ── Season Split ──────────────────────────────────────────────────────────────
# df['Review_Date'] = pd.to_datetime(df['Review_Date'], errors='coerce')
# df['Month'] = df['Review_Date'].dt.month

# def assign_season(m):
#     if m in [12, 1, 2]: return 'Winter'
#     elif m in [3, 4, 5]: return 'Spring'
#     elif m in [6, 7, 8]: return 'Summer'
#     return 'Autumn'

# df['Season'] = df['Month'].apply(assign_season)
# season_dfs = {s: g.reset_index(drop=True) for s, g in df.groupby('Season')}

# # ── Zero-shot Topic List ──────────────────────────────────────────────────────
# zeroshot_topic_list = [
#     "Amsterdam city centre location", "Paris city centre location",
#     "Barcelona city centre location", "Vienna city centre location",
#     "London central location", "Milan Duomo area location",
#     "proximity to metro station", "tram stop nearby", "train station access",
#     "underground or tube station nearby", "airport shuttle service",
#     "taxi and airport transfer", "public transport links",
#     "walking distance to attractions", "city centre distance",
#     "Eiffel Tower proximity", "Louvre and Notre Dame proximity",
#     "Champs Elysees proximity", "Sagrada Familia proximity",
#     "Barcelona Nou Camp proximity", "Ramblas proximity",
#     "Amsterdam canals and Dam Square", "Milan Duomo and shopping",
#     "Schonbrunn Palace Vienna", "Vienna opera house proximity",
#     "Hyde Park proximity", "Buckingham Palace proximity",
#     "Oxford Street shopping", "Covent Garden and theatres",
#     "Trafalgar Square proximity", "Natural History or science museum proximity",
#     "Kensington museums proximity", "Wembley Stadium event access",
#     "O2 Arena event access", "Canary Wharf location",
#     "Euston or Kings Cross proximity", "Paddington and Heathrow Express access",
#     "Earls Court and Kensington area", "Greenwich location",
#     "Shoreditch or east London vibe", "Marylebone or Baker Street area",
#     "Camden Market proximity", "Liverpool Street proximity",
#     "Waterloo or South Bank proximity", "Montmartre or Montparnasse proximity",
#     "Gare du Nord or Gare de Lyon proximity", "Lyon train station proximity",
#     "Barcelona Placa Catalunya proximity", "Barcelona beach proximity",
#     "Barcelona Gracia neighbourhood", "small room size", "spacious room",
#     "comfortable bed", "hard or uncomfortable mattress", "pillow quality",
#     "bathroom quality", "shower pressure and temperature",
#     "hot water availability", "air conditioning and heating",
#     "room noise and soundproofing", "room view", "river or canal view",
#     "city skyline view", "room cleanliness", "room decor and design",
#     "modern room style", "dated or tired room decor", "balcony or terrace",
#     "window and ventilation", "room lighting", "dark room lighting",
#     "minibar items and pricing", "Nespresso or coffee machine in room",
#     "iron and ironing board", "in-room safe", "towel quality",
#     "toiletries and amenities", "slippers and bathrobe",
#     "fridge or refrigerator in room", "TV channels and quality",
#     "Wi-Fi internet connection speed", "USB sockets and plug adapters",
#     "room photos not matching reality", "carpet stains or dirty carpets",
#     "dusty or dirty room", "mould or damp problems",
#     "insects or pest infestation", "bad smell or drainage issues",
#     "friendly and helpful staff", "rude or unhelpful reception",
#     "concierge service", "check-in speed and efficiency",
#     "late or early check-in availability", "check-out process",
#     "housekeeping service", "customer service quality",
#     "language barrier with staff", "staff knowledge of local area",
#     "breakfast quality and variety", "buffet breakfast",
#     "cooked full breakfast", "croissants and pastries", "eggs at breakfast",
#     "breakfast pricing", "gluten-free breakfast options",
#     "vegetarian or vegan food options", "bar and cocktails",
#     "restaurant food quality", "coffee and tea facilities", "room service",
#     "drinks prices at bar", "afternoon tea",
#     "juice and fresh fruit at breakfast", "swimming pool",
#     "spa and massage treatments", "gym and fitness equipment",
#     "sauna and steam room", "executive lounge or club access",
#     "rooftop bar or terrace", "bike rental", "laundry facilities",
#     "parking facilities", "luggage storage service",
#     "printing or boarding pass service", "value for money",
#     "overpriced or expensive hotel", "expensive breakfast pricing",
#     "city tax charges", "hidden charges or billing issues",
#     "fire alarm disturbance", "noise from street or neighbours",
#     "construction or drilling noise", "slow or broken lift or elevator",
#     "key card problems", "theft or stolen belongings",
#     "double booking or wrong room type", "building renovation disruption",
#     "early morning disturbance by housekeeping", "plumbing or toilet issues",
#     "cold room temperature", "overheating room",
#     "birthday or anniversary celebration", "honeymoon stay",
#     "room upgrade experience", "junior suite or suite",
#     "executive room or club room", "welcome gift or arrival treat",
#     "loyalty programme benefits", "disabled or wheelchair accessibility",
#     "pet-friendly accommodation", "water bottles provided in room",
#     "boutique hotel atmosphere", "historic building charm",
#     "overall exceptional stay", "overall disappointing stay",
# ]

# print(f"Zero-shot topics: {len(zeroshot_topic_list)}")

# # ── Models ────────────────────────────────────────────────────────────────────
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {DEVICE}")

# embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

# sentiment_pipe = pipeline(
#     "sentiment-analysis",
#     model="cardiffnlp/twitter-roberta-base-sentiment-latest",
#     device=0 if DEVICE == "cuda" else -1,
#     truncation=True,
#     max_length=128,
#     model_kwargs={"use_safetensors": True},
# )
# LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}

# SAMPLE_PER_TOPIC = 50    # random sample size — large enough for stable signal
# RANDOM_SEED      = 42

# # ── Sentiment on random sample of assigned docs ───────────────────────────────
# def sentiment_from_assigned_docs(
#     docs: list[str],
#     topics: list[int],
#     topic_info: pd.DataFrame,
#     sample_n: int = SAMPLE_PER_TOPIC,
# ) -> pd.DataFrame:
#     """
#     For each topic, randomly sample up to `sample_n` docs that were actually
#     assigned to it, then run sentiment. This gives a far more accurate signal
#     than using BERTopic's centroid-nearest representative docs.
#     """
#     LABELS    = ["Negative", "Neutral", "Positive"]
#     topic_arr = np.array(topics)
#     docs_arr  = np.array(docs, dtype=object)
#     rng       = np.random.default_rng(RANDOM_SEED)
#     rows      = []

#     for tid in topic_info['Topic']:
#         if tid == -1:
#             rows.append({'Topic': -1, 'Neg': None, 'Neu': None, 'Pos': None,
#                          'Sentiment': None, 'Net_Sentiment': None})
#             continue

#         idx = np.where(topic_arr == tid)[0]
#         if len(idx) == 0:
#             rows.append({'Topic': tid, 'Neg': None, 'Neu': None, 'Pos': None,
#                          'Sentiment': None, 'Net_Sentiment': None})
#             continue

#         # random sample (without replacement, up to sample_n)
#         chosen   = rng.choice(idx, size=min(sample_n, len(idx)), replace=False)
#         sampled  = docs_arr[chosen].tolist()

#         results  = sentiment_pipe(sampled, batch_size=32)

#         scores   = np.zeros((len(results), 3))
#         for i, r in enumerate(results):
#             label_idx         = LABEL_MAP[r['label'].lower()]
#             scores[i, label_idx] = r['score']
#             remainder         = (1 - r['score']) / 2
#             for j in range(3):
#                 if j != label_idx:
#                     scores[i, j] = remainder

#         mean_scores = scores.mean(axis=0)
#         dominant    = LABELS[int(np.argmax(mean_scores))]
#         net         = float(mean_scores[2] - mean_scores[0])

#         rows.append({
#             'Topic':          tid,
#             'Neg':            round(float(mean_scores[0]), 4),
#             'Neu':            round(float(mean_scores[1]), 4),
#             'Pos':            round(float(mean_scores[2]), 4),
#             'Sentiment':      dominant,
#             'Net_Sentiment':  round(net, 4),
#         })

#     return pd.DataFrame(rows)

# # ── Output Dir ────────────────────────────────────────────────────────────────
# os.makedirs('results_zeroshot_seasons', exist_ok=True)

# # ── Main Loop ─────────────────────────────────────────────────────────────────
# summary_rows = []

# for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
#     print(f"\n{'='*55}\n  {season}\n{'='*55}")

#     docs = season_dfs[season]['review_text'].dropna().tolist()
#     docs = [d for d in docs if d.strip()]
#     print(f"  Reviews: {len(docs):,}")

#     # 1. Embed
#     print("  Embedding...")
#     embeddings = embedding_model.encode(docs, show_progress_bar=True, batch_size=512)

#     # 2. Zero-shot BERTopic
#     print("  Fitting zero-shot BERTopic...")
#     model = BERTopic(
#         embedding_model=embedding_model,
#         zeroshot_topic_list=zeroshot_topic_list,
#         zeroshot_min_similarity=0.70,
#         language="multilingual",
#         verbose=False,
#     )
#     topics, _ = model.fit_transform(docs, embeddings)
#     topic_info = model.get_topic_info()

#     # 3. Sentiment on random sample of ASSIGNED docs (not repr docs)
#     print(f"  Sentiment ({SAMPLE_PER_TOPIC} random docs × {len(topic_info)-1} topics)...")
#     sent_df = sentiment_from_assigned_docs(docs, topics, topic_info)

#     # 4. Merge and save
#     out = topic_info.merge(sent_df, on='Topic', how='left')
#     out['Season'] = season

#     out_path = f'results_zeroshot_seasons/{season}_topics.csv'
#     out.to_csv(out_path, index=False)
#     print(f"  ✓ Saved {out_path}  ({len(out)} topics)")

#     noise     = int(out[out['Topic'] == -1]['Count'].values[0]) if -1 in out['Topic'].values else 0
#     total     = int(out['Count'].sum())
#     clustered = total - noise

#     summary_rows.append({
#         'Season':            season,
#         'Total_Reviews':     total,
#         'Noise_Reviews':     noise,
#         'Clustered_Reviews': clustered,
#         'Pct_Clustered':     round(100 * clustered / total, 1),
#         'Num_Topics':        len(out[out['Topic'] != -1]),
#     })

#     preview = out[out['Topic'] != -1][
#         ['Topic', 'Count', 'Name', 'Sentiment', 'Net_Sentiment']
#     ].head(20)
#     print(preview.to_string(index=False))

#     safe_name = season.replace(" ", "_")
#     model_path = f'results_zeroshot_seasons/{safe_name}_bertopic_model'
    
#     # You can save using the "safetensors" format (recommended) 
#     # or the standard pickle format.
#     model.save(model_path, serialization="safetensors", 
#                save_ctfidf=True, save_embedding_model=False)
    
#     print(f"  ✓ Saved Model to {model_path}")

# summary_df = pd.DataFrame(summary_rows)
# summary_df.to_csv('results_zeroshot_seasons/summary.csv', index=False)
# print("\n\n===== SUMMARY =====")
# print(summary_df.to_string(index=False))
# print("\n✓ All done — results in ./results_zeroshot/")