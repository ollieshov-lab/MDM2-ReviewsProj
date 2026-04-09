import kagglehub
import pandas as pd
import os
import ast
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
    if pos_invalid and neg_invalid:
        return ""
    elif pos_invalid:
        return neg
    elif neg_invalid:
        return pos
    return pos + " " + neg

df['review_text'] = df.apply(
    lambda x: clean_review(x['Positive_Review'], x['Negative_Review']), axis=1
).str.strip()

# ── Extract Traveler Type from Tags ───────────────────────────────────────────
def extract_traveler_type(tags_str):
    """
    Parse the 'Tags' column (string representation of a list) and return
    one of: 'Couple', 'Solo traveler', 'Family with young children', 'Group'.
    Returns None if none of the patterns match.
    """
    if not isinstance(tags_str, str) or tags_str.strip() == "":
        return None
    try:
        # Convert string like "[' Leisure trip ', ' Couple ']" to a list
        tags_list = ast.literal_eval(tags_str)
    except (SyntaxError, ValueError):
        return None

    # Normalise each tag (strip spaces, lower case)
    tags_lower = [tag.strip().lower() for tag in tags_list]

    # Priority order: if multiple match, the first found is used
    if any('couple' in tag for tag in tags_lower):
        return 'Couple'
    if any('solo' in tag or 'single' in tag for tag in tags_lower):
        return 'Solo traveler'
    if any('family' in tag or 'child' in tag or 'kid' in tag for tag in tags_lower):
        return 'Family with young children'
    if any('group' in tag or 'friend' in tag for tag in tags_lower):
        return 'Group'
    return None

df['TravelerType'] = df['Tags'].apply(extract_traveler_type)

# Drop rows where we could not determine a traveler type
initial_len = len(df)
df = df.dropna(subset=['TravelerType'])
print(f"Dropped {initial_len - len(df):,} rows without a recognisable traveler type")

# Group by traveler type
traveler_types = ['Couple', 'Solo traveler', 'Family with young children', 'Group']
traveler_dfs = {t: df[df['TravelerType'] == t].reset_index(drop=True) for t in traveler_types}
for t, subdf in traveler_dfs.items():
    print(f"{t}: {len(subdf):,} reviews")

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
for traveler_type in traveler_types:
    print(f"\n{'='*55}\n  {traveler_type}\n{'='*55}")

    docs = traveler_dfs[traveler_type]['review_text'].dropna().tolist()
    docs = [d for d in docs if d.strip()]
    print(f"  Reviews: {len(docs):,}")

    if len(docs) == 0:
        print("  No reviews, skipping.")
        continue

    # 1. Embed
    print("  Embedding...")
    embeddings = embedding_model.encode(
        docs, show_progress_bar=True, batch_size=512
    )

    # 2. BERTopic
    print("  Fitting BERTopic...")
    model = BERTopic(
        embedding_model=embedding_model,
        language="multilingual",
        verbose=False,
    )
    model.nr_repr_docs = REPR_DOCS_PER_TOPIC
    topics, _ = model.fit_transform(docs, embeddings)
    topic_info = model.get_topic_info()

    # 3. Sentiment on representative docs only
    print(f"  Running sentiment on {REPR_DOCS_PER_TOPIC} repr docs × {len(topic_info)-1} topics...")
    sent_df = sentiment_from_repr_docs(model, topic_info)

    # 4. Merge and save
    out = topic_info.merge(sent_df, on='Topic', how='left')
    out['TravelerType'] = traveler_type

    # Clean file name: replace spaces with underscores
    safe_name = traveler_type.replace(' ', '_')
    out_path = f'results/{safe_name}_topics.csv'
    out.to_csv(out_path, index=False)
    print(f"  ✓ Saved {out_path}  ({len(out)} topics)")

    preview = out[out['Topic'] != -1][
        ['Topic', 'Count', 'Name', 'Sentiment', 'Net_Sentiment']
    ].head(15)
    print(preview.to_string(index=False))
    safe_name = traveler_type.replace(" ", "_")
    model_path = f'results/{safe_name}_bertopic_model'
    
    # You can save using the "safetensors" format (recommended) 
    # or the standard pickle format.
    model.save(model_path, serialization="safetensors", 
               save_ctfidf=True, save_embedding_model=False)
    
    print(f"  ✓ Saved Model to {model_path}")

print("\n✓ All done — 4 CSVs in ./results/")


# zeroshot_topic_list = [
#     # ── WiFi & Connectivity ──────────────────────────────────────────────────
#     "free WiFi speed and reliability",
#     "slow or weak WiFi signal",
#     "WiFi not working in room",

#     # ── Breakfast ────────────────────────────────────────────────────────────
#     "breakfast quality and variety",
#     "buffet breakfast with hot and cold options",
#     "cooked full breakfast with eggs and bacon",
#     "continental breakfast with pastries",
#     "breakfast included in room rate",
#     "breakfast not included and expensive",
#     "breakfast too small or poor choice",
#     "gluten-free or dietary breakfast options",

#     # ── In-Room Beverages ────────────────────────────────────────────────────
#     "tea and coffee making facilities in room",
#     "Nespresso or coffee machine in room",
#     "no kettle or tea facilities in room",

#     # ── Bed & Sleep Quality ──────────────────────────────────────────────────
#     "comfortable bed and mattress",
#     "uncomfortable or hard mattress",
#     "soft or overly firm mattress",
#     "twin beds pushed together instead of double",
#     "twin room booked but given double",
#     "pillows and bedding quality",

#     # ── Room Size & Layout ───────────────────────────────────────────────────
#     "small or cramped room size",
#     "spacious and well-sized room",
#     "room smaller than photos suggested",
#     "basement room with no window",

#     # ── Shower & Bathroom ────────────────────────────────────────────────────
#     "shower pressure and water temperature",
#     "powerful shower head",
#     "bath or bathtub availability",
#     "bathroom size and quality",
#     "bathroom cleanliness",
#     "toilet and plumbing issues",

#     # ── Room Cleanliness & Maintenance ───────────────────────────────────────
#     "room cleanliness and tidiness",
#     "dirty carpets or stains",
#     "mould or damp in room",
#     "bad smell or drainage odour",
#     "sewage or musty smell in room",
#     "cigarette smoke smell in non-smoking room",
#     "smoking policy and designated areas",

#     # ── Air Conditioning & Temperature ───────────────────────────────────────
#     "air conditioning working and effective",
#     "no air conditioning in room",
#     "room too hot or overheating",
#     "room too cold or no heating",
#     "temperature control in room",

#     # ── Noise & Soundproofing ────────────────────────────────────────────────
#     "noise from street or traffic at night",
#     "noise from other guests or thin walls",
#     "noisy hotel corridors",
#     "construction or drilling noise nearby",
#     "fire alarm disturbance during stay",
#     "quiet room and good soundproofing",

#     # ── Staff ────────────────────────────────────────────────────────────────
#     "friendly and helpful staff",
#     "rude or unhelpful staff",
#     "welcoming and attentive reception team",
#     "front desk efficiency and professionalism",
#     "concierge service and local recommendations",
#     "housekeeping quality and frequency",
#     "early morning disturbance by housekeeping",
#     "reception staff going above and beyond",

#     # ── Check-in & Billing ───────────────────────────────────────────────────
#     "smooth and fast check-in process",
#     "late or early check-in availability",
#     "credit card charged incorrectly or twice",
#     "security deposit hold on credit card",
#     "billing issues or unexpected charges",
#     "key card problems or room access issues",

#     # ── City Location – Amsterdam ────────────────────────────────────────────
#     "Amsterdam city centre location",
#     "Amsterdam canals and Dam Square proximity",
#     "Amsterdam tram stop outside hotel",
#     "Amsterdam Centraal station access",

#     # ── City Location – Paris ────────────────────────────────────────────────
#     "Paris city centre location",
#     "Paris metro station nearby",
#     "Eiffel Tower proximity or view",
#     "Louvre and Notre Dame walking distance",
#     "Champs Elysees or Arc de Triomphe proximity",
#     "Gare du Nord or Gare de Lyon proximity",
#     "Montmartre neighbourhood location",

#     # ── City Location – Barcelona ────────────────────────────────────────────
#     "Barcelona city centre location",
#     "La Rambla and Placa Catalunya proximity",
#     "Sagrada Familia proximity",
#     "Barcelona beach proximity",
#     "Barcelona Sants station access",
#     "Barcelona metro access",

#     # ── City Location – Vienna ────────────────────────────────────────────────
#     "Vienna city centre location",
#     "Vienna Opera House proximity",
#     "Schonbrunn Palace proximity",
#     "Vienna U-Bahn metro station nearby",

#     # ── City Location – London ────────────────────────────────────────────────
#     "London central location",
#     "London Underground tube station nearby",
#     "Paddington and Heathrow Express access",
#     "Hyde Park proximity",
#     "Kensington museums and Royal Albert Hall",
#     "Oxford Street shopping proximity",
#     "Covent Garden and West End theatres",
#     "Wembley Stadium and Arena event access",
#     "O2 Arena concert venue proximity",
#     "Canary Wharf and DLR location",
#     "Euston or Kings Cross station proximity",
#     "Earls Court and Kensington area",

#     # ── City Location – Milan ────────────────────────────────────────────────
#     "Milan city centre and Duomo proximity",
#     "Milan Centrale train station access",
#     "Milan metro station nearby",

#     # ── Public Transport (General) ───────────────────────────────────────────
#     "tram stop directly outside hotel",
#     "metro or subway station close to hotel",
#     "train station walking distance",
#     "public transport links from hotel",
#     "airport shuttle or transfer service",
#     "taxi availability at hotel",

#     # ── Pool & Water Facilities ──────────────────────────────────────────────
#     "swimming pool at hotel",
#     "rooftop pool with views",
#     "indoor swimming pool",
#     "pool booking slots or timed sessions",
#     "pool too small or too cold",

#     # ── Wellness & Spa ───────────────────────────────────────────────────────
#     "spa and massage treatments",
#     "sauna and steam room",
#     "gym and fitness equipment",
#     "wellness facilities closed or unavailable",

#     # ── Rooftop & Outdoor Spaces ─────────────────────────────────────────────
#     "rooftop terrace or bar with views",
#     "balcony or private terrace",
#     "city skyline or landmark view from room",
#     "river or canal view from room",

#     # ── Bar & Drinks ─────────────────────────────────────────────────────────
#     "hotel bar and cocktail menu",
#     "expensive drinks prices at bar",
#     "good range of beers and cocktails",

#     # ── Restaurant & Food ────────────────────────────────────────────────────
#     "hotel restaurant food quality",
#     "room service availability and quality",
#     "afternoon tea offering",

#     # ── Room Amenities & Equipment ───────────────────────────────────────────
#     "fridge or minibar in room",
#     "no fridge in room",
#     "iron and ironing board in room",
#     "in-room safe",
#     "TV channels and quality",
#     "complimentary bottled water in room",
#     "toiletries and bathroom amenities",
#     "towel quality and replacement",
#     "slippers and bathrobe provided",

#     # ── Parking ──────────────────────────────────────────────────────────────
#     "hotel parking facilities",
#     "expensive parking charges",
#     "no on-site parking available",
#     "valet parking service",

#     # ── Special Occasions ────────────────────────────────────────────────────
#     "birthday or anniversary celebration",
#     "honeymoon or romantic couple stay",
#     "birthday cake or card left in room",
#     "champagne or flowers on arrival",
#     "welcome cookies or complimentary treat on arrival",
#     "room upgrade for special occasion",

#     # ── Value & Pricing ──────────────────────────────────────────────────────
#     "good value for money",
#     "overpriced or expensive for quality offered",
#     "city tax or hidden charges",
#     "hotel star rating not matching expectations",

#     # ── Lift & Accessibility ─────────────────────────────────────────────────
#     "lift or elevator slow or broken",
#     "disabled or wheelchair accessible facilities",

#     # ── Overall Experience ───────────────────────────────────────────────────
#     "overall exceptional and highly recommended stay",
#     "overall disappointing stay",
#     "boutique hotel atmosphere and charm",
#     "historic or characterful building",
#     "modern and newly renovated hotel",
#     "clean and well-maintained hotel",
#     "best hotel ever visited",
# ]

# import kagglehub
# import pandas as pd
# import os
# import ast
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

# # ── Extract Traveler Type (TAG SPLIT) ─────────────────────────────────────────
# def extract_traveler_type(tags_str):
#     if not isinstance(tags_str, str) or tags_str.strip() == "":
#         return None
#     try:
#         tags_list = ast.literal_eval(tags_str)
#     except:
#         return None

#     tags_lower = [tag.strip().lower() for tag in tags_list]

#     if any('couple' in tag for tag in tags_lower):
#         return 'Couple'
#     if any('solo' in tag or 'single' in tag for tag in tags_lower):
#         return 'Solo traveler'
#     if any('family' in tag or 'child' in tag for tag in tags_lower):
#         return 'Family with young children'
#     if any('group' in tag or 'friend' in tag for tag in tags_lower):
#         return 'Group'
#     return None

# df['TravelerType'] = df['Tags'].apply(extract_traveler_type)
# df = df.dropna(subset=['TravelerType'])

# traveler_types = ['Couple', 'Solo traveler', 'Family with young children', 'Group']
# traveler_dfs = {t: df[df['TravelerType'] == t].reset_index(drop=True) for t in traveler_types}

# # ── Zero-shot Topic List ──────────────────────────────────────────────────────

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

# SAMPLE_PER_TOPIC = 50
# RANDOM_SEED = 42

# # ── Sentiment Function (same as seasons) ──────────────────────────────────────
# def sentiment_from_assigned_docs(docs, topics, topic_info, sample_n=SAMPLE_PER_TOPIC):
#     LABELS = ["Negative", "Neutral", "Positive"]
#     topic_arr = np.array(topics)
#     docs_arr = np.array(docs, dtype=object)
#     rng = np.random.default_rng(RANDOM_SEED)
#     rows = []

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

#         chosen = rng.choice(idx, size=min(sample_n, len(idx)), replace=False)
#         sampled = docs_arr[chosen].tolist()

#         results = sentiment_pipe(sampled, batch_size=32)

#         scores = np.zeros((len(results), 3))
#         for i, r in enumerate(results):
#             label_idx = LABEL_MAP[r['label'].lower()]
#             scores[i, label_idx] = r['score']
#             remainder = (1 - r['score']) / 2
#             for j in range(3):
#                 if j != label_idx:
#                     scores[i, j] = remainder

#         mean_scores = scores.mean(axis=0)
#         dominant = LABELS[int(np.argmax(mean_scores))]
#         net = float(mean_scores[2] - mean_scores[0])

#         rows.append({
#             'Topic': tid,
#             'Neg': round(float(mean_scores[0]), 4),
#             'Neu': round(float(mean_scores[1]), 4),
#             'Pos': round(float(mean_scores[2]), 4),
#             'Sentiment': dominant,
#             'Net_Sentiment': round(net, 4),
#         })

#     return pd.DataFrame(rows)

# # ── Output Dir ────────────────────────────────────────────────────────────────
# os.makedirs('results_zeroshot_tags', exist_ok=True)

# # ── Main Loop (IDENTICAL STRUCTURE TO SEASONS) ────────────────────────────────
# summary_rows = []

# for traveler_type in traveler_types:
#     print(f"\n{'='*55}\n  {traveler_type}\n{'='*55}")
    
#     docs = traveler_dfs[traveler_type]['review_text'].dropna().tolist()
#     docs = [d for d in docs if d.strip()]
    
#     # ... [Keep your Embedding and Fit logic here] ...
#     embeddings = embedding_model.encode(docs, show_progress_bar=True, batch_size=512)
    
#     model = BERTopic(
#         embedding_model=embedding_model,
#         zeroshot_topic_list=zeroshot_topic_list,
#         zeroshot_min_similarity=0.70,
#         language="multilingual",
#         verbose=False,
#     )
#     topics, _ = model.fit_transform(docs, embeddings)
    
#     # ─── NEW: Export Model ──────────────────────────────────────────────
#     safe_name = traveler_type.replace(" ", "_")
#     model_path = f'results_zeroshot_tags/{safe_name}_bertopic_model'
    
#     # You can save using the "safetensors" format (recommended) 
#     # or the standard pickle format.
#     model.save(model_path, serialization="safetensors", 
#                save_ctfidf=True, save_embedding_model=False)
    
#     print(f"  ✓ Saved Model to {model_path}")
#     # ────────────────────────────────────────────────────────────────────

#     # ... [Keep your Sentiment and CSV export logic here] ...
#     topic_info = model.get_topic_info()
#     sent_df = sentiment_from_assigned_docs(docs, topics, topic_info)
#     out = topic_info.merge(sent_df, on='Topic', how='left')
#     out['TravelerType'] = traveler_type
    
#     out_csv_path = f'results_zeroshot_tags/{safe_name}_topics.csv'
#     out.to_csv(out_csv_path, index=False)

# summary_df = pd.DataFrame(summary_rows)
# summary_df.to_csv('results_zeroshot/summary.csv', index=False)

# print("\n===== SUMMARY =====")
# print(summary_df.to_string(index=False))
# print("\n✓ Done — results in ./results_zeroshot/")