"""
Topic Visualisation Suite
=========================
Inputs : one or more *_topics.csv files (same schema as Couple_topics.csv)
Outputs: interactive_topics.html   – all charts in one page
         pyldavis_<group>.html     – per-group pyLDAvis panel
"""

import ast, os, re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pyLDAvis
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import normalize

# ── Config ────────────────────────────────────────────────────────────
GROUPS = ['Couple', 'Solo traveler', 'Family with young children', 'Group']
DATA_DIR = Path('results_zeroshot_tags')
OUT_DIR  = Path('.')        # where to write html files

COLORS = px.colors.qualitative.Set2   # one colour per group

def safe_name(name):
    return name.replace(" ", "_")

# ── Load data ─────────────────────────────────────────────────────────
def load_group(g):
    path = DATA_DIR / f"{safe_name(g)}_topics.csv"
    df   = pd.read_csv(path)
    df   = df[df['Topic'] != -1].copy()
    df['Net_Sentiment'] = pd.to_numeric(df['Net_Sentiment'], errors='coerce')
    df['Count']         = pd.to_numeric(df['Count'],         errors='coerce')
    df['keywords']      = df['Representation'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    df['Group'] = g
    return df

dfs = {g: load_group(g) for g in GROUPS}

def is_zeroshot(name):
    return not str(name).split('_')[0].isdigit()

zs = {g: dfs[g][dfs[g]['Name'].apply(is_zeroshot)].set_index('Name')
      for g in GROUPS}

# ═══════════════════════════════════════════════════════════════════════
# 1.  pyLDAvis  (one panel per group)
# ═══════════════════════════════════════════════════════════════════════

def build_pyldavis(df_group: pd.DataFrame, group_name: str) -> str:
    """
    Reconstruct the three matrices pyLDAvis needs from keyword lists + counts.
    Returns the rendered HTML string.
    """
    df = df_group[df_group['Topic'] != -1].copy()
    df = df[df['keywords'].apply(lambda x: len(x) > 0)].reset_index(drop=True)

    # ── vocabulary ────────────────────────────────────────────────────
    vocab      = sorted({w for kws in df['keywords'] for w in kws})
    word_idx   = {w: i for i, w in enumerate(vocab)}
    V, K       = len(vocab), len(df)

    # ── topic-term matrix  (K x V) ────────────────────────────────────
    # Equal weight across the 10 keywords of each topic (uniform within-topic)
    topic_term = np.zeros((K, V))
    for row_i, row in df.iterrows():
        for w in row['keywords']:
            topic_term[row_i, word_idx[w]] = 1.0
    # small smoothing so no zero columns
    topic_term = topic_term + 1e-6
    topic_term = normalize(topic_term, norm='l1', axis=1)   # rows sum to 1

    # ── doc-topic matrix  (D x K) ────────────────────────────────────
    # We treat each topic as a single 'document' whose size is its Count.
    # The doc is 100 % composed of that topic.
    doc_lengths    = df['Count'].fillna(1).astype(int).values
    doc_topic      = np.eye(K)          # each doc belongs entirely to one topic

    # ── term frequency across corpus ──────────────────────────────────
    term_freq = topic_term.T @ doc_lengths          # weighted vocab frequency

    vocab_series     = pd.Series(vocab)
    doc_lengths_s    = pd.Series(doc_lengths)

    try:
        prepared = pyLDAvis.prepare(
            topic_term_dists = topic_term,
            doc_topic_dists  = doc_topic,
            doc_lengths      = doc_lengths_s,
            vocab            = vocab_series,
            term_frequency   = pd.Series(term_freq),
            mds              = 'tsne',
            sort_topics      = False,
            n_jobs           = 1,
        )
        html = pyLDAvis.prepared_data_to_html(prepared)
    except Exception as e:
        html = f"<p>pyLDAvis error for {group_name}: {e}</p>"

    return html


# ── build per-group pyLDAvis files ───────────────────────────────────
pyldavis_htmls = {}
for g in GROUPS:
    print(f"  Building pyLDAvis for {g} …")
    html_str = build_pyldavis(dfs[g], g)
    out_path = OUT_DIR / f"pyldavis_{safe_name(g)}.html"
    out_path.write_text(html_str)
    pyldavis_htmls[g] = html_str
    print(f"    → saved {out_path}")


# ═══════════════════════════════════════════════════════════════════════
# 2.  Hierarchy / Dendrogram  (topic similarity within each group)
# ═══════════════════════════════════════════════════════════════════════

def make_dendrogram_fig(g: str) -> go.Figure:
    df  = zs[g].copy().reset_index()
    if len(df) < 3:
        return go.Figure().update_layout(title=f"Not enough topics for {g}")

    # ── feature vector per topic: keyword presence + sentiment + count ─
    vocab  = sorted({w for kws in df['keywords'] for w in kws})
    vidx   = {w: i for i, w in enumerate(vocab)}
    V      = len(vocab)

    mat = np.zeros((len(df), V + 2))
    for i, row in df.iterrows():
        for w in row['keywords']:
            mat[i, vidx[w]] = 1.0
        mat[i, V]   = row['Net_Sentiment'] if not pd.isna(row['Net_Sentiment']) else 0
        mat[i, V+1] = np.log1p(row['Count']) / 10

    dist   = squareform(pdist(mat, metric='cosine') + 1e-9)
    Z      = linkage(squareform(dist), method='ward')

    # scipy dendrogram to get leaf ordering
    labels = df['Name'].tolist()
    dend   = dendrogram(Z, labels=labels, no_plot=True)

    # ── Plotly heat-map of pairwise distances, reordered ──────────────
    order       = dend['leaves']
    ordered_lbl = [labels[i] for i in order]
    dist_ord    = dist[np.ix_(order, order)]

    fig = go.Figure(go.Heatmap(
        z          = 1 - dist_ord,          # similarity
        x          = ordered_lbl,
        y          = ordered_lbl,
        colorscale = 'Blues',
        zmin=0, zmax=1,
        hovertemplate = "%{x}<br>%{y}<br>Similarity: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title  = f"Topic Similarity Hierarchy — {g}",
        height = max(600, len(df) * 18),
        xaxis  = dict(tickangle=-45, tickfont=dict(size=9)),
        yaxis  = dict(tickfont=dict(size=9)),
        template = 'plotly_dark',
    )
    return fig


hier_figs = {g: make_dendrogram_fig(g) for g in GROUPS}


# ═══════════════════════════════════════════════════════════════════════
# 3.  Sentiment Drift  (shared topics across groups)
# ═══════════════════════════════════════════════════════════════════════

topic_sets = {g: set(zs[g].index) for g in GROUPS}
all_topics = set.union(*topic_sets.values())

topic_presence = pd.DataFrame(
    {g: [t in topic_sets[g] for t in all_topics] for g in GROUPS},
    index=list(all_topics),
)
topic_presence['n_groups'] = topic_presence[GROUPS].sum(axis=1)

# Topics shared by ≥ 2 groups
shared_topics = topic_presence[topic_presence['n_groups'] >= 2].index.tolist()

sent_df = pd.DataFrame(
    {t: [zs[g].loc[t, 'Net_Sentiment'] if t in zs[g].index else np.nan
         for g in GROUPS]
     for t in shared_topics},
    index=GROUPS,
).T

# ── a) Line chart (interactive – click legend to show/hide) ──────────
fig_drift = go.Figure()
for i, topic in enumerate(sent_df.index):
    row = sent_df.loc[topic]
    fig_drift.add_trace(go.Scatter(
        x      = GROUPS,
        y      = row.values,
        mode   = 'lines+markers',
        name   = topic,
        visible= 'legendonly',
        line   = dict(width=2),
        hovertemplate = f"<b>{topic}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>",
    ))

fig_drift.update_layout(
    title    = "Sentiment Drift Across Traveller Types (click topics in legend)",
    template = 'plotly_dark',
    height   = 620,
    yaxis    = dict(title='Net Sentiment', range=[-1, 1]),
    xaxis    = dict(title='Traveller Type'),
    legend   = dict(font=dict(size=9)),
)

# ── b) Heatmap of shared-topic sentiments ────────────────────────────
fig_heat = go.Figure(go.Heatmap(
    z          = sent_df.values,
    x          = GROUPS,
    y          = sent_df.index.tolist(),
    colorscale = 'RdYlGn',
    zmin=-1, zmax=1,
    hovertemplate = "%{y}<br>%{x}: %{z:.3f}<extra></extra>",
))
fig_heat.update_layout(
    title    = "Shared Topics – Sentiment Heatmap",
    template = 'plotly_dark',
    height   = max(600, len(sent_df) * 16),
    yaxis    = dict(tickfont=dict(size=9)),
)

# ── c) Drift variance: which topics polarise across groups? ──────────
drift_var = sent_df.var(axis=1).sort_values(ascending=False)

fig_var = go.Figure(go.Bar(
    x    = drift_var.values[:40],
    y    = drift_var.index[:40],
    orientation = 'h',
    marker_color = px.colors.sequential.Oranges_r[:40],
    hovertemplate = "%{y}<br>Variance: %{x:.4f}<extra></extra>",
))
fig_var.update_layout(
    title    = "Top 40 Topics by Sentiment Drift (variance across groups)",
    template = 'plotly_dark',
    height   = 700,
    xaxis    = dict(title='Variance'),
    yaxis    = dict(tickfont=dict(size=9)),
)


# ═══════════════════════════════════════════════════════════════════════
# 4.  Shared topics bar + bubble  (from original script, kept)
# ═══════════════════════════════════════════════════════════════════════

pairs       = list(combinations(GROUPS, 2))
pair_labels = [f"{g1} ∩ {g2}" for g1, g2 in pairs]
pair_counts = [len(topic_sets[g1] & topic_sets[g2]) for g1, g2 in pairs]

fig_shared = go.Figure(go.Bar(
    x    = pair_labels,
    y    = pair_counts,
    text = pair_counts,
    textposition = 'auto',
    marker_color = COLORS[:len(pairs)],
))
fig_shared.update_layout(
    title        = "Shared Topics Between Traveller Types",
    template     = 'plotly_dark',
    xaxis_tickangle = -30,
)

# Bubble: volume vs sentiment
bubble_df = pd.concat([
    zs[g].reset_index().assign(Group=g) for g in GROUPS
])
fig_bubble = px.scatter(
    bubble_df,
    x          = 'Net_Sentiment',
    y          = 'Count',
    size       = 'Count',
    color      = 'Group',
    hover_data = ['Name'],
    title      = 'Topic Volume vs Sentiment (bubble = volume)',
    log_y      = True,
    template   = 'plotly_dark',
)

# Unique topics
unique_data = []
for g in GROUPS:
    unique_idx = topic_presence[
        (topic_presence[g]) & (topic_presence['n_groups'] == 1)
    ].index
    for t in unique_idx:
        if t in zs[g].index:
            unique_data.append({
                'Group':     g,
                'Topic':     t,
                'Sentiment': zs[g].loc[t, 'Net_Sentiment'],
                'Count':     zs[g].loc[t, 'Count'],
            })

unique_df = pd.DataFrame(unique_data)
fig_unique = px.bar(
    unique_df.sort_values('Sentiment'),
    x           = 'Sentiment',
    y           = 'Topic',
    color       = 'Group',
    orientation = 'h',
    title       = 'Unique Topics per Traveller Type',
    height      = max(600, len(unique_df) * 18),
    template    = 'plotly_dark',
)


# ═══════════════════════════════════════════════════════════════════════
# 5.  Assemble everything into one HTML
# ═══════════════════════════════════════════════════════════════════════

def section(title: str, content: str) -> str:
    return f"""
<div style="margin:30px 0">
  <h2 style="color:#aef; font-family:sans-serif; border-bottom:1px solid #446; padding-bottom:6px">
    {title}
  </h2>
  {content}
</div>
"""

def fig_html(fig, first=False) -> str:
    return fig.to_html(full_html=False,
                       include_plotlyjs='cdn' if first else False)

out_path = OUT_DIR / 'interactive_topics.html'

with open(out_path, 'w') as f:
    # page shell
    f.write("""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Topic Analysis Dashboard</title>
  <style>
    body { background:#0d1117; color:#c9d1d9; font-family:sans-serif; margin:20px; }
    h1   { color:#58a6ff; }
    hr   { border-color:#30363d; }
  </style>
</head>
<body>
<h1>🗺️ Topic Analysis Dashboard</h1>
""")

    # ── shared / bubble / unique ──────────────────────────────────────
    f.write(section("Shared Topics Between Traveller Types",
                    fig_html(fig_shared, first=True)))
    f.write(section("Topic Volume vs Sentiment",
                    fig_html(fig_bubble)))
    f.write(section("Unique Topics per Traveller Type",
                    fig_html(fig_unique)))

    # ── sentiment drift ───────────────────────────────────────────────
    f.write(section("Sentiment Drift – Line Chart (click legend to toggle topics)",
                    fig_html(fig_drift)))
    f.write(section("Sentiment Drift – Heatmap",
                    fig_html(fig_heat)))
    f.write(section("Topics with Greatest Sentiment Drift (Variance)",
                    fig_html(fig_var)))

    # ── hierarchy per group ───────────────────────────────────────────
    for g in GROUPS:
        f.write(section(f"Topic Similarity Hierarchy — {g}",
                        fig_html(hier_figs[g])))

    # ── pyLDAvis panels ───────────────────────────────────────────────
    for g in GROUPS:
        f.write(section(f"pyLDAvis Panel — {g}", pyldavis_htmls[g]))

    f.write("</body></html>")

print(f"\n✅  Saved → {out_path.resolve()}")
print(f"✅  Per-group pyLDAvis files saved separately too.")