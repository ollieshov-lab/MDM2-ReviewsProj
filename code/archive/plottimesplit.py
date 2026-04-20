import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

CURRENT_DIR = Path(__file__).resolve().parent
LIB_DIR = next(
    candidate
    for candidate in (CURRENT_DIR / "lib", CURRENT_DIR.parent / "lib")
    if candidate.is_dir()
)
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from rp_paths import resolve_existing_path, resolve_output_dir
from topic_modeling import ZERO_SHOT_MAJOR_TOPICS


TAG_RESULTS_DIR = os.fspath(
    resolve_existing_path(
        "outputs/models/zeroshot_tags",
        "results_zeroshot_tags",
        "results_tags",
        env_var="RP_TAG_INPUT_DIR",
    )
)
SEASON_RESULTS_DIR = os.fspath(
    resolve_existing_path(
        "outputs/models/zeroshot_seasons",
        "results_zeroshot_seasons",
        env_var="RP_SEASON_INPUT_DIR",
    )
)
INTERACTIVE_DIR = os.fspath(
    resolve_output_dir(
        "RP_CROSS_SEGMENT_OUTPUT_DIR",
        "outputs/interactive/cross_segment",
    )
)
TABLE_DIR = os.fspath(
    resolve_output_dir(
        "RP_CROSS_SEGMENT_TABLE_DIR",
        "outputs/tables/cross_segment",
    )
)

TRAVELER_TYPES = ['Couple', 'Solo traveler', 'Family with young children', 'Group']
SEASONS        = ['Spring', 'Summer', 'Autumn', 'Winter']

TRAVELER_COLOURS = {
    'Couple':                      '#4575b4',
    'Solo traveler':               '#d73027',
    'Family with young children':  '#1a9850',
    'Group':                       '#f46d43',
}
SEASON_COLOURS = {
    'Spring': '#85C88A',
    'Summer': '#F5A623',
    'Autumn': '#D4735E',
    'Winter': '#7EB8D4',
}

POS_THRESHOLD = 0.1
NEG_THRESHOLD = -0.1
DELTA_THRESHOLD = 0.05

MONTH_NAMES  = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}


def interactive_path(*parts: str) -> str:
    return os.path.join(INTERACTIVE_DIR, *parts)


def table_path(*parts: str) -> str:
    return os.path.join(TABLE_DIR, *parts)


os.makedirs(INTERACTIVE_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

# ── Topic-level sentiment summaries ──────────────────────────────────────────
tag_sentiment_frames = []
for tt in TRAVELER_TYPES:
    safe = tt.replace(' ', '_')
    path = os.path.join(TAG_RESULTS_DIR, f'{safe}_Topic_Sentiment.csv')
    if os.path.exists(path):
        tag_sentiment_frames.append(pd.read_csv(path))
    else:
        print(f'  WARNING: missing {path}')

season_sentiment_frames = []
for s in SEASONS:
    path = os.path.join(SEASON_RESULTS_DIR, f'{s}_Topic_Sentiment.csv')
    if os.path.exists(path):
        season_sentiment_frames.append(pd.read_csv(path))
    else:
        print(f'  WARNING: missing {path}')

tags_sent    = pd.concat(tag_sentiment_frames,    ignore_index=True)
seasons_sent = pd.concat(season_sentiment_frames, ignore_index=True)

# Filter to only the 10 canonical topics
tags_sent    = tags_sent[tags_sent['Semantic_Label'].isin(ZERO_SHOT_MAJOR_TOPICS)]
seasons_sent = seasons_sent[seasons_sent['Semantic_Label'].isin(ZERO_SHOT_MAJOR_TOPICS)]

print(f'Loaded tags_sent:    {len(tags_sent):,} rows')
print(f'Loaded seasons_sent: {len(seasons_sent):,} rows')


# ── Fragment-level document info ──────────────────────────────────────────────
tag_doc_frames = []
for tt in TRAVELER_TYPES:
    safe = tt.replace(' ', '_')
    path = os.path.join(TAG_RESULTS_DIR, f'{safe}_Document_Info.csv')
    if os.path.exists(path):
        tag_doc_frames.append(pd.read_csv(path))
    else:
        print(f'  WARNING: missing {path}')

season_doc_frames = []
for s in SEASONS:
    path = os.path.join(SEASON_RESULTS_DIR, f'{s}_Document_Info.csv')
    if os.path.exists(path):
        season_doc_frames.append(pd.read_csv(path))
    else:
        print(f'  WARNING: missing {path}')

tags_doc    = pd.concat(tag_doc_frames,    ignore_index=True)
seasons_doc = pd.concat(season_doc_frames, ignore_index=True)

# Keep only scored, non-outlier fragments within the 10 topics
tags_doc = tags_doc[
    (tags_doc['Topic'] != -1) &
    (tags_doc['Semantic_Label'].isin(ZERO_SHOT_MAJOR_TOPICS)) &
    (tags_doc['Sentiment_Score'].notna())
].copy()

seasons_doc = seasons_doc[
    (seasons_doc['Topic'] != -1) &
    (seasons_doc['Semantic_Label'].isin(ZERO_SHOT_MAJOR_TOPICS)) &
    (seasons_doc['Sentiment_Score'].notna())
].copy()

print(f'Loaded tags_doc:    {len(tags_doc):,} fragments')
print(f'Loaded seasons_doc: {len(seasons_doc):,} fragments')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TRAVELER TYPE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print('\n── Section 1: Traveler Type Plots ──')

# 1a. Donut charts: topic distribution per traveler type ──────────────────────
for tt in TRAVELER_TYPES:
    sub = tags_doc[tags_doc['TravelerType'] == tt]
    counts = (
        sub.groupby('Semantic_Label').size()
        .reset_index(name='Count')
        .sort_values('Count', ascending=False)
    )
    fig = go.Figure(go.Pie(
        labels=counts['Semantic_Label'],
        values=counts['Count'],
        hole=0.4,
        textinfo='label+percent',
        textposition='outside',
        marker=dict(colors=px.colors.sequential.Blues[::-1])
    ))
    fig.update_layout(
        title=dict(text=f'<b>Topic Distribution — {tt}</b>', x=0.5),
        showlegend=False,
        margin=dict(t=80, b=40, l=40, r=40)
    )
    safe = tt.replace(' ', '_')
    fig.write_html(interactive_path(f"1a_Donut_{safe}.html"))
print('  ✓ 1a Donut charts (per traveler type)')


# 1b. Heatmap: TravelerType × Topic mean sentiment ────────────────────────────
pivot_tt = tags_sent.pivot_table(
    index='TravelerType', columns='Semantic_Label', values='Mean_Sentiment', aggfunc='mean'
).reindex(TRAVELER_TYPES)

fig = px.imshow(
    pivot_tt,
    color_continuous_scale='RdBu', color_continuous_midpoint=0, aspect='auto',
    title='<b>Mean Sentiment by Traveler Type and Topic</b>',
    labels=dict(color='Sentiment')
)
fig.update_layout(
    xaxis_title='Topic', yaxis_title='Traveler Type',
    margin=dict(t=70, l=200, r=20, b=160)
)
fig.update_xaxes(tickangle=40)
fig.write_html(interactive_path("1b_Heatmap_TravelerType_Topic.html"))
print('  ✓ 1b Heatmap TravelerType × Topic')


# 1c. Bar chart per topic: traveler types ranked by sentiment ─────────────────
os.makedirs(interactive_path("PerTopic_Tags"), exist_ok=True)
for topic in ZERO_SHOT_MAJOR_TOPICS:
    sub = tags_sent[tags_sent['Semantic_Label'] == topic].sort_values('Mean_Sentiment')
    if sub.empty:
        continue
    colors = [TRAVELER_COLOURS.get(t, '#888') for t in sub['TravelerType']]
    fig = go.Figure(go.Bar(
        x=sub['Mean_Sentiment'], y=sub['TravelerType'], orientation='h',
        marker_color=colors,
        customdata=sub[['Fragment_Count', 'Mean_Sentiment']].values,
        hovertemplate='<b>%{y}</b><br>Sentiment: %{customdata[1]:.3f}<br>'
                      'Fragments: %{customdata[0]}<extra></extra>'
    ))
    fig.add_vline(x=0, line_dash='dash', opacity=0.4)
    fig.update_layout(
        title=f'<b>{topic}</b>: Sentiment by Traveler Type',
        xaxis_title='Mean Sentiment (-1 to +1)', xaxis=dict(range=[-1, 1]),
        margin=dict(t=70, l=220, r=20, b=50), height=300
    )
    safe = topic.replace(' ', '_').replace('/', '_').replace('&', 'and')
    fig.write_html(interactive_path("PerTopic_Tags", f"1c_{safe}_TravelerType.html"))
print('  ✓ 1c Bar charts per topic (traveler type ranking)')


# 1d. Radar: all traveler types overlaid ──────────────────────────────────────
all_topics = sorted(ZERO_SHOT_MAJOR_TOPICS)
fig = go.Figure()
for tt in TRAVELER_TYPES:
    sub = tags_sent[tags_sent['TravelerType'] == tt]
    vals = (
        sub.set_index('Semantic_Label')['Mean_Sentiment']
        .reindex(all_topics, fill_value=0)
        .tolist()
    )
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=all_topics + [all_topics[0]],
        name=tt, line=dict(color=TRAVELER_COLOURS[tt]),
        fill='toself', opacity=0.5
    ))
fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[-1, 1])),
    title=dict(text='<b>Topic Sentiment Radar — All Traveler Types</b>', x=0.5),
    legend=dict(orientation='h', yanchor='bottom', y=-0.3),
    margin=dict(t=80, l=60, r=20, b=80)
)
fig.write_html(interactive_path("1d_Radar_AllTravelerTypes.html"))
print('  ✓ 1d Radar all traveler types')


# 1e. Delta heatmap: each traveler type vs overall average ────────────────────
overall_tag_avg = (
    tags_sent.groupby('Semantic_Label')['Mean_Sentiment']
    .mean()
    .rename('Overall_Mean')
)
tags_sent_delta = tags_sent.merge(overall_tag_avg, on='Semantic_Label', how='left')
tags_sent_delta['Delta_vs_Overall'] = tags_sent_delta['Mean_Sentiment'] - tags_sent_delta['Overall_Mean']

pivot_tt_delta = tags_sent_delta.pivot_table(
    index='TravelerType', columns='Semantic_Label', values='Delta_vs_Overall', aggfunc='mean'
).reindex(TRAVELER_TYPES)

fig = px.imshow(
    pivot_tt_delta,
    color_continuous_scale='RdBu', color_continuous_midpoint=0, aspect='auto',
    title=('<b>Traveler Type vs Overall Average (Delta Sentiment)</b><br>'
           '<sup>Blue = above overall average | Red = below overall average</sup>'),
    labels=dict(color='Delta')
)
fig.update_layout(
    xaxis_title='Topic', yaxis_title='Traveler Type',
    margin=dict(t=90, l=200, r=20, b=160)
)
fig.update_xaxes(tickangle=40)
fig.write_html(interactive_path("1e_Delta_TravelerType_vs_Overall.html"))
print('  ✓ 1e Delta heatmap traveler type vs overall')


# 1f. Stacked bar: sentiment label distribution across traveler types ─────────
def score_to_label(s):
    if s > POS_THRESHOLD:  return 'Positive'
    if s < NEG_THRESHOLD:  return 'Negative'
    return 'Neutral'

tags_doc['Sent_Label'] = tags_doc['Sentiment_Score'].apply(score_to_label)

sent_counts_tt = {}
for tt in TRAVELER_TYPES:
    vc = tags_doc[tags_doc['TravelerType'] == tt]['Sent_Label'].value_counts()
    sent_counts_tt[tt] = {
        'Positive': vc.get('Positive', 0),
        'Neutral':  vc.get('Neutral',  0),
        'Negative': vc.get('Negative', 0),
    }

bar_spec = [('Positive', '#4575b4'), ('Neutral', '#888'), ('Negative', '#d73027')]
fig = go.Figure()
for label, color in bar_spec:
    fig.add_trace(go.Bar(
        x=TRAVELER_TYPES,
        y=[sent_counts_tt[tt][label] for tt in TRAVELER_TYPES],
        name=label, marker_color=color, opacity=0.85,
        text=[f"{sent_counts_tt[tt][label]:,}" for tt in TRAVELER_TYPES],
        textposition='inside', insidetextanchor='middle',
    ))
fig.update_layout(
    barmode='stack',
    title=dict(text=f'<b>Sentiment Distribution Across Traveler Types</b><br>'
                    f'<sup>Threshold ±{POS_THRESHOLD}</sup>'),
    xaxis_title='Traveler Type', yaxis_title='Number of Fragments',
    height=520, margin=dict(t=80, l=60, r=20, b=80),
    legend=dict(orientation='h', yanchor='bottom', y=-0.2)
)
fig.write_html(interactive_path("1f_StackedBar_SentimentDist_TravelerType.html"))
print('  ✓ 1f Stacked bar sentiment distribution (traveler types)')


# 1g. Grouped bar: fragment count per topic per traveler type ─────────────────
fig = px.bar(
    tags_sent.sort_values(['Semantic_Label', 'TravelerType']),
    x='Semantic_Label', y='Fragment_Count', color='TravelerType',
    barmode='group',
    color_discrete_map=TRAVELER_COLOURS,
    title='<b>Fragment Count per Topic by Traveler Type</b>',
    labels={'Fragment_Count': 'Scored Fragments', 'Semantic_Label': 'Topic', 'TravelerType': 'Traveler Type'}
)
fig.update_layout(
    xaxis_tickangle=40,
    margin=dict(t=70, l=60, r=20, b=160), height=520
)
fig.write_html(interactive_path("1g_FragmentCount_TravelerType.html"))
print('  ✓ 1g Fragment count per topic (traveler type)')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SEASONAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print('\n── Section 2: Season Plots ──')

# 2a. Donut charts: topic distribution per season ─────────────────────────────
for s in SEASONS:
    sub = seasons_doc[seasons_doc['Season'] == s]
    counts = (
        sub.groupby('Semantic_Label').size()
        .reset_index(name='Count')
        .sort_values('Count', ascending=False)
    )
    fig = go.Figure(go.Pie(
        labels=counts['Semantic_Label'],
        values=counts['Count'],
        hole=0.4,
        textinfo='label+percent',
        textposition='outside',
        marker=dict(colors=px.colors.sequential.Blues[::-1])
    ))
    fig.update_layout(
        title=dict(text=f'<b>Topic Distribution — {s}</b>', x=0.5),
        showlegend=False,
        margin=dict(t=80, b=40, l=40, r=40)
    )
    fig.write_html(interactive_path(f"2a_Donut_{s}.html"))
print('  ✓ 2a Donut charts (per season)')


# 2b. Heatmap: Season × Topic mean sentiment ──────────────────────────────────
pivot_s = seasons_sent.pivot_table(
    index='Season', columns='Semantic_Label', values='Mean_Sentiment', aggfunc='mean'
).reindex(SEASONS)

fig = px.imshow(
    pivot_s,
    color_continuous_scale='RdBu', color_continuous_midpoint=0, aspect='auto',
    title='<b>Mean Sentiment by Season and Topic</b>',
    labels=dict(color='Sentiment')
)
fig.update_layout(
    xaxis_title='Topic', yaxis_title='Season',
    margin=dict(t=70, l=100, r=20, b=160)
)
fig.update_xaxes(tickangle=40)
fig.write_html(interactive_path("2b_Heatmap_Season_Topic.html"))
print('  ✓ 2b Heatmap Season × Topic')


# 2c. Bar chart per topic: seasons ranked by sentiment ────────────────────────
os.makedirs(interactive_path("PerTopic_Seasons"), exist_ok=True)
for topic in ZERO_SHOT_MAJOR_TOPICS:
    sub = seasons_sent[seasons_sent['Semantic_Label'] == topic].sort_values('Mean_Sentiment')
    if sub.empty:
        continue
    colors = [SEASON_COLOURS.get(s, '#888') for s in sub['Season']]
    fig = go.Figure(go.Bar(
        x=sub['Mean_Sentiment'], y=sub['Season'], orientation='h',
        marker_color=colors,
        customdata=sub[['Fragment_Count', 'Mean_Sentiment']].values,
        hovertemplate='<b>%{y}</b><br>Sentiment: %{customdata[1]:.3f}<br>'
                      'Fragments: %{customdata[0]}<extra></extra>'
    ))
    fig.add_vline(x=0, line_dash='dash', opacity=0.4)
    fig.update_layout(
        title=f'<b>{topic}</b>: Sentiment by Season',
        xaxis_title='Mean Sentiment (-1 to +1)', xaxis=dict(range=[-1, 1]),
        margin=dict(t=70, l=100, r=20, b=50), height=300
    )
    safe = topic.replace(' ', '_').replace('/', '_').replace('&', 'and')
    fig.write_html(interactive_path("PerTopic_Seasons", f"2c_{safe}_Season.html"))
print('  ✓ 2c Bar charts per topic (season ranking)')


# 2d. Radar: all seasons overlaid ─────────────────────────────────────────────
fig = go.Figure()
for s in SEASONS:
    sub = seasons_sent[seasons_sent['Season'] == s]
    vals = (
        sub.set_index('Semantic_Label')['Mean_Sentiment']
        .reindex(all_topics, fill_value=0)
        .tolist()
    )
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=all_topics + [all_topics[0]],
        name=s, line=dict(color=SEASON_COLOURS[s]),
        fill='toself', opacity=0.5
    ))
fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[-1, 1])),
    title=dict(text='<b>Topic Sentiment Radar — All Seasons</b>', x=0.5),
    legend=dict(orientation='h', yanchor='bottom', y=-0.3),
    margin=dict(t=80, l=60, r=20, b=80)
)
fig.write_html(interactive_path("2d_Radar_AllSeasons.html"))
print('  ✓ 2d Radar all seasons')


# 2e. Delta heatmap: each season vs overall average ───────────────────────────
overall_season_avg = (
    seasons_sent.groupby('Semantic_Label')['Mean_Sentiment']
    .mean()
    .rename('Overall_Mean')
)
seasons_sent_delta = seasons_sent.merge(overall_season_avg, on='Semantic_Label', how='left')
seasons_sent_delta['Delta_vs_Overall'] = seasons_sent_delta['Mean_Sentiment'] - seasons_sent_delta['Overall_Mean']

pivot_s_delta = seasons_sent_delta.pivot_table(
    index='Season', columns='Semantic_Label', values='Delta_vs_Overall', aggfunc='mean'
).reindex(SEASONS)

fig = px.imshow(
    pivot_s_delta,
    color_continuous_scale='RdBu', color_continuous_midpoint=0, aspect='auto',
    title=('<b>Season vs Overall Average (Delta Sentiment)</b><br>'
           '<sup>Blue = above overall average | Red = below overall average</sup>'),
    labels=dict(color='Delta')
)
fig.update_layout(
    xaxis_title='Topic', yaxis_title='Season',
    margin=dict(t=90, l=100, r=20, b=160)
)
fig.update_xaxes(tickangle=40)
fig.write_html(interactive_path("2e_Delta_Season_vs_Overall.html"))
print('  ✓ 2e Delta heatmap season vs overall')


# 2f. Stacked bar: sentiment label distribution across seasons ────────────────
seasons_doc['Sent_Label'] = seasons_doc['Sentiment_Score'].apply(score_to_label)

sent_counts_s = {}
for s in SEASONS:
    vc = seasons_doc[seasons_doc['Season'] == s]['Sent_Label'].value_counts()
    sent_counts_s[s] = {
        'Positive': vc.get('Positive', 0),
        'Neutral':  vc.get('Neutral',  0),
        'Negative': vc.get('Negative', 0),
    }

fig = go.Figure()
for label, color in bar_spec:
    fig.add_trace(go.Bar(
        x=SEASONS,
        y=[sent_counts_s[s][label] for s in SEASONS],
        name=label, marker_color=color, opacity=0.85,
        text=[f"{sent_counts_s[s][label]:,}" for s in SEASONS],
        textposition='inside', insidetextanchor='middle',
    ))
fig.update_layout(
    barmode='stack',
    title=dict(text=f'<b>Sentiment Distribution Across Seasons</b><br>'
                    f'<sup>Threshold ±{POS_THRESHOLD}</sup>'),
    xaxis_title='Season', yaxis_title='Number of Fragments',
    height=520, margin=dict(t=80, l=60, r=20, b=80),
    legend=dict(orientation='h', yanchor='bottom', y=-0.2)
)
fig.write_html(interactive_path("2f_StackedBar_SentimentDist_Season.html"))
print('  ✓ 2f Stacked bar sentiment distribution (seasons)')


# 2g. Fragment count per topic per season ─────────────────────────────────────
fig = px.bar(
    seasons_sent.sort_values(['Semantic_Label', 'Season']),
    x='Semantic_Label', y='Fragment_Count', color='Season',
    barmode='group',
    color_discrete_map=SEASON_COLOURS,
    title='<b>Fragment Count per Topic by Season</b>',
    labels={'Fragment_Count': 'Scored Fragments', 'Semantic_Label': 'Topic', 'Season': 'Season'}
)
fig.update_layout(
    xaxis_tickangle=40,
    margin=dict(t=70, l=60, r=20, b=160), height=520
)
fig.write_html(interactive_path("2g_FragmentCount_Season.html"))
print('  ✓ 2g Fragment count per topic (season)')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CROSS-SEGMENT: TAGS × SEASONS
# ══════════════════════════════════════════════════════════════════════════════
print('\n── Section 3: Cross-Segment Plots ──')

# 3a. Side-by-side heatmap comparison: tags vs seasons for every topic ─────────
# Normalise both to the same colour scale by combining into one figure with subplots
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Traveler Type × Topic', 'Season × Topic'),
    horizontal_spacing=0.12
)

# Left: traveler type
for i, tt in enumerate(TRAVELER_TYPES):
    for j, topic in enumerate(all_topics):
        val = pivot_tt.loc[tt, topic] if topic in pivot_tt.columns else np.nan
        # We'll add as a heatmap trace below

z_tt = pivot_tt.reindex(columns=all_topics).values
z_s  = pivot_s.reindex(columns=all_topics).values

vmax = max(np.nanmax(np.abs(z_tt)), np.nanmax(np.abs(z_s)))

fig.add_trace(
    go.Heatmap(
        z=z_tt, x=all_topics, y=TRAVELER_TYPES,
        colorscale='RdBu', zmid=0, zmin=-vmax, zmax=vmax,
        colorbar=dict(x=0.44, len=0.9, title='Sentiment'),
        showscale=True
    ),
    row=1, col=1
)
fig.add_trace(
    go.Heatmap(
        z=z_s, x=all_topics, y=SEASONS,
        colorscale='RdBu', zmid=0, zmin=-vmax, zmax=vmax,
        colorbar=dict(x=1.01, len=0.9, title='Sentiment'),
        showscale=True
    ),
    row=1, col=2
)
fig.update_xaxes(tickangle=45)
fig.update_layout(
    title=dict(text='<b>Mean Sentiment Comparison: Traveler Types vs Seasons</b>', x=0.5),
    height=420,
    margin=dict(t=90, l=180, r=80, b=160)
)
fig.write_html(interactive_path("3a_SideBySide_Heatmap_Tags_vs_Seasons.html"))
print('  ✓ 3a Side-by-side heatmap tags vs seasons')


# 3b. Line chart: mean sentiment per topic — each traveler type as a line,
#     seasons on x-axis (requires merging both doc tables) ────────────────────
#     For each traveler type, we compute sentiment per season from the raw fragments
#     by joining seasons_doc (has Season) with tags_doc (has TravelerType) via Person_id.

# tags_doc has Person_id and TravelerType; seasons_doc has Person_id and Season
# Merge on Person_id to get both dimensions at once
cross_doc = tags_doc[['Person_id', 'TravelerType', 'Semantic_Label', 'Sentiment_Score']].merge(
    seasons_doc[['Person_id', 'Season']].drop_duplicates(subset='Person_id'),
    on='Person_id', how='inner'
)

cross_agg = (
    cross_doc
    .groupby(['TravelerType', 'Season', 'Semantic_Label'])
    .agg(Mean_Sentiment=('Sentiment_Score', 'mean'), Fragment_Count=('Sentiment_Score', 'count'))
    .reset_index()
)
cross_agg = cross_agg[
    (cross_agg['Semantic_Label'].isin(ZERO_SHOT_MAJOR_TOPICS)) &
    (cross_agg['Fragment_Count'] >= 3)
]
cross_agg['Season'] = pd.Categorical(cross_agg['Season'], categories=SEASONS, ordered=True)
cross_agg = cross_agg.sort_values('Season')

cross_agg.to_csv(table_path("cross_TravelerType_Season_Topic.csv"), index=False)
print(f'  Saved cross_agg: {len(cross_agg):,} rows')


# 3b. Line: overall sentiment across seasons, one line per traveler type ───────
overall_cross = (
    cross_doc
    .groupby(['TravelerType', 'Season'])
    .agg(Mean_Sentiment=('Sentiment_Score', 'mean'))
    .reset_index()
)
overall_cross['Season'] = pd.Categorical(overall_cross['Season'], categories=SEASONS, ordered=True)
overall_cross = overall_cross.sort_values('Season')

fig = px.line(
    overall_cross, x='Season', y='Mean_Sentiment', color='TravelerType',
    markers=True,
    color_discrete_map=TRAVELER_COLOURS,
    title='<b>Overall Sentiment Across Seasons by Traveler Type</b>',
    labels={'Mean_Sentiment': 'Mean Sentiment (-1 to +1)', 'Season': 'Season'}
)
fig.add_hline(y=0, line_dash='dot', opacity=0.25)
fig.update_layout(
    yaxis=dict(range=[-1, 1]),
    height=480, margin=dict(t=80, l=60, r=20, b=80),
    legend=dict(orientation='h', yanchor='bottom', y=-0.25)
)
fig.write_html(interactive_path("3b_Line_Overall_TravelerType_by_Season.html"))
print('  ✓ 3b Line: overall sentiment by season × traveler type')


# 3c. Heatmap per topic: Season (rows) × TravelerType (cols) ──────────────────
os.makedirs(interactive_path("CrossTopic_Heatmaps"), exist_ok=True)
for topic in ZERO_SHOT_MAJOR_TOPICS:
    sub = cross_agg[cross_agg['Semantic_Label'] == topic]
    if sub.empty:
        continue
    pivot_cross = sub.pivot_table(
        index='Season', columns='TravelerType', values='Mean_Sentiment', aggfunc='mean'
    ).reindex(SEASONS)
    pivot_cross = pivot_cross.reindex(columns=[c for c in TRAVELER_TYPES if c in pivot_cross.columns])

    fig = px.imshow(
        pivot_cross,
        color_continuous_scale='RdBu', color_continuous_midpoint=0, aspect='auto',
        title=f'<b>{topic}</b><br><sup>Season × Traveler Type</sup>',
        labels=dict(color='Sentiment', x='Traveler Type', y='Season')
    )
    fig.update_layout(margin=dict(t=90, l=100, r=20, b=120), height=380)
    fig.update_xaxes(tickangle=30)
    safe = topic.replace(' ', '_').replace('/', '_').replace('&', 'and')
    fig.write_html(
        interactive_path(
            "CrossTopic_Heatmaps",
            f"3c_{safe}_Season_x_TravelerType.html",
        )
    )
print('  ✓ 3c Heatmap per topic: Season × TravelerType')


# 3d. Radar per topic: traveler types vs seasons on the same chart ─────────────
#     One radar per topic showing all 8 segments (4 types + 4 seasons) ─────────
os.makedirs(interactive_path("CrossRadar"), exist_ok=True)
for topic in ZERO_SHOT_MAJOR_TOPICS:
    fig = go.Figure()
    # Traveler types — overall sentiment per topic per traveler type
    for tt in TRAVELER_TYPES:
        row = tags_sent[(tags_sent['TravelerType'] == tt) & (tags_sent['Semantic_Label'] == topic)]
        val = float(row['Mean_Sentiment'].values[0]) if not row.empty else 0
        fig.add_trace(go.Scatterpolar(
            r=[val, val], theta=[topic, topic],
            name=tt, mode='markers',
            marker=dict(color=TRAVELER_COLOURS[tt], size=14, symbol='circle'),
            showlegend=True
        ))
    # Seasons
    for s in SEASONS:
        row = seasons_sent[(seasons_sent['Season'] == s) & (seasons_sent['Semantic_Label'] == topic)]
        val = float(row['Mean_Sentiment'].values[0]) if not row.empty else 0
        fig.add_trace(go.Scatterpolar(
            r=[val, val], theta=[topic, topic],
            name=s, mode='markers',
            marker=dict(color=SEASON_COLOURS[s], size=14, symbol='diamond'),
            showlegend=True
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-1, 1])),
        title=dict(text=f'<b>{topic}</b><br><sup>Circles = Traveler Types | Diamonds = Seasons</sup>', x=0.5),
        legend=dict(orientation='h', yanchor='bottom', y=-0.4),
        margin=dict(t=90, l=60, r=20, b=100)
    )
    safe = topic.replace(' ', '_').replace('/', '_').replace('&', 'and')
    fig.write_html(interactive_path("CrossRadar", f"3d_{safe}_AllSegments.html"))
print('  ✓ 3d Per-topic dot radar: all segments')


# 3e. Full radar: all traveler types AND all seasons on one chart ──────────────
fig = go.Figure()
for tt in TRAVELER_TYPES:
    sub = tags_sent[tags_sent['TravelerType'] == tt]
    vals = sub.set_index('Semantic_Label')['Mean_Sentiment'].reindex(all_topics, fill_value=0).tolist()
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=all_topics + [all_topics[0]],
        name=tt, line=dict(color=TRAVELER_COLOURS[tt], dash='solid'),
        fill='none', opacity=0.8
    ))
for s in SEASONS:
    sub = seasons_sent[seasons_sent['Season'] == s]
    vals = sub.set_index('Semantic_Label')['Mean_Sentiment'].reindex(all_topics, fill_value=0).tolist()
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=all_topics + [all_topics[0]],
        name=s, line=dict(color=SEASON_COLOURS[s], dash='dot'),
        fill='none', opacity=0.8
    ))
fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[-1, 1])),
    title=dict(text='<b>All Segments Radar</b><br><sup>Solid = Traveler Types | Dotted = Seasons</sup>', x=0.5),
    legend=dict(orientation='h', yanchor='bottom', y=-0.4),
    margin=dict(t=90, l=60, r=20, b=120)
)
fig.write_html(interactive_path("3e_Radar_AllSegments_Combined.html"))
print('  ✓ 3e Combined radar all segments')


# 3f. Grouped bar: sentiment per topic, grouped by segment ────────────────────
#     All 8 segments side-by-side per topic
tags_for_combined    = tags_sent[['TravelerType', 'Semantic_Label', 'Mean_Sentiment', 'Fragment_Count']].copy()
tags_for_combined    = tags_for_combined.rename(columns={'TravelerType': 'Segment'})
seasons_for_combined = seasons_sent[['Season', 'Semantic_Label', 'Mean_Sentiment', 'Fragment_Count']].copy()
seasons_for_combined = seasons_for_combined.rename(columns={'Season': 'Segment'})
combined_sent = pd.concat([tags_for_combined, seasons_for_combined], ignore_index=True)

segment_order  = TRAVELER_TYPES + SEASONS
colour_map_all = {**TRAVELER_COLOURS, **SEASON_COLOURS}

fig = px.bar(
    combined_sent.sort_values(['Semantic_Label', 'Segment']),
    x='Semantic_Label', y='Mean_Sentiment', color='Segment',
    barmode='group',
    color_discrete_map=colour_map_all,
    category_orders={'Segment': segment_order},
    title='<b>Mean Sentiment per Topic — All Segments</b>',
    labels={'Mean_Sentiment': 'Mean Sentiment (-1 to +1)', 'Semantic_Label': 'Topic'}
)
fig.add_hline(y=0, line_dash='dot', opacity=0.3)
fig.update_layout(
    xaxis_tickangle=40,
    height=560, margin=dict(t=80, l=60, r=20, b=180),
    legend=dict(orientation='h', yanchor='bottom', y=-0.45)
)
fig.write_html(interactive_path("3f_GroupedBar_AllSegments_AllTopics.html"))
print('  ✓ 3f Grouped bar all segments × all topics')


# 3g. Stacked bar: combined sentiment distribution across all 8 segments ───────
all_segments = TRAVELER_TYPES + SEASONS
sent_counts_all = {}
for tt in TRAVELER_TYPES:
    vc = tags_doc[tags_doc['TravelerType'] == tt]['Sent_Label'].value_counts()
    sent_counts_all[tt] = {'Positive': vc.get('Positive', 0), 'Neutral': vc.get('Neutral', 0), 'Negative': vc.get('Negative', 0)}
for s in SEASONS:
    vc = seasons_doc[seasons_doc['Season'] == s]['Sent_Label'].value_counts()
    sent_counts_all[s] = {'Positive': vc.get('Positive', 0), 'Neutral': vc.get('Neutral', 0), 'Negative': vc.get('Negative', 0)}

fig = go.Figure()
for label, color in bar_spec:
    fig.add_trace(go.Bar(
        x=all_segments,
        y=[sent_counts_all[seg][label] for seg in all_segments],
        name=label, marker_color=color, opacity=0.85,
        text=[f"{sent_counts_all[seg][label]:,}" for seg in all_segments],
        textposition='inside', insidetextanchor='middle',
    ))
fig.update_layout(
    barmode='stack',
    title=dict(text=f'<b>Sentiment Distribution — All 8 Segments</b><br>'
                    f'<sup>Left 4 = Traveler Types | Right 4 = Seasons | Threshold ±{POS_THRESHOLD}</sup>'),
    xaxis_title='Segment', yaxis_title='Number of Fragments',
    height=540, margin=dict(t=90, l=60, r=20, b=120),
    legend=dict(orientation='h', yanchor='bottom', y=-0.3)
)
fig.add_vline(x=3.5, line_dash='dash', line_color='black', opacity=0.3)
fig.write_html(interactive_path("3g_StackedBar_AllSegments.html"))
print('  ✓ 3g Stacked bar sentiment distribution all segments')


# 3h. Line: per-topic sentiment across seasons, faceted by traveler type ───────
if not cross_agg.empty:
    fig = px.line(
        cross_agg.sort_values('Season'),
        x='Season', y='Mean_Sentiment', color='Semantic_Label',
        facet_col='TravelerType', facet_col_wrap=2,
        markers=True,
        title='<b>Topic Sentiment Across Seasons — by Traveler Type</b>',
        labels={'Mean_Sentiment': 'Mean Sentiment', 'Semantic_Label': 'Topic'},
        hover_data=['Fragment_Count']
    )
    fig.add_hline(y=0, line_dash='dot', opacity=0.2)
    fig.update_yaxes(range=[-1, 1])
    fig.update_layout(
        height=700, margin=dict(t=90, l=60, r=20, b=80),
        legend=dict(orientation='h', yanchor='bottom', y=-0.2)
    )
    fig.write_html(interactive_path("3h_FacetLine_TopicBySeason_PerTravelerType.html"))
    print('  ✓ 3h Facet line: topic × season, faceted by traveler type')
else:
    print('  ⚠ 3h skipped — no cross-segment data (Person_id overlap required)')


print(f'\n✓ All plots saved to ./{INTERACTIVE_DIR}/')
print('\nPlot index:')
print('  Section 1 — Traveler Types:')
print('    1a  Donut charts (one per traveler type)')
print('    1b  Heatmap: TravelerType × Topic')
print('    1c  Bar charts per topic (traveler type ranking)')
print('    1d  Radar: all traveler types')
print('    1e  Delta heatmap: traveler type vs overall average')
print('    1f  Stacked bar: sentiment distribution by traveler type')
print('    1g  Fragment count per topic by traveler type')
print('  Section 2 — Seasons:')
print('    2a  Donut charts (one per season)')
print('    2b  Heatmap: Season × Topic')
print('    2c  Bar charts per topic (season ranking)')
print('    2d  Radar: all seasons')
print('    2e  Delta heatmap: season vs overall average')
print('    2f  Stacked bar: sentiment distribution by season')
print('    2g  Fragment count per topic by season')
print('  Section 3 — Cross-Segment:')
print('    3a  Side-by-side heatmap: tags vs seasons')
print('    3b  Line: overall sentiment by season × traveler type')
print('    3c  Per-topic heatmap: Season × TravelerType (10 files)')
print('    3d  Per-topic dot radar: all segments (10 files)')
print('    3e  Combined radar: all 8 segments overlaid')
print('    3f  Grouped bar: all segments × all topics')
print('    3g  Stacked bar: sentiment distribution all 8 segments')
print('    3h  Facet line: topic × season, per traveler type')
