import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
SEASONS       = ['Winter', 'Spring', 'Summer', 'Autumn']
DATA_DIR      = 'results_zeroshot_seasons'
OUT_DIR       = 'plots'

PALETTE = {
    'Winter': '#7EB8D4',
    'Spring': '#85C88A',
    'Summer': '#F5A623',
    'Autumn': '#D4735E',
}
SENTIMENT_CMAP = LinearSegmentedColormap.from_list(
    'sent', ['#D45E5E', '#F5F0E8', '#5EAD7A'], N=256
)

BG      = '#0F0F13'
PANEL   = '#1A1A22'
BORDER  = '#2A2A38'
TEXT    = '#E8E4DC'
SUBTEXT = '#8A8698'

plt.rcParams.update({
    'figure.facecolor':  BG,
    'axes.facecolor':    PANEL,
    'axes.edgecolor':    BORDER,
    'axes.labelcolor':   TEXT,
    'axes.titlecolor':   TEXT,
    'xtick.color':       SUBTEXT,
    'ytick.color':       SUBTEXT,
    'text.color':        TEXT,
    'grid.color':        BORDER,
    'grid.linewidth':    0.5,
    'font.family':       'monospace',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

import os
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
dfs = {}
for s in SEASONS:
    path = f'{DATA_DIR}/{s}_topics.csv'
    df   = pd.read_csv(path)
    df   = df[df['Topic'] != -1].copy()          # drop noise
    df['Net_Sentiment'] = pd.to_numeric(df['Net_Sentiment'], errors='coerce')
    df['Count']         = pd.to_numeric(df['Count'], errors='coerce')
    dfs[s] = df

# ── Identify zero-shot matched topics (clean names, no leading digit_) ────────
def is_zeroshot(name):
    """Zero-shot names don't start with a digit."""
    return not str(name).split('_')[0].isdigit()

for s in SEASONS:
    dfs[s]['is_zeroshot'] = dfs[s]['Name'].apply(is_zeroshot)

zs = {s: dfs[s][dfs[s]['is_zeroshot']].set_index('Name') for s in SEASONS}

# ── Shared / unique topic sets ────────────────────────────────────────────────
topic_sets  = {s: set(zs[s].index) for s in SEASONS}
all_topics  = set.union(*topic_sets.values())

def season_mask(topic):
    return tuple(topic in topic_sets[s] for s in SEASONS)

topic_presence = pd.DataFrame(
    {s: [t in topic_sets[s] for t in all_topics] for s in SEASONS},
    index=list(all_topics)
)
topic_presence['n_seasons'] = topic_presence.sum(axis=1)

shared_topics = topic_presence[topic_presence['n_seasons'] > 1].index.tolist()
unique_topics = {
    s: topic_presence[
        (topic_presence[s] == True) & (topic_presence['n_seasons'] == 1)
    ].index.tolist()
    for s in SEASONS
}


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Shared topic count between every pair of seasons  (bar matrix)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor=BG)
fig.suptitle('SHARED TOPICS BETWEEN SEASONS', fontsize=14,
             fontweight='bold', color=TEXT, y=0.98)

pairs = list(combinations(SEASONS, 2))
for ax, (s1, s2) in zip(axes.flatten(), pairs):
    shared = topic_sets[s1] & topic_sets[s2]
    n      = len(shared)
    ax.set_facecolor(PANEL)
    ax.bar([f'{s1}\n∩\n{s2}'], [n],
           color=[PALETTE[s1]], edgecolor=PALETTE[s2], linewidth=2, width=0.5)
    ax.text(0, n + 0.3, str(n), ha='center', va='bottom',
            fontsize=22, fontweight='bold', color=TEXT)
    ax.set_ylim(0, max(n * 1.3, 5))
    ax.set_title(f'{s1} ∩ {s2}', color=TEXT, fontsize=10)
    ax.tick_params(bottom=False, labelbottom=False)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylabel('# shared topics', color=SUBTEXT, fontsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

# last panel — all 4
ax = axes.flatten()[-1]
all_4 = set.intersection(*topic_sets.values())
ax.set_facecolor(PANEL)
ax.bar(['All 4\nSeasons'], [len(all_4)],
       color='#A89ECC', edgecolor='#D4C9FF', linewidth=2, width=0.4)
ax.text(0, len(all_4) + 0.3, str(len(all_4)), ha='center', va='bottom',
        fontsize=22, fontweight='bold', color=TEXT)
ax.set_ylim(0, max(len(all_4) * 1.3, 5))
ax.set_title('All 4 Seasons', color=TEXT, fontsize=10)
ax.tick_params(bottom=False, labelbottom=False)
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.set_ylabel('# shared topics', color=SUBTEXT, fontsize=8)
for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/1_shared_topic_counts.png', dpi=150, bbox_inches='tight',
            facecolor=BG)
plt.close()
print('✓ Plot 1 saved')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Heatmap: shared topics × seasons, coloured by Net_Sentiment
# ══════════════════════════════════════════════════════════════════════════════
# Build matrix: rows = topics present in ≥2 seasons
heat_rows = []
for t in shared_topics:
    row = {'Topic': t}
    for s in SEASONS:
        row[s] = zs[s].loc[t, 'Net_Sentiment'] if t in zs[s].index else np.nan
    heat_rows.append(row)

heat_df = (pd.DataFrame(heat_rows)
             .set_index('Topic')
             .sort_values('Winter', ascending=False))

# Limit to top 40 by mean sentiment magnitude for readability
if len(heat_df) > 40:
    heat_df = heat_df.loc[heat_df.abs().mean(axis=1).nlargest(40).index]

fig_h = max(8, len(heat_df) * 0.28)
fig, ax = plt.subplots(figsize=(10, fig_h), facecolor=BG)
ax.set_facecolor(BG)

im = ax.imshow(heat_df.values, cmap=SENTIMENT_CMAP,
               aspect='auto', vmin=-1, vmax=1)

ax.set_xticks(range(4))
ax.set_xticklabels(SEASONS, fontsize=11, fontweight='bold')
for i, s in enumerate(SEASONS):
    ax.get_xticklabels()[i].set_color(PALETTE[s])

ax.set_yticks(range(len(heat_df)))
ax.set_yticklabels(heat_df.index, fontsize=7.5, color=TEXT)
ax.tick_params(length=0)

# cell values
for r in range(len(heat_df)):
    for c in range(4):
        v = heat_df.values[r, c]
        if not np.isnan(v):
            ax.text(c, r, f'{v:+.2f}', ha='center', va='center',
                    fontsize=6.5,
                    color='#0F0F13' if abs(v) > 0.3 else TEXT)

cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
cbar.ax.yaxis.set_tick_params(color=SUBTEXT)
cbar.set_label('Net Sentiment  (pos − neg)', color=SUBTEXT, fontsize=9)

ax.set_title('SHARED TOPICS — NET SENTIMENT BY SEASON\n'
             '(topics present in ≥ 2 seasons, top 40 by magnitude)',
             color=TEXT, fontsize=12, fontweight='bold', pad=12)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/2_shared_sentiment_heatmap.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print('✓ Plot 2 saved')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Line plot: how shared-topic sentiment shifts across seasons
# ══════════════════════════════════════════════════════════════════════════════
# Pick topics present in all 4 seasons with meaningful sentiment variance
all4_topics = topic_presence[topic_presence['n_seasons'] == 4].index.tolist()

sent_matrix = {}
for t in all4_topics:
    vals = [zs[s].loc[t, 'Net_Sentiment'] if t in zs[s].index else np.nan
            for s in SEASONS]
    sent_matrix[t] = vals

sent_4 = pd.DataFrame(sent_matrix, index=SEASONS).T
sent_4['variance'] = sent_4.var(axis=1)
top_var = sent_4.nlargest(min(15, len(sent_4)), 'variance').drop(columns='variance')

fig, ax = plt.subplots(figsize=(13, 7), facecolor=BG)
ax.set_facecolor(PANEL)

cmap_lines = plt.cm.get_cmap('tab20', len(top_var))
x = np.arange(4)

for i, (topic, row) in enumerate(top_var.iterrows()):
    vals  = row.values.astype(float)
    color = cmap_lines(i)
    ax.plot(x, vals, 'o-', color=color, linewidth=1.8,
            markersize=5, alpha=0.9)
    # label at right end
    last_valid = np.where(~np.isnan(vals))[0]
    if len(last_valid):
        end = last_valid[-1]
        ax.annotate(topic[:45], xy=(end, vals[end]),
                    xytext=(end + 0.08, vals[end]),
                    fontsize=6.5, color=color, va='center',
                    annotation_clip=False)

ax.axhline(0, color=BORDER, linewidth=1, linestyle='--')
ax.set_xticks(x)
ax.set_xticklabels(SEASONS, fontsize=11)
for i, s in enumerate(SEASONS):
    ax.get_xticklabels()[i].set_color(PALETTE[s])

ax.set_ylabel('Net Sentiment', color=SUBTEXT)
ax.set_xlim(-0.2, 5.5)
ax.set_title('SEASONAL SENTIMENT DRIFT\nTopics present all year — highest variance',
             color=TEXT, fontsize=12, fontweight='bold', pad=10)
ax.grid(axis='y', alpha=0.3)
for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/3_shared_topic_sentiment_drift.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print('✓ Plot 3 saved')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Unique topics per season: horizontal bars coloured by sentiment
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(18, 8), facecolor=BG)
fig.suptitle('UNIQUE TOPICS PER SEASON — NET SENTIMENT',
             fontsize=13, fontweight='bold', color=TEXT, y=1.01)

for ax, s in zip(axes, SEASONS):
    utopics = unique_topics[s]
    if not utopics:
        ax.set_visible(False)
        continue

    sub = zs[s].loc[[t for t in utopics if t in zs[s].index]].copy()
    sub = sub.sort_values('Net_Sentiment')

    colors = [SENTIMENT_CMAP((v + 1) / 2) for v in sub['Net_Sentiment']]
    ax.set_facecolor(PANEL)
    bars = ax.barh(range(len(sub)), sub['Net_Sentiment'],
                   color=colors, edgecolor='none', height=0.7)

    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub.index, fontsize=7, color=TEXT)
    ax.axvline(0, color=BORDER, linewidth=1)
    ax.set_xlabel('Net Sentiment', color=SUBTEXT, fontsize=8)
    ax.set_title(s, color=PALETTE[s], fontsize=13, fontweight='bold')
    ax.set_xlim(-1, 1)
    ax.grid(axis='x', alpha=0.25)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/4_unique_topics_sentiment.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print('✓ Plot 4 saved')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Bubble chart: topic volume vs sentiment (all zeroshot topics)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=BG)
fig.suptitle('TOPIC VOLUME vs. SENTIMENT', fontsize=13,
             fontweight='bold', color=TEXT)

for ax, s in zip(axes.flatten(), SEASONS):
    df_s = zs[s].copy().reset_index()
    df_s = df_s.dropna(subset=['Net_Sentiment', 'Count'])

    sizes  = (df_s['Count'] / df_s['Count'].max()) * 800 + 20
    colors = [SENTIMENT_CMAP((v + 1) / 2) for v in df_s['Net_Sentiment']]

    ax.set_facecolor(PANEL)
    sc = ax.scatter(df_s.index, df_s['Net_Sentiment'],
                    s=sizes, c=colors, alpha=0.75, edgecolors='none')

    # label top 5 by count
    top5 = df_s.nlargest(5, 'Count')
    for _, row in top5.iterrows():
        ax.annotate(row['Name'][:30],
                    xy=(row.name, row['Net_Sentiment']),
                    xytext=(3, 3), textcoords='offset points',
                    fontsize=6, color=TEXT, alpha=0.85)

    ax.axhline(0, color=BORDER, linewidth=1, linestyle='--')
    ax.set_ylabel('Net Sentiment', color=SUBTEXT, fontsize=8)
    ax.set_xlabel('Topic index', color=SUBTEXT, fontsize=8)
    ax.set_title(s, color=PALETTE[s], fontsize=11, fontweight='bold')
    ax.set_ylim(-1.1, 1.1)
    ax.grid(alpha=0.2)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/5_volume_vs_sentiment_bubbles.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print('✓ Plot 5 saved')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 6 — Top 10 most positive & negative topics across all seasons combined
# ══════════════════════════════════════════════════════════════════════════════
all_zs = pd.concat(
    [zs[s].assign(Season=s).reset_index() for s in SEASONS],
    ignore_index=True
)
# average sentiment per topic name across seasons
topic_avg = (all_zs.groupby('Name')
               .agg(Avg_Sentiment=('Net_Sentiment', 'mean'),
                    Total_Count=('Count', 'sum'))
               .dropna()
               .sort_values('Avg_Sentiment'))

top_neg  = topic_avg.head(10)
top_pos  = topic_avg.tail(10).iloc[::-1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
fig.suptitle('TOP 10 MOST POSITIVE & NEGATIVE TOPICS (all seasons combined)',
             fontsize=12, fontweight='bold', color=TEXT)

for ax, sub, title, cval in [
    (ax1, top_neg, 'Most Negative', '#D45E5E'),
    (ax2, top_pos, 'Most Positive', '#5EAD7A'),
]:
    ax.set_facecolor(PANEL)
    bars = ax.barh(range(len(sub)), sub['Avg_Sentiment'],
                   color=cval, alpha=0.85, edgecolor='none', height=0.65)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub.index, fontsize=8, color=TEXT)
    ax.axvline(0, color=BORDER, linewidth=1)
    ax.set_xlabel('Avg Net Sentiment', color=SUBTEXT, fontsize=9)
    ax.set_title(title, color=cval, fontsize=11, fontweight='bold')
    ax.set_xlim(-1, 1)
    ax.grid(axis='x', alpha=0.25)
    for bar, val in zip(bars, sub['Avg_Sentiment']):
        ax.text(val + (0.02 if val >= 0 else -0.02),
                bar.get_y() + bar.get_height() / 2,
                f'{val:+.2f}', va='center',
                ha='left' if val >= 0 else 'right',
                fontsize=8, color=TEXT)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/6_top_positive_negative_topics.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print('✓ Plot 6 saved')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 7 — Stacked bar: sentiment label distribution per season
# ══════════════════════════════════════════════════════════════════════════════
sent_counts = {}
for s in SEASONS:
    vc = dfs[s]['Sentiment'].value_counts()
    sent_counts[s] = {
        'Positive': vc.get('Positive', 0),
        'Neutral':  vc.get('Neutral', 0),
        'Negative': vc.get('Negative', 0),
    }

sc_df = pd.DataFrame(sent_counts).T

fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
ax.set_facecolor(PANEL)

bottoms = np.zeros(4)
colors_s = ['#5EAD7A', '#8A8698', '#D45E5E']
labels_s = ['Positive', 'Neutral', 'Negative']

x = np.arange(4)
for col, color, label in zip(['Positive', 'Neutral', 'Negative'],
                               colors_s, labels_s):
    vals = sc_df[col].values
    ax.bar(x, vals, bottom=bottoms, color=color, label=label,
           edgecolor=BG, linewidth=0.5, width=0.55)
    for xi, (v, b) in enumerate(zip(vals, bottoms)):
        if v > 0:
            ax.text(xi, b + v / 2, str(v), ha='center', va='center',
                    fontsize=9, fontweight='bold', color=BG)
    bottoms += vals

ax.set_xticks(x)
ax.set_xticklabels(SEASONS, fontsize=11)
for i, s in enumerate(SEASONS):
    ax.get_xticklabels()[i].set_color(PALETTE[s])

ax.set_ylabel('Number of topics', color=SUBTEXT)
ax.set_title('SENTIMENT LABEL DISTRIBUTION ACROSS SEASONS',
             color=TEXT, fontsize=12, fontweight='bold', pad=10)
ax.legend(loc='upper right', framealpha=0.15, labelcolor=TEXT,
          edgecolor=BORDER, fontsize=9)
ax.grid(axis='y', alpha=0.2)
for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/7_sentiment_label_distribution.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print('✓ Plot 7 saved')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 8 — Radar chart: avg sentiment by topic CATEGORY per season
# ══════════════════════════════════════════════════════════════════════════════
CATEGORIES = {
    'Location':    ['proximity', 'city centre', 'metro', 'tram', 'train station',
                    'tube station', 'airport', 'taxi', 'transport', 'walking distance'],
    'Room':        ['room size', 'bed', 'mattress', 'pillow', 'bathroom',
                    'shower', 'air conditioning', 'soundproof', 'view', 'cleanliness',
                    'decor', 'lighting', 'minibar', 'coffee machine', 'towel',
                    'fridge', 'TV', 'Wi-Fi', 'balcony'],
    'Staff':       ['staff', 'reception', 'concierge', 'check-in', 'check-out',
                    'housekeeping', 'customer service', 'language'],
    'Food':        ['breakfast', 'buffet', 'restaurant', 'bar', 'cocktail',
                    'coffee', 'room service', 'gluten', 'vegetarian', 'juice'],
    'Facilities':  ['pool', 'spa', 'gym', 'sauna', 'lounge', 'rooftop',
                    'bike', 'laundry', 'parking', 'luggage'],
    'Value':       ['value for money', 'overpriced', 'expensive', 'city tax', 'charges'],
    'Issues':      ['fire alarm', 'noise', 'construction', 'lift', 'key card',
                    'theft', 'mould', 'insects', 'smell', 'plumbing', 'cold room',
                    'overheating', 'renovation'],
}

def assign_category(name):
    nl = name.lower()
    for cat, kws in CATEGORIES.items():
        if any(kw in nl for kw in kws):
            return cat
    return 'Other'

all_zs['Category'] = all_zs['Name'].apply(assign_category)

cat_season = (all_zs.groupby(['Season', 'Category'])['Net_Sentiment']
                .mean().unstack('Category').reindex(SEASONS))

cats    = list(CATEGORIES.keys())
n_cats  = len(cats)
angles  = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True), facecolor=BG)
ax.set_facecolor(BG)
ax.spines['polar'].set_edgecolor(BORDER)

for s in SEASONS:
    vals = [cat_season.loc[s, c] if c in cat_season.columns else 0 for c in cats]
    vals = [v if not np.isnan(v) else 0 for v in vals]
    vals += vals[:1]
    ax.plot(angles, vals, 'o-', linewidth=2, color=PALETTE[s],
            label=s, markersize=4)
    ax.fill(angles, vals, alpha=0.08, color=PALETTE[s])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(cats, fontsize=10, color=TEXT)
ax.set_ylim(-1, 1)
ax.set_yticks([-0.5, 0, 0.5])
ax.set_yticklabels(['-0.5', '0', '0.5'], fontsize=7, color=SUBTEXT)
ax.yaxis.grid(True, color=BORDER, linewidth=0.5)
ax.xaxis.grid(True, color=BORDER, linewidth=0.5)

ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15),
          framealpha=0.1, edgecolor=BORDER, labelcolor=TEXT, fontsize=10)
ax.set_title('AVG SENTIMENT BY TOPIC CATEGORY & SEASON',
             color=TEXT, fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/8_radar_category_sentiment.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print('✓ Plot 8 saved')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 9 — Review volume per topic (top 20) across seasons — grouped bar
# ══════════════════════════════════════════════════════════════════════════════
top_by_vol = (all_zs.groupby('Name')['Count'].sum()
                .nlargest(20).index.tolist())

vol_df = (all_zs[all_zs['Name'].isin(top_by_vol)]
            .pivot_table(index='Name', columns='Season',
                         values='Count', aggfunc='sum')
            .reindex(columns=SEASONS)
            .fillna(0))
vol_df = vol_df.loc[vol_df.sum(axis=1).sort_values(ascending=False).index]

x  = np.arange(len(vol_df))
w  = 0.2
fig, ax = plt.subplots(figsize=(16, 6), facecolor=BG)
ax.set_facecolor(PANEL)

for i, s in enumerate(SEASONS):
    ax.bar(x + i * w, vol_df[s], width=w, color=PALETTE[s],
           label=s, edgecolor='none', alpha=0.9)

ax.set_xticks(x + 1.5 * w)
ax.set_xticklabels(vol_df.index, rotation=35, ha='right', fontsize=8, color=TEXT)
ax.set_ylabel('Review count', color=SUBTEXT)
ax.set_title('TOP 20 TOPICS BY REVIEW VOLUME — SEASONAL BREAKDOWN',
             color=TEXT, fontsize=12, fontweight='bold', pad=10)
ax.legend(framealpha=0.1, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
ax.grid(axis='y', alpha=0.2)
for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/9_top20_volume_by_season.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print('✓ Plot 9 saved')


print(f'\n✓ All 9 plots saved to ./{OUT_DIR}/')