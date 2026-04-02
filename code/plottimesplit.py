# analyse_topics.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os

seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
SEASON_COLORS = {'Winter': '#5B8CFF', 'Spring': '#4CAF82', 'Summer': '#FF7043', 'Autumn': '#FFA726'}
BG, FG, GRID = '#0F1117', '#E8EAF0', '#2A2D3A'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG,
    'axes.edgecolor': GRID, 'axes.labelcolor': FG,
    'xtick.color': FG, 'ytick.color': FG,
    'text.color': FG, 'grid.color': GRID,
    'grid.linestyle': '--', 'grid.alpha': 0.4,
    'font.family': 'monospace',
})

os.makedirs('plots', exist_ok=True)

# --- Load and label topics ---
def extract_keyword(name):
    parts = name.split('_')
    if len(parts) > 1 and parts[0].lstrip('-').isdigit():
        # unsupervised format: "0_wifi_wi_fi_internet"
        return parts[1]
    else:
        # zero-shot format: "wifi internet connection slow free"
        return name.split()[0]  # just first word as keyword

def extract_label(name):
    parts = name.split('_')
    if len(parts) > 1 and parts[0].lstrip('-').isdigit():
        # unsupervised format
        return ' / '.join(parts[1:4])
    else:
        # zero-shot format — use first 4 words
        return ' / '.join(name.split()[:4])

# --- Load and label topics ---
MIN_COUNT = 100

infos = {}
for season in seasons:
    df = pd.read_csv(f'results_zeroshot/{season}_topics.csv')
    df = df[df['Topic'] != -1].copy()
    df = df[df['Count'] >= MIN_COUNT]
    df['keyword'] = df['Name'].apply(extract_keyword)
    df['label']   = df['Name'].apply(extract_label)
    infos[season] = df
    print(f"{season}: {len(df)} significant topics (>= {MIN_COUNT} reviews)")

# Build keyword -> set of seasons map
keyword_season_map = defaultdict(set)
for season, df in infos.items():
    for kw in df['keyword']:
        keyword_season_map[kw].add(season)

def classify(kw):
    s = keyword_season_map[kw]
    if len(s) == 4:   return 'all'
    elif len(s) >= 2: return 'partial'
    else:             return 'unique'

for season, df in infos.items():
    df['overlap_class'] = df['keyword'].apply(classify)

# ═══════════════════════════════════════════════════════════
# PLOT 1 — Stacked bar: overlap classification per season
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Topic Overlap Classification per Season", fontsize=15, fontweight='bold', color=FG)

class_colors = {'unique': '#E040FB', 'partial': '#FFD740', 'all': '#69F0AE'}
labels_map   = {'unique': 'Season-specific', 'partial': 'Shared (2-3 seasons)', 'all': 'Universal (all 4)'}

for i, season in enumerate(seasons):
    df = infos[season]
    counts = df['overlap_class'].value_counts()
    bottom = 0
    for cls in ['all', 'partial', 'unique']:
        val = counts.get(cls, 0)
        ax.bar(season, val, bottom=bottom, color=class_colors[cls],
               label=labels_map[cls] if i == 0 else '')
        if val > 0:
            ax.text(i, bottom + val / 2, str(val), ha='center', va='center',
                    fontsize=10, fontweight='bold', color=BG)
        bottom += val

ax.set_ylabel("Number of Topics")
ax.legend(facecolor=BG, edgecolor=GRID)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("plots/p1_overlap_classification.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()

# ═══════════════════════════════════════════════════════════
# PLOTS 2a-2d — One plot per season: unique topics only
# ═══════════════════════════════════════════════════════════
for season in seasons:
    df = infos[season]
    unique = df[df['overlap_class'] == 'unique'].sort_values('Count', ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(4, len(unique) * 0.45 + 2)))
    fig.suptitle(f"{season} — Season-Specific Topics", fontsize=15, fontweight='bold',
                 color=SEASON_COLORS[season])

    if unique.empty:
        ax.text(0.5, 0.5, 'No unique topics found', ha='center', va='center',
                transform=ax.transAxes, color=FG, fontsize=12)
    else:
        bars = ax.barh(unique['label'], unique['Count'],
                       color=SEASON_COLORS[season], alpha=0.85, edgecolor='none')
        for bar in bars:
            ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                    f'{int(bar.get_width()):,}', va='center', fontsize=9, color=FG)

        # highlight the biggest bar
        max_idx = unique['Count'].idxmax()
        bars[unique.index.get_loc(max_idx)].set_edgecolor('white')
        bars[unique.index.get_loc(max_idx)].set_linewidth(1.5)

    ax.set_xlabel("Review Count")
    ax.grid(axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"plots/p2_{season.lower()}_unique_topics.png", dpi=150,
                bbox_inches='tight', facecolor=BG)
    plt.show()
    print(f"Saved plots/p2_{season.lower()}_unique_topics.png")

# ═══════════════════════════════════════════════════════════
# PLOT 3 — Universal topics: combined count across all seasons
# ═══════════════════════════════════════════════════════════
universal_kws = [kw for kw, s in keyword_season_map.items() if len(s) == 4]

rows = []
for season in seasons:
    df = infos[season]
    for kw in universal_kws:
        match = df[df['keyword'] == kw]
        if not match.empty:
            rows.append({'keyword': kw, 'label': match['label'].values[0], 'Count': match['Count'].values[0]})

uni_df = pd.DataFrame(rows).groupby(['keyword', 'label'], as_index=False)['Count'].sum()
uni_df = uni_df.sort_values('Count', ascending=True)

fig, ax = plt.subplots(figsize=(12, max(4, len(uni_df) * 0.5 + 2)))
fig.suptitle("Universal Topics — Combined Count (All Seasons)", fontsize=15, fontweight='bold', color=FG)

bars = ax.barh(uni_df['label'], uni_df['Count'], color='#69F0AE', alpha=0.85, edgecolor='none')
for bar in bars:
    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
            f'{int(bar.get_width()):,}', va='center', fontsize=9, color=FG)

ax.set_xlabel("Total Review Count (all seasons combined)")
ax.grid(axis='x')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("plots/p3_universal_topics_combined.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()

# ═══════════════════════════════════════════════════════════
# TEXT SUMMARY
# ═══════════════════════════════════════════════════════════
print("\n===== UNIVERSAL TOPICS (all 4 seasons) =====")
for kw in sorted(universal_kws):
    print(f"  {kw}")

print("\n===== SEASON-SPECIFIC TOPICS =====")
for season in seasons:
    df = infos[season]
    unique = df[df['overlap_class'] == 'unique'].sort_values('Count', ascending=False)
    print(f"\n  {season}:")
    for _, row in unique.iterrows():
        print(f"    {row['label']:35s} {int(row['Count']):,} reviews")