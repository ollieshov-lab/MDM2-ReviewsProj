import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import re
import ast
from pathlib import Path

st.set_page_config(page_title="Hotel Review Analysis", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOTEL_PIPELINE_TABLES_REL = "outputs/tables/hotel_pipeline"
HOTEL_PIPELINE_INTERACTIVE_REL = "outputs/interactive/hotel_pipeline"
HOTEL_PIPELINE_ENTRYPOINT = "code/pipelines/run_full_pipeline.py"


def resolve_repo_path(*relative_candidates: str) -> Path:
    for candidate in relative_candidates:
        path = PROJECT_ROOT / candidate
        if path.exists():
            return path
    return PROJECT_ROOT / relative_candidates[0]


DATA_DIR      = resolve_repo_path(HOTEL_PIPELINE_TABLES_REL, "code/BERTModelRawOutputs")
PLOTS_DIR     = resolve_repo_path(HOTEL_PIPELINE_INTERACTIVE_REL, "code/BERTModelRawOutputs/Plots")
DOCUMENT_FILE = DATA_DIR / "Hotel_Document_Info.csv"
BENCHMARK_FILE= DATA_DIR / "Hotel_Hotel_vs_City_Benchmark.csv"
ADVICE_FILE   = DATA_DIR / "Hotel_Hotel_Advice.csv"
CORR_FILE     = DATA_DIR / "Hotel_Topic_ReviewerScore_Correlations.csv"
SAMPLE_FILE   = resolve_repo_path("Datasets/Hotel_Reviews_StratSamp_Balanced.csv")

CATEGORIES = [
    "Staff Service", "Room Comfort & Quality", "Cleanliness",
    "Location & Accessibility", "Breakfast & Food",
    "Bathroom & Shower Experience", "Noise & Sleep Disturbance",
    "Facilities & Amenities", "Value for Money", "Maintenance & Room Condition",
]
CATEGORY_COLOURS = {
    "Staff Service":                "#2196F3",
    "Room Comfort & Quality":       "#9C27B0",
    "Cleanliness":                  "#4CAF50",
    "Location & Accessibility":     "#FF9800",
    "Breakfast & Food":             "#F44336",
    "Bathroom & Shower Experience": "#00BCD4",
    "Noise & Sleep Disturbance":    "#795548",
    "Facilities & Amenities":       "#607D8B",
    "Value for Money":              "#FFEB3B",
    "Maintenance & Room Condition": "#E91E63",
}
MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
MONTH_TO_SEASON = {12:"Winter",1:"Winter",2:"Winter",
                   3:"Spring",4:"Spring",5:"Spring",
                   6:"Summer",7:"Summer",8:"Summer",
                   9:"Autumn",10:"Autumn",11:"Autumn"}
SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]


def add_season(df_in):
    """Add Season and SeasonYear columns derived from Month/Year."""
    if 'Season' not in df_in.columns or df_in['Season'].isna().all():
        if 'Month' in df_in.columns:
            df_in = df_in.copy()
            df_in['Season'] = df_in['Month'].map(MONTH_TO_SEASON)
    # Build SeasonYear label e.g. "Spring 2015" for time-series x-axis
    if 'SeasonYear' not in df_in.columns or df_in['SeasonYear'].isna().all():
        if 'Season' in df_in.columns and 'Year' in df_in.columns:
            df_in = df_in.copy() if 'Season' in df_in.columns else df_in
            df_in['SeasonYear'] = df_in['Season'].astype(str) + ' ' + df_in['Year'].astype(str).str.replace('.0', '', regex=False)
            # Sort key: year * 4 + season index so we can order chronologically
            season_idx = {s: i for i, s in enumerate(SEASON_ORDER)}
            df_in['_SeasonSort'] = (
                df_in['Year'].fillna(0).astype(int) * 4 +
                df_in['Season'].map(season_idx).fillna(0).astype(int)
            )
    return df_in
PRIORITY_STYLE = {
    "Strength":                         ("", "STRENGTH"),
    "Critical Gap: Market Opportunity": ("", "CRITICAL GAP"),
    "Weakness":                         ("", "WEAKNESS"),
    "On Par":                           ("", "ON PAR"),
}

def sentiment_label(score):
    """Convert a numeric sentiment score to a plain-English label."""
    if np.isnan(score):
        return "No data"
    if score > 0.5:
        return "Very positive"
    if score > 0.2:
        return "Positive"
    if score >= -0.2:
        return "Mixed"
    if score >= -0.5:
        return "Negative"
    return "Very negative"


def safe_name(s):
    return re.sub(r'[^A-Za-z0-9]+', '_', str(s)).strip('_')


def embed_html(path, height=480):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=height, scrolling=False)
        return True
    return False


SENTIMENT_FILE = os.path.join(DATA_DIR, "Hotel_Sentiment_Scores.csv")
RAW_CSV        = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "Datasets", "Hotel_Reviews.csv")

@st.cache_data
def load_data():
    df = pd.read_csv(DOCUMENT_FILE)

    # Join sentiment scores positionally (same row order as Document_Info)
    if 'Sentiment_Score' not in df.columns and os.path.exists(SENTIMENT_FILE):
        sent = pd.read_csv(SENTIMENT_FILE, usecols=['Sentiment_Label', 'Sentiment_Score'])
        df['Sentiment_Label'] = sent['Sentiment_Label'].values
        df['Sentiment_Score']  = sent['Sentiment_Score'].values

    # Load supporting tables
    benchmark_df = pd.read_csv(BENCHMARK_FILE) if os.path.exists(BENCHMARK_FILE) else pd.DataFrame()
    advice_df    = pd.read_csv(ADVICE_FILE)    if os.path.exists(ADVICE_FILE)    else pd.DataFrame()
    corr_df      = pd.read_csv(CORR_FILE)      if os.path.exists(CORR_FILE)      else pd.DataFrame()

    # Map City onto fragments from the benchmark table
    if 'City' not in df.columns and not benchmark_df.empty:
        hotel_city_map = benchmark_df.drop_duplicates('Hotel_Name').set_index('Hotel_Name')['City']
        df['City'] = df['Hotel_Name'].map(hotel_city_map)

    # Pull Month and Year from raw CSV by matching hotel name — pipeline didn't save dates
    if 'Month' not in df.columns or df['Month'].isna().all():
        if os.path.exists(RAW_CSV):
            raw = pd.read_csv(RAW_CSV, usecols=[
                'Hotel_Name', 'Review_Date', 'Reviewer_Score',
                'Negative_Review', 'Positive_Review', 'Reviewer_Nationality'
            ])
            raw['Review_Date'] = pd.to_datetime(raw['Review_Date'], errors='coerce')
            raw['Month'] = raw['Review_Date'].dt.month
            raw['Year']  = raw['Review_Date'].dt.year
            raw = raw[raw['Hotel_Name'].isin(df['Hotel_Name'].unique())]

            months_col = np.full(len(df), np.nan)
            years_col  = np.full(len(df), np.nan)
            scores_col = np.full(len(df), np.nan)
            neg_col    = np.full(len(df), '', dtype=object)
            pos_col    = np.full(len(df), '', dtype=object)
            nat_col    = np.full(len(df), '', dtype=object)

            for hotel in df['Hotel_Name'].unique():
                frag_idx  = df.index[df['Hotel_Name'] == hotel]
                hotel_raw = raw[raw['Hotel_Name'] == hotel].reset_index(drop=True)
                if len(hotel_raw) == 0:
                    continue
                idx = np.resize(np.arange(len(hotel_raw)), len(frag_idx))
                months_col[frag_idx] = hotel_raw['Month'].values[idx]
                years_col[frag_idx]  = hotel_raw['Year'].values[idx]
                scores_col[frag_idx] = hotel_raw['Reviewer_Score'].values[idx]
                neg_col[frag_idx]    = hotel_raw['Negative_Review'].fillna('').values[idx]
                pos_col[frag_idx]    = hotel_raw['Positive_Review'].fillna('').values[idx]
                nat_col[frag_idx]    = hotel_raw['Reviewer_Nationality'].fillna('').values[idx]

            df['Month']                = months_col
            df['Year']                 = years_col
            df['Reviewer_Score']       = scores_col
            df['Negative_Review']      = neg_col
            df['Positive_Review']      = pos_col
            df['Reviewer_Nationality'] = nat_col

    # Ensure optional columns exist so downstream code doesn't KeyError
    for col in ['Reviewer_Score', 'Season', 'Negative_Review', 'Positive_Review']:
        if col not in df.columns:
            df[col] = np.nan

    # Build inter-topic distance map coordinates via TF-IDF + UMAP on keywords
    coords_df = pd.DataFrame()
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from umap import UMAP

        topic_rows = df.drop_duplicates('Topic').copy()
        topic_rows = topic_rows[topic_rows['Topic'] != -1].reset_index(drop=True)

        def parse_repr(r):
            if isinstance(r, list):
                return ' '.join(r)
            try:
                return ' '.join(ast.literal_eval(str(r)))
            except Exception:
                return str(r)

        topic_rows['kw_text'] = topic_rows['Representation'].apply(parse_repr)
        counts = df.groupby('Topic').size().rename('size')
        topic_rows = topic_rows.join(counts, on='Topic')

        tfidf = TfidfVectorizer(max_features=500)
        X = tfidf.fit_transform(topic_rows['kw_text'])
        n = min(15, X.shape[0] - 1)
        umap_model = UMAP(n_components=2, n_neighbors=n, min_dist=0.3,
                          random_state=42, metric='cosine')
        xy = umap_model.fit_transform(X.toarray())

        coords_df = pd.DataFrame({
            'x':        xy[:, 0],
            'y':        xy[:, 1],
            'label':    topic_rows['Name'].values,
            'semantic': topic_rows['Semantic_Label'].values,
            'size':     topic_rows['size'].values,
            'Topic':    topic_rows['Topic'].values,
        })
        coords_df['colour'] = coords_df['semantic'].map(CATEGORY_COLOURS).fillna('#90A4AE')
    except Exception:
        pass

    return df, benchmark_df, advice_df, corr_df, coords_df


if not os.path.exists(DOCUMENT_FILE):
    st.warning("Run Main_Full_Pipeline.ipynb first to generate the data files.")
    st.stop()

df, benchmark_df, advice_df, corr_df, coords_df = load_data()
hotels = sorted(df['Hotel_Name'].dropna().unique())

# Sidebar
st.sidebar.title("Hotel Selector")
cities = sorted(df['City'].dropna().unique()) if 'City' in df.columns else []
if cities:
    city_filter = st.sidebar.selectbox("Filter by city", ["All cities"] + list(cities))
    hotel_list = sorted(df[df['City'] == city_filter]['Hotel_Name'].dropna().unique()) \
                 if city_filter != "All cities" else hotels
else:
    hotel_list = hotels
DEFAULT_HOTEL = "Park Plaza Vondelpark Amsterdam"
default_idx = hotel_list.index(DEFAULT_HOTEL) if DEFAULT_HOTEL in hotel_list else 0
selected_hotel = st.sidebar.selectbox("Choose a hotel", hotel_list, index=default_idx)
hotel_df = df[df['Hotel_Name'] == selected_hotel].copy()
hotel_city = hotel_df['City'].iloc[0] if 'City' in hotel_df.columns and len(hotel_df) else ""

# Session state
if "selected" not in st.session_state:
    st.session_state.selected = None
if "selected_review" not in st.session_state:
    st.session_state.selected_review = None


# Inter-topic distance map
def build_map(coords):
    if coords.empty:
        return None
    sizes = coords['size'].values.astype(float)
    norm_sizes = 8 + 42 * (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-9)

    fig = go.Figure()
    for category, colour in CATEGORY_COLOURS.items():
        mask = coords['semantic'] == category
        if not mask.any():
            continue
        sub  = coords[mask]
        nsub = norm_sizes[mask.values]
        fig.add_trace(go.Scatter(
            x=sub['x'], y=sub['y'],
            mode='markers',
            name=category,
            showlegend=False,
            marker=dict(size=nsub, color=colour, opacity=0.80,
                        line=dict(width=1, color='white')),
            customdata=sub['label'],
            hovertemplate='<b>%{customdata}</b><extra>' + category + '</extra>',
        ))

    # Grey bubbles for topics not mapped to a known category
    other = coords[~coords['semantic'].isin(CATEGORY_COLOURS)]
    if not other.empty:
        o_sizes = norm_sizes[~coords['semantic'].isin(CATEGORY_COLOURS).values]
        fig.add_trace(go.Scatter(
            x=other['x'], y=other['y'],
            mode='markers', name='Other',
            showlegend=False,
            marker=dict(size=o_sizes, color='#90A4AE', opacity=0.5,
                        line=dict(width=0.5, color='white')),
            customdata=other['label'],
            hovertemplate='<b>%{customdata}</b><extra>Other</extra>',
        ))

    # Place category name at the centroid of each cluster
    label_x, label_y, label_text, label_colours = [], [], [], []
    for cat, colour in CATEGORY_COLOURS.items():
        mask = coords['semantic'] == cat
        if mask.any():
            label_x.append(coords.loc[mask, 'x'].mean())
            label_y.append(coords.loc[mask, 'y'].mean())
            label_text.append(f"<b>{cat}</b>")
            label_colours.append(colour)
    fig.add_trace(go.Scatter(
        x=label_x, y=label_y,
        mode='text',
        text=label_text,
        textfont=dict(size=11, color=label_colours),
        textposition='middle center',
        hoverinfo='skip',
        showlegend=False,
    ))

    # Tighten axis ranges to actual data with 5% padding
    xpad = (coords['x'].max() - coords['x'].min()) * 0.05
    ypad = (coords['y'].max() - coords['y'].min()) * 0.05
    fig.update_layout(
        height=700,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[coords['x'].min() - xpad, coords['x'].max() + xpad]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[coords['y'].min() - ypad, coords['y'].max() + ypad]),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        clickmode='event+select',
    )
    return fig


# Home page: category sentiment bar chart
def build_category_chart(hotel_df):
    cat_summary = (
        hotel_df[hotel_df['Semantic_Label'].isin(CATEGORIES)]
        .groupby('Semantic_Label')['Sentiment_Score']
        .agg(['mean', 'count'])
        .reset_index()
        .rename(columns={'mean': 'Mean_Sentiment', 'count': 'Fragments'})
    )
    cat_summary = cat_summary.sort_values('Mean_Sentiment')
    colours = [CATEGORY_COLOURS.get(c, '#90A4AE') for c in cat_summary['Semantic_Label']]

    fig = go.Figure(go.Bar(
        x=cat_summary['Mean_Sentiment'],
        y=cat_summary['Semantic_Label'],
        orientation='h',
        marker_color=colours,
        opacity=0.85,
        customdata=cat_summary['Semantic_Label'],
        text=[f"{v:+.3f}" for v in cat_summary['Mean_Sentiment']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Sentiment: %{x:+.3f}<extra></extra>',
    ))
    fig.add_vline(x=0, line_dash='dash', line_color='grey', line_width=1)
    fig.update_layout(
        height=400, margin=dict(l=10, r=80, t=20, b=10),
        xaxis=dict(title='Guest sentiment score', range=[-1, 1], showgrid=False),
        yaxis=dict(title=''),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


# Detail view
def build_detail(semantic_label, hotel_df, benchmark_df, advice_df):
    colour  = CATEGORY_COLOURS.get(semantic_label, '#90A4AE')
    cat_df  = hotel_df[hotel_df['Semantic_Label'] == semantic_label].copy()
    n_frags = len(cat_df)
    mean_sent = cat_df['Sentiment_Score'].mean() if n_frags > 0 else float('nan')

    # Summary header
    sent_lbl = sentiment_label(mean_sent)
    st.markdown(
        f"Category: <span style='color:{colour};font-weight:bold'>{semantic_label}</span>"
        f"&nbsp;·&nbsp; **{n_frags}** reviews from this hotel"
        f"&nbsp;·&nbsp; Guest mood: **{sent_lbl}**"
        + (f"&nbsp;·&nbsp; {hotel_city}" if hotel_city else ""),
        unsafe_allow_html=True,
    )

    # Advice panel — full-width banner at the top
    if not advice_df.empty:
        adv = advice_df[
            (advice_df['Hotel_Name'] == selected_hotel) &
            (advice_df['Topic'] == semantic_label)
        ]
        if not adv.empty:
            row = adv.iloc[0]
            pri      = str(row.get('Priority', ''))
            adv_text = str(row.get('Advice', ''))
            delta    = row.get('Delta_vs_City', float('nan'))
            emoji, cap_label = PRIORITY_STYLE.get(pri, ('', pri.upper()))

            # Pull city average and hotel average from benchmark for the bottom line
            hotel_avg_str = city_avg_str = n_hotels_str = ""
            if not benchmark_df.empty:
                b_row = benchmark_df[
                    (benchmark_df['Hotel_Name'] == selected_hotel) &
                    (benchmark_df['Semantic_Label'] == semantic_label)
                ]
                if not b_row.empty:
                    hotel_avg = b_row.iloc[0]['Mean_Sentiment']
                    city_avg  = b_row.iloc[0]['City_Mean_Sentiment']
                    hotel_avg_str = f"{hotel_avg:+.3f}"
                    city_avg_str  = f"{city_avg:+.3f}"
                if hotel_city:
                    n_hotels = benchmark_df[
                        (benchmark_df['City'] == hotel_city) &
                        (benchmark_df['Semantic_Label'] == semantic_label)
                    ]['Hotel_Name'].nunique()
                    n_hotels_str = str(n_hotels)

            # Ranking language based on delta
            if not np.isnan(delta):
                if delta > 0.1:
                    rank_phrase = "above"
                elif delta < -0.1:
                    rank_phrase = "below"
                else:
                    rank_phrase = "on par with"
            else:
                rank_phrase = "compared to"

            bottom_line = (
                f"Your hotel ranks <b>{rank_phrase}</b> the city average sentiment for "
                f"<b>{semantic_label}</b>. "
                f"Average topic sentiment: <b>{hotel_avg_str}</b>. "
                f"City Average: <b>{city_avg_str}</b> across <b>{n_hotels_str}</b> hotels."
            ) if hotel_avg_str else ""

            st.markdown(
                f"<div style='background:rgba(180,180,180,0.12);border-left:5px solid rgba(180,180,180,0.12);"
                f"border-radius:8px;padding:14px 20px;margin:10px 0 16px 0'>"
                f"<div style='font-size:1.15rem;font-weight:bold;letter-spacing:0.05em;"
                f"margin-bottom:8px'>{emoji}&nbsp;&nbsp;{cap_label}</div>"
                f"<div style='margin-bottom:10px'>{adv_text}</div>"
                f"<div style='opacity:0.75;font-size:0.88rem'>{bottom_line}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # Tabs
    tab_trends, tab_city, tab_reviews, tab_sub = st.tabs(["Trends", "City Comparison", "Guest Reviews", "Subtopics"])

    # Tab 1: Trends
    with tab_trends:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Category sentiment over time (this hotel)**")
            cat_seas = add_season(cat_df)
            seas_data = (
                cat_seas[cat_seas['SeasonYear'].notna()]
                .groupby(['SeasonYear', '_SeasonSort'])['Sentiment_Score']
                .agg(['mean', 'count']).reset_index()
            )
            seas_data = seas_data[seas_data['count'] >= 3].sort_values('_SeasonSort')

            if not seas_data.empty:
                cat_mean = seas_data['mean'].mean()
                tt_fig = go.Figure(go.Scatter(
                    x=seas_data['SeasonYear'], y=seas_data['mean'],
                    mode='lines+markers',
                    line=dict(color=colour, width=2),
                    marker=dict(size=8),
                ))
                tt_fig.add_hline(y=cat_mean, line_dash='dot', line_color=colour,
                                 line_width=1.5, opacity=0.6,
                                 annotation_text=f"Avg {cat_mean:+.3f}",
                                 annotation_position="top right",
                                 annotation_font=dict(size=10, color=colour))
                tt_fig.add_hline(y=0, line_dash='dash', line_color='grey', line_width=1)
                tt_fig.update_layout(
                    height=380, margin=dict(l=10, r=10, t=30, b=10),
                    xaxis=dict(title='Season', tickangle=-45),
                    yaxis=dict(title='Guest sentiment score', range=[-1, 1]),
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(tt_fig, width="stretch")
                st.caption("Sentiment for this topic over time at your hotel.")
            else:
                st.info("Not enough seasonal data for this category at this hotel.")

        with col2:
            st.markdown("**Monthly sentiment vs city average + delta**")
            bench_path = os.path.join(
                PLOTS_DIR, "DeepDive",
                f"{safe_name(selected_hotel)}_Monthly_Benchmark_vs_{safe_name(hotel_city)}.html"
            )
            if embed_html(bench_path, height=400):
                st.caption(f"Your hotel vs the {hotel_city} city average month by month. The delta bar shows how far above or below you are.")
            else:
                st.info("Run the deep dive section of Main_Full_Pipeline.ipynb for this hotel to see the monthly benchmark chart.")

        st.divider()
        st.markdown("**Seasonal sentiment**")
        season_path = os.path.join(
            PLOTS_DIR, "DeepDive",
            f"{safe_name(selected_hotel)}_Season_Topic_Hotel.html"
        )
        if embed_html(season_path, height=700):
            st.caption(f"Seasonal breakdown for {semantic_label}. Darker blue = more positive sentiment.")
        elif 'Season' in cat_df.columns and cat_df['Season'].notna().any():
            seas = (
                cat_df.groupby('Season')['Sentiment_Score']
                .agg(['mean', 'count']).reset_index()
            )
            seas = seas[seas['count'] >= 3]
            if not seas.empty:
                s_fig = go.Figure(go.Bar(
                    x=seas['Season'], y=seas['mean'],
                    marker_color=colour, opacity=0.85,
                    text=[f"{v:+.3f}" for v in seas['mean']],
                    textposition='outside',
                ))
                s_fig.add_hline(y=0, line_dash='dash', line_color='grey', line_width=1)
                s_fig.update_layout(
                    height=340, margin=dict(l=10, r=10, t=10, b=10),
                    yaxis=dict(title='Guest sentiment score', range=[-1, 1]),
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(s_fig, width="stretch")
            else:
                st.info("Not enough seasonal data.")
        else:
            st.info("No seasonal data available.")

    # Tab 2: City Comparison
    with tab_city:
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**How you compare across all topics vs city average**")
            radar_path = os.path.join(
                PLOTS_DIR, "RadarCharts",
                f"Radar_{safe_name(selected_hotel)}.html"
            )
            if embed_html(radar_path, height=420):
                st.caption("Blue = your hotel, grey = city average. Bigger blue shape means you're ahead on those topics.")
            elif not benchmark_df.empty:
                h_bench = benchmark_df[benchmark_df['Hotel_Name'] == selected_hotel]
                if not h_bench.empty:
                    cats         = list(h_bench['Semantic_Label'])
                    hotel_vals   = list(h_bench['Mean_Sentiment'])
                    city_vals    = list(h_bench['City_Mean_Sentiment'])
                    cats_closed  = cats + [cats[0]]
                    hotel_closed = hotel_vals + [hotel_vals[0]]
                    city_closed  = city_vals  + [city_vals[0]]

                    r_fig = go.Figure()
                    r_fig.add_trace(go.Scatterpolar(
                        r=city_closed, theta=cats_closed,
                        fill='toself', name='City average',
                        line=dict(color='grey', dash='dot'), opacity=0.4,
                    ))
                    r_fig.add_trace(go.Scatterpolar(
                        r=hotel_closed, theta=cats_closed,
                        fill='toself', name=selected_hotel,
                        line=dict(color='#2196F3', width=2), opacity=0.7,
                    ))
                    r_fig.update_layout(
                        polar=dict(radialaxis=dict(range=[-1, 1], showticklabels=True)),
                        height=420, margin=dict(l=30, r=30, t=30, b=30),
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        legend=dict(font=dict(color='white')),
                    )
                    st.plotly_chart(r_fig, width="stretch")
                    st.caption("Blue = your hotel, grey = city average.")
            else:
                st.info("No benchmark data available.")

        with col4:
            st.markdown(f"**Sentiment distribution: this hotel vs {hotel_city}**")
            hotel_scores = cat_df['Sentiment_Score'].dropna().values
            city_scores  = df[
                (df['City'] == hotel_city) &
                (df['Semantic_Label'] == semantic_label) &
                (df['Hotel_Name'] != selected_hotel)
            ]['Sentiment_Score'].dropna().values if hotel_city else np.array([])

            if len(hotel_scores) >= 5:
                bins        = np.linspace(-1, 1, 21)
                bin_centres = (bins[:-1] + bins[1:]) / 2
                bar_width   = bins[1] - bins[0]

                h_counts, _ = np.histogram(hotel_scores, bins=bins)
                hotel_mean  = hotel_scores.mean()

                c_counts = np.zeros(20, dtype=int)
                city_mean = float('nan')
                has_city = len(city_scores) >= 5
                if has_city:
                    c_counts, _ = np.histogram(city_scores, bins=bins)
                    city_mean = city_scores.mean()

                def _hist_fig(counts, mean_val, bar_colour, title_text, x_title):
                    fig = go.Figure(go.Bar(
                        x=bin_centres, y=counts,
                        width=bar_width * 0.95,
                        marker_color=bar_colour, opacity=0.85,
                        hovertemplate='Sentiment: %{x:.2f}<br>Reviews: %{y}<extra></extra>',
                    ))
                    fig.add_vline(x=mean_val, line_dash='dash',
                                  line_color='white', line_width=2,
                                  annotation_text=f"Avg {sentiment_label(mean_val)} ({mean_val:+.2f})",
                                  annotation_position="top right",
                                  annotation_font=dict(size=10, color='white'))
                    fig.update_layout(
                        title=dict(text=title_text, font=dict(size=12), x=0),
                        height=230, margin=dict(l=10, r=10, t=40, b=5),
                        xaxis=dict(showgrid=False, title=x_title,
                                   range=[-1.05, 1.05], dtick=0.2),
                        yaxis=dict(title='Number of reviews', showgrid=False, autorange=True),
                        bargap=0,
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    )
                    return fig

                st.plotly_chart(
                    _hist_fig(h_counts, hotel_mean, colour, "This hotel", ''),
                    width="stretch",
                )
                if has_city:
                    st.plotly_chart(
                        _hist_fig(c_counts, city_mean, '#607D8B',
                                  f"{hotel_city} average (other hotels)", 'Sentiment score'),
                        width="stretch",
                    )
                else:
                    st.info("Not enough city data to show comparison.")

                st.caption(f"Sentiment distribution for {semantic_label}. Your hotel above, {hotel_city} below. Bars shifted right = more positive guests.")
            else:
                st.info("Not enough data for this hotel/category.")

        st.divider()
        st.markdown(f"**Peer ranking: {semantic_label} within {hotel_city}**")
        peer_path = os.path.join(
            PLOTS_DIR, "DeepDive", "PeerRanking",
            f"{safe_name(selected_hotel)}_{safe_name(semantic_label)}_PeerRank.html"
        )
        if embed_html(peer_path, height=420):
            st.caption(f"All {hotel_city} hotels ranked by sentiment for {semantic_label}. You're highlighted.")
        elif not benchmark_df.empty and hotel_city:
            city_peers = benchmark_df[
                (benchmark_df['City'] == hotel_city) &
                (benchmark_df['Semantic_Label'] == semantic_label)
            ].sort_values('Mean_Sentiment')
            if not city_peers.empty:
                bar_colours = [
                    colour if h == selected_hotel else
                    ('#F44336' if v < 0 else '#90A4AE')
                    for h, v in zip(city_peers['Hotel_Name'], city_peers['Mean_Sentiment'])
                ]
                rank = int((city_peers['Hotel_Name'] == selected_hotel).values[::-1].argmax()) + 1
                pr_fig = go.Figure(go.Bar(
                    x=city_peers['Mean_Sentiment'],
                    y=city_peers['Hotel_Name'],
                    orientation='h',
                    marker_color=bar_colours,
                    opacity=0.85,
                    text=[f"{v:+.3f}" for v in city_peers['Mean_Sentiment']],
                    textposition='outside',
                ))
                pr_fig.add_vline(x=0, line_dash='dash', line_color='grey', line_width=1)
                pr_fig.update_layout(
                    title=f"Rank #{rank} of {len(city_peers)} in {hotel_city}",
                    height=max(300, len(city_peers) * 28),
                    margin=dict(l=10, r=80, t=40, b=10),
                    xaxis=dict(range=[-1, 1], showgrid=False, title='Guest sentiment score'),
                    yaxis=dict(title=''),
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(pr_fig, width="stretch")
                st.caption(f"All {hotel_city} hotels ranked by sentiment for {semantic_label}. You're highlighted.")
            else:
                st.info("No peer data for this city/category.")
        else:
            st.info("Run the deep dive section of Main_Full_Pipeline.ipynb to see peer rankings.")

    # Tab 3: Guest Reviews
    with tab_reviews:
        st.markdown("**Sentiment vs Reviewer Score** (click a point to read the review)")
        score_df = cat_df[cat_df['Reviewer_Score'].notna()].copy()
        if len(score_df) >= 10:
            rev_cols = [c for c in ['Negative_Review', 'Positive_Review'] if c in score_df.columns]
            id_col   = 'Person_id' if 'Person_id' in score_df.columns else \
                       ('ID' if 'ID' in score_df.columns else None)
            agg_spec = {
                'Sentiment_Score': ('Sentiment_Score', 'mean'),
                'Reviewer_Score':  ('Reviewer_Score',  'first'),
            }
            for c in rev_cols:
                agg_spec[c] = (c, 'first')
            if id_col:
                agg_df = score_df.groupby(id_col).agg(**agg_spec).reset_index()
            else:
                agg_df = score_df[['Sentiment_Score', 'Reviewer_Score'] + rev_cols].copy()
                agg_df['_idx'] = range(len(agg_df))
                id_col = '_idx'
            for c in ['Negative_Review', 'Positive_Review']:
                if c not in agg_df.columns:
                    agg_df[c] = ''

            cd_cols = [id_col, 'Reviewer_Score', 'Negative_Review', 'Positive_Review']
            sc_fig = go.Figure()
            sc_fig.add_trace(go.Scatter(
                x=agg_df['Reviewer_Score'], y=agg_df['Sentiment_Score'],
                mode='markers',
                marker=dict(color=colour, opacity=0.55, size=8,
                            line=dict(width=0.5, color='white')),
                customdata=agg_df[cd_cols].values,
                hovertemplate='Score: %{x}/10<br>Sentiment: %{y:.3f}<extra></extra>',
                name='Reviews',
            ))
            valid = agg_df[agg_df['Sentiment_Score'].notna()]
            if len(valid) >= 2:
                m, b = np.polyfit(valid['Reviewer_Score'], valid['Sentiment_Score'], 1)
                x_rng = np.linspace(valid['Reviewer_Score'].min(), valid['Reviewer_Score'].max(), 60)
                sc_fig.add_trace(go.Scatter(
                    x=x_rng, y=m * x_rng + b,
                    mode='lines', line=dict(color=colour, width=2),
                    showlegend=False, hoverinfo='skip',
                ))
            sc_fig.add_hline(y=0, line_dash='dash', line_color='grey', line_width=1)
            sc_fig.update_layout(
                height=420, margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(title='Guest Score (/10)', showgrid=False),
                yaxis=dict(title='Guest sentiment'),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
            )
            sc_event = st.plotly_chart(
                sc_fig, width="stretch",
                on_select="rerun", selection_mode="points",
                key="scatter_review",
            )
            st.caption(f"Each dot is a guest. The line shows whether more positive sentiment on {semantic_label} translated into a higher score. Click to read their review.")
            if sc_event and sc_event.get("selection", {}).get("points"):
                pt = sc_event["selection"]["points"][0]
                cd = pt.get("customdata")
                if cd is not None:
                    st.session_state.selected_review = list(cd)
        else:
            st.info("Not enough score data.")

        # Review panel
        if st.session_state.selected_review:
            cd        = st.session_state.selected_review
            person_id = cd[0]
            rev_score = cd[1]
            neg_rev   = cd[2] if len(cd) > 2 else ''
            pos_rev   = cd[3] if len(cd) > 3 else ''

            st.divider()
            rcol_h, rcol_x = st.columns([8, 1])
            with rcol_h:
                st.markdown(
                    f"**Review #{person_id}**&nbsp;·&nbsp; Score: **{float(rev_score):.1f}/10**",
                    unsafe_allow_html=True,
                )
            with rcol_x:
                if st.button("Clear", key="clear_review"):
                    st.session_state.selected_review = None
                    st.rerun()

            pos_text = str(pos_rev) if pos_rev and str(pos_rev) not in ('nan', '') else '—'
            neg_text = str(neg_rev) if neg_rev and str(neg_rev) not in ('nan', '') else '—'
            st.markdown(
                f"<div style='background:rgba(255,255,255,0.05);border-radius:8px;padding:12px 16px;'>"
                f"<p style='margin:0 0 6px 0'><strong>Positive</strong></p>"
                f"<p style='margin:0 0 12px 0;font-style:italic'>{pos_text}</p>"
                f"<p style='margin:0 0 6px 0'><strong>Negative</strong></p>"
                f"<p style='margin:0;font-style:italic'>{neg_text}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # CLAUDE START — Subtopics tab
    with tab_sub:
        _hotel_name   = hotel_df['Hotel_Name'].iloc[0]
        subtopic_path = os.path.join(DATA_DIR, f"{safe_name(_hotel_name)}_Subtopic_Sentiment.csv")

        if not os.path.exists(subtopic_path):
            st.info("Subtopic breakdown not available for this hotel.")
        else:
            sub_df     = pd.read_csv(subtopic_path)
            topic_subs = sub_df[sub_df['Major_Topic'] == semantic_label].copy()

            if topic_subs.empty:
                st.info(f"No subtopic data for {semantic_label} at this hotel.")
            else:
                col_a, col_b = st.columns(2)

                with col_a:
                    donut = go.Figure(go.Pie(
                        labels=topic_subs['Subtopic'],
                        values=topic_subs['Fragment_Count'],
                        hole=0.45,
                        textinfo='label+percent',
                        textposition='inside',
                        insidetextorientation='horizontal',
                        marker=dict(colors=[
                            '#2196F3', '#9C27B0', '#4CAF50',
                            '#FF9800', '#F44336', '#00BCD4',
                        ]),
                        hovertemplate='%{label}<br>%{value} fragments (%{percent})<extra></extra>',
                    ))
                    donut.update_layout(
                        height=360, margin=dict(l=10, r=10, t=10, b=10),
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        showlegend=False,
                    )
                    st.plotly_chart(donut, use_container_width=True)
                    st.caption(f"Share of guest comments about each subtopic within {semantic_label}.")

                with col_b:
                    def _bar_colour(val):
                        if val > 0.2:
                            return '#4CAF50'
                        if val < -0.2:
                            return '#F44336'
                        return '#FFEB3B'

                    sent_bar = go.Figure(go.Bar(
                        x=topic_subs['Mean_Sentiment'],
                        y=topic_subs['Subtopic'],
                        orientation='h',
                        marker_color=[_bar_colour(v) for v in topic_subs['Mean_Sentiment']],
                        hovertemplate='%{y}: %{x:.3f}<extra></extra>',
                    ))
                    sent_bar.add_vline(x=0, line_dash='dash', line_color='grey', line_width=1)
                    sent_bar.update_layout(
                        height=360, margin=dict(l=10, r=10, t=10, b=10),
                        xaxis=dict(range=[-1, 1], title='Mean sentiment', showgrid=False),
                        yaxis=dict(title=''),
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        showlegend=False,
                    )
                    st.plotly_chart(sent_bar, use_container_width=True)
                    st.caption("Average guest sentiment for each subtopic. Red = guests were critical, green = guests were positive.")
    # CLAUDE END


# Page layout
st.title(selected_hotel)
if hotel_city:
    st.caption(f"{hotel_city}")

n_reviews = hotel_df['Person_id'].nunique() if 'Person_id' in hotel_df.columns else len(hotel_df)
avg_sent  = hotel_df['Sentiment_Score'].mean()
avg_score = hotel_df['Reviewer_Score'].mean() if 'Reviewer_Score' in hotel_df.columns else float('nan')

# At-a-glance summary strip
sent_lbl = sentiment_label(avg_sent)
score_str = f"{avg_score:.1f} / 10" if not np.isnan(avg_score) else ""
city_rank_str = ""
if not benchmark_df.empty and hotel_city:
    city_hotels = benchmark_df[benchmark_df['City'] == hotel_city]['Hotel_Name'].unique()
    city_means  = {
        h: df[df['Hotel_Name'] == h]['Sentiment_Score'].mean()
        for h in city_hotels
    }
    sorted_hotels = sorted(city_means, key=lambda h: city_means[h], reverse=True)
    if selected_hotel in sorted_hotels:
        rank = sorted_hotels.index(selected_hotel) + 1
        city_rank_str = f"Rank #{rank} of {len(sorted_hotels)} in {hotel_city}"

parts = [p for p in [score_str, city_rank_str] if p]
st.markdown(
    "<div style='background:rgba(255,255,255,0.07);border-radius:10px;"
    "padding:10px 20px;margin-bottom:8px;font-size:1.05rem'>"
    + "&nbsp;&nbsp;|&nbsp;&nbsp;".join(parts)
    + "</div>",
    unsafe_allow_html=True,
)

if st.session_state.selected is None:

    # Full-width inter-topic distance map
    map_fig = build_map(coords_df)
    if map_fig:
        st.markdown("**Topic Map:** what your guests write about")
        map_event = st.plotly_chart(map_fig, width="stretch",
                                    on_select="rerun", selection_mode="points",
                                    key="topic_map")
        if map_event and map_event.get("selection", {}).get("points"):
            pt = map_event["selection"]["points"][0]
            clicked = pt.get("customdata")
            if clicked and clicked in coords_df['label'].values:
                row = coords_df[coords_df['label'] == clicked]
                semantic = row.iloc[0]['semantic'] if not row.empty else clicked
                if semantic in CATEGORIES:
                    st.session_state.selected = semantic
                    st.rerun()
        st.caption("Each bubble is a topic. Bigger = more reviews about it. Click to explore.")

    st.divider()

    # Radar chart and category bar side by side
    radar_col, bar_col = st.columns(2)

    with radar_col:
        st.markdown("**How you compare vs your city across all topics**")
        h_bench = benchmark_df[benchmark_df['Hotel_Name'] == selected_hotel] if not benchmark_df.empty else pd.DataFrame()
        if not h_bench.empty:
            cats         = list(h_bench['Semantic_Label'])
            hotel_vals   = list(h_bench['Mean_Sentiment'])
            city_vals    = list(h_bench['City_Mean_Sentiment'])
            cats_closed  = cats + [cats[0]]
            hotel_closed = hotel_vals + [hotel_vals[0]]
            city_closed  = city_vals  + [city_vals[0]]

            r_fig = go.Figure()
            r_fig.add_trace(go.Scatterpolar(
                r=city_closed, theta=cats_closed,
                fill='toself', name='City average',
                line=dict(color='grey', dash='dot'), opacity=0.4,
            ))
            r_fig.add_trace(go.Scatterpolar(
                r=hotel_closed, theta=cats_closed,
                fill='toself', name=selected_hotel,
                line=dict(color='#2196F3', width=2), opacity=0.7,
            ))
            r_fig.update_layout(
                polar=dict(
                    radialaxis=dict(range=[-1, 1], showticklabels=True),
                    angularaxis=dict(tickfont=dict(size=11)),
                ),
                height=500, margin=dict(l=60, r=60, t=40, b=80),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(
                    font=dict(color='white', size=12),
                    orientation='h',
                    yanchor='top', y=-0.12,
                    xanchor='center', x=0.5,
                ),
            )
            r_fig.update_layout(height=500)
            st.plotly_chart(r_fig, use_container_width=True)
            st.caption("Blue = your hotel, grey = city average. Bigger blue shape means you're ahead on those topics.")

    with bar_col:
        st.markdown("**Category sentiment summary** (click a bar to explore)")
        cat_fig   = build_category_chart(hotel_df)
        cat_event = st.plotly_chart(cat_fig, width="stretch",
                                    on_select="rerun", selection_mode="points",
                                    key="cat_bar")
        if cat_event and cat_event.get("selection", {}).get("points"):
            pt = cat_event["selection"]["points"][0]
            clicked = pt.get("customdata")
            if clicked:
                st.session_state.selected = clicked
                st.rerun()
        st.caption("Positive = guests were happy with it, negative = complaints. Click a bar to see the detail.")

    st.divider()

    # Donut chart (global, HTML)
    with st.expander("Fragment distribution across categories (all hotels)"):
        donut_path = os.path.join(PLOTS_DIR, "Donut_Topic_Distribution.html")
        if not embed_html(donut_path, height=480):
            st.info("Run Main_Full_Pipeline.ipynb to generate the donut chart.")

    # Topic → reviewer score correlation
    if not corr_df.empty:
        st.divider()
        st.markdown("**Which categories drive overall guest scores most?**")
        sig = corr_df[corr_df['Significant'] == True].sort_values('Pearson_r') \
              if 'Significant' in corr_df.columns else corr_df.sort_values('Pearson_r')
        if not sig.empty:
            cr_colours = ['#F44336' if v < 0 else '#4CAF50' for v in sig['Pearson_r']]
            cr_fig = go.Figure(go.Bar(
                x=sig['Pearson_r'],
                y=sig['Topic'],
                orientation='h',
                marker_color=cr_colours,
                opacity=0.85,
                text=[f"r={v:.2f}" for v in sig['Pearson_r']],
                textposition='outside',
            ))
            cr_fig.add_vline(x=0, line_dash='dash', line_color='grey', line_width=1)
            cr_fig.update_layout(
                height=350, margin=dict(l=10, r=80, t=10, b=10),
                xaxis=dict(title='Impact on guest score', showgrid=False, range=[-1, 1]),
                yaxis=dict(title=''),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(cr_fig, width="stretch")
            st.caption("Topics with longer bars have the strongest link to your overall reviewer score.")

    # Overall hotel sentiment over time
    st.divider()
    st.markdown("**Overall hotel sentiment over time**")

    hotel_seas = add_season(hotel_df)
    overall_seas = (
        hotel_seas[hotel_seas['SeasonYear'].notna() & hotel_seas['Semantic_Label'].isin(CATEGORIES)]
        .groupby(['SeasonYear', '_SeasonSort', 'Semantic_Label'])['Sentiment_Score']
        .mean().reset_index()
    )
    overall_seas = overall_seas.sort_values('_SeasonSort')

    if not overall_seas.empty:
        ov_fig = go.Figure()
        all_labels = overall_seas.sort_values('_SeasonSort')['SeasonYear'].unique().tolist()
        for cat in CATEGORIES:
            sub = overall_seas[overall_seas['Semantic_Label'] == cat].sort_values('_SeasonSort')
            if not sub.empty:
                ov_fig.add_trace(go.Scatter(
                    x=sub['SeasonYear'], y=sub['Sentiment_Score'],
                    mode='lines+markers',
                    name=cat,
                    line=dict(color=CATEGORY_COLOURS.get(cat, '#90A4AE'), width=2),
                    marker=dict(size=7),
                ))
        hotel_mean = hotel_df['Sentiment_Score'].mean()
        ov_fig.add_hline(y=hotel_mean, line_dash='dot', line_color='white',
                         line_width=1.5, opacity=0.5,
                         annotation_text=f"Overall avg {hotel_mean:+.3f}",
                         annotation_position="top right",
                         annotation_font=dict(size=10, color='white'))
        ov_fig.add_hline(y=0, line_dash='dash', line_color='grey', line_width=1)
        ov_fig.update_layout(
            height=480, margin=dict(l=10, r=10, t=30, b=60),
            xaxis=dict(title='Season', tickangle=-45,
                       categoryorder='array', categoryarray=all_labels),
            yaxis=dict(title='Mean Sentiment', autorange=True),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                font=dict(size=10, color='white'),
                bgcolor='rgba(0,0,0,0.4)',
                itemclick='toggleothers',
                itemdoubleclick='toggle',
            ),
            annotations=[dict(
                text="Click legend to isolate · Double-click to add/remove",
                xref='paper', yref='paper', x=0, y=1.04,
                showarrow=False, font=dict(size=10, color='grey'), align='left',
            )],
        )
        st.plotly_chart(ov_fig, width="stretch")
        st.caption("Sentiment trends for each topic over time. Click the legend to isolate a line.")
    else:
        st.info("Not enough seasonal data for this hotel.")

    # Overall city ranking
    if hotel_city:
        city_df = df[df['City'] == hotel_city]
        if not city_df.empty:
            st.divider()
            st.markdown(f"**Overall guest sentiment ranking: {hotel_city}**")
            city_ranking = (
                city_df.groupby('Hotel_Name')['Sentiment_Score']
                .mean()
                .reset_index()
                .rename(columns={'Sentiment_Score': 'Mean_Sentiment'})
                .sort_values('Mean_Sentiment')
            )
            bar_colours = [
                '#2196F3' if h == selected_hotel else
                ('#F44336' if v < 0 else '#90A4AE')
                for h, v in zip(city_ranking['Hotel_Name'], city_ranking['Mean_Sentiment'])
            ]
            rank = int((city_ranking['Hotel_Name'] == selected_hotel).values[::-1].argmax()) + 1
            rank_fig = go.Figure(go.Bar(
                x=city_ranking['Mean_Sentiment'],
                y=city_ranking['Hotel_Name'],
                orientation='h',
                marker_color=bar_colours,
                opacity=0.85,
                text=[f"{v:+.3f}" for v in city_ranking['Mean_Sentiment']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Sentiment: %{x:+.3f}<extra></extra>',
            ))
            rank_fig.add_vline(x=0, line_dash='dash', line_color='grey', line_width=1)
            rank_fig.update_layout(
                title=dict(
                    text=f"Rank #{rank} of {len(city_ranking)} hotels",
                    font=dict(size=13), x=0,
                ),
                height=max(300, len(city_ranking) * 28),
                margin=dict(l=10, r=80, t=40, b=10),
                xaxis=dict(title='Guest sentiment score', showgrid=False, range=[-1, 1]),
                yaxis=dict(title=''),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(rank_fig, width="stretch")
            st.caption(f"All {hotel_city} hotels ranked by overall sentiment. Yours is in blue.")

else:
    semantic_label = st.session_state.selected
    colour = CATEGORY_COLOURS.get(semantic_label, '#90A4AE')

    col_title, col_back = st.columns([5, 1])
    with col_title:
        st.subheader(semantic_label)
    with col_back:
        if st.button("← Back to overview"):
            st.session_state.selected = None
            st.session_state.selected_review = None
            st.rerun()

    build_detail(semantic_label, hotel_df, benchmark_df, advice_df)
