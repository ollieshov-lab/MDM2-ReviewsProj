import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
from scipy import stats

st.set_page_config(page_title="Hotel Reviews — Macro Dashboard", layout="wide")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "BERTModelRawOutputs")
RAW_CSV    = os.path.join(BASE_DIR, "..", "Datasets", "Hotel_Reviews.csv")
CITY_BENCH = os.path.join(DATA_DIR, "Hotel_City_Topic_Benchmark.csv")
HTL_BENCH  = os.path.join(DATA_DIR, "Hotel_Hotel_vs_City_Benchmark.csv")
CORR_FILE  = os.path.join(DATA_DIR, "Hotel_Topic_ReviewerScore_Correlations.csv")
PLOTS_DIR  = os.path.join(DATA_DIR, "Plots")
DOC_FILE   = os.path.join(DATA_DIR, "Hotel_Document_Info.csv")
SENT_FILE  = os.path.join(DATA_DIR, "Hotel_Sentiment_Scores.csv")
SAMP_FILE  = os.path.join(BASE_DIR, "..", "Datasets", "Hotel_Reviews_StratSamp_Balanced.csv")

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
CITIES = ["Amsterdam", "Barcelona", "London", "Milan", "Paris", "Vienna"]
CITY_COLOURS = {
    "Amsterdam": "#FF6B35", "Barcelona": "#E91E63", "London":  "#2196F3",
    "Milan":     "#9C27B0", "Paris":     "#4CAF50", "Vienna":  "#FF9800",
}

MONTH_TO_SEASON = {12:"Winter",1:"Winter",2:"Winter",
                   3:"Spring",4:"Spring",5:"Spring",
                   6:"Summer",7:"Summer",8:"Summer",
                   9:"Autumn",10:"Autumn",11:"Autumn"}
SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]


# Tag parsers for hypothesis testing
def parse_trip_type(tags):
    t = str(tags)
    if 'Leisure' in t:  return 'Leisure'
    if 'Business' in t: return 'Business'
    return None

def parse_group_type(tags):
    t = str(tags)
    if 'Solo'   in t: return 'Solo'
    if 'Couple' in t: return 'Couple'
    if 'Family' in t: return 'Family'
    if 'Group'  in t: return 'Group'
    return None


# City lookup from hotel address
def extract_city(address):
    if pd.isna(address):
        return "Unknown"
    addr = str(address).lower()
    for city in CITIES:
        if city.lower() in addr:
            return city
    return "Unknown"


@st.cache_data(show_spinner="Loading full dataset (515k reviews)…")
def load_data():
    # Full raw dataset
    raw = pd.read_csv(RAW_CSV)
    raw['Review_Date'] = pd.to_datetime(raw['Review_Date'], errors='coerce')
    raw['Month']  = raw['Review_Date'].dt.month
    raw['Year']   = raw['Review_Date'].dt.year
    raw['Season'] = raw['Month'].map(MONTH_TO_SEASON)
    raw['City']   = raw['Hotel_Address'].apply(extract_city)

    # BERTopic pipeline outputs
    city_bench = pd.read_csv(CITY_BENCH) if os.path.exists(CITY_BENCH) else pd.DataFrame()
    htl_bench  = pd.read_csv(HTL_BENCH)  if os.path.exists(HTL_BENCH)  else pd.DataFrame()
    corr_df    = pd.read_csv(CORR_FILE)  if os.path.exists(CORR_FILE)  else pd.DataFrame()

    # Build enriched fragment dataframe for hypothesis testing
    frag_df = pd.DataFrame()
    if os.path.exists(DOC_FILE) and os.path.exists(SENT_FILE):
        doc_inf = pd.read_csv(DOC_FILE)
        sent    = pd.read_csv(SENT_FILE)
        doc_inf['Sentiment_Score'] = sent['Sentiment_Score'].values
        doc_inf['Sentiment_Label'] = sent['Sentiment_Label'].values

        # City: map via Hotel_Name from benchmark (reliable)
        if not htl_bench.empty:
            hotel_city_map = htl_bench.drop_duplicates('Hotel_Name').set_index('Hotel_Name')['City']
            doc_inf['City'] = doc_inf['Hotel_Name'].map(hotel_city_map)

        # Load or regenerate the balanced sample so we can join real reviewer metadata
        # Person_id in doc_inf == ID (row index) in the balanced sample
        if os.path.exists(SAMP_FILE):
            df_sampled = pd.read_csv(SAMP_FILE)
        else:
            # Regenerate deterministically — same logic as Main_Full_Pipeline.ipynb cell 4
            df = raw.copy()
            hotel_counts    = df.groupby('Hotel_Name').size()
            eligible_hotels = hotel_counts[hotel_counts >= 200].index
            df_eligible     = df[df['Hotel_Name'].isin(eligible_hotels)].copy()
            hotel_city_map_raw = df_eligible[['Hotel_Name', 'City']].drop_duplicates()
            balanced_hotels = hotel_city_map_raw.groupby('City').sample(n=30, random_state=42)
            df_city_balanced = df_eligible.merge(balanced_hotels, on=['Hotel_Name', 'City'], how='inner')
            df_sampled = (
                df_city_balanced
                .groupby(['Hotel_Name', 'City'])
                .sample(n=200, random_state=42)
                .reset_index(drop=True)
            )
            df_sampled['ID'] = df_sampled.index
            # Save it so future loads are instant
            df_sampled.to_csv(SAMP_FILE, index=False)

        # Parse trip/group type from Tags column
        df_sampled['Trip_Type']  = df_sampled['Tags'].apply(parse_trip_type)
        df_sampled['Group_Type'] = df_sampled['Tags'].apply(parse_group_type)

        # Join actual reviewer metadata to each fragment via Person_id = ID
        reviewer_meta = df_sampled[['ID', 'Reviewer_Nationality', 'Trip_Type', 'Group_Type']].copy()
        reviewer_meta['Reviewer_Nationality'] = reviewer_meta['Reviewer_Nationality'].str.strip()
        reviewer_meta = reviewer_meta.rename(columns={'ID': 'Person_id'})
        doc_inf = doc_inf.merge(reviewer_meta, on='Person_id', how='left')

        frag_df = doc_inf[doc_inf['Semantic_Label'].isin(CATEGORIES)].copy()

    return raw, city_bench, htl_bench, corr_df, frag_df


# Load
if not os.path.exists(RAW_CSV):
    st.error("Hotel_Reviews.csv not found in Datasets/. Cannot run macro dashboard.")
    st.stop()

raw, city_bench, htl_bench, corr_df, frag_df = load_data()
st.success(f"Dataset loaded — {len(raw):,} reviews across {raw['Hotel_Name'].nunique():,} hotels in {raw['City'].nunique()} cities.")

# Sidebar
st.sidebar.title("Macro Dashboard")
city_filter = st.sidebar.selectbox("Filter by city", ["All cities"] + CITIES)
filtered = raw if city_filter == "All cities" else raw[raw['City'] == city_filter]
filtered_bench = city_bench if city_filter == "All cities" else city_bench[city_bench['City'] == city_filter]
filtered_htl   = htl_bench  if city_filter == "All cities" else htl_bench[htl_bench['City'] == city_filter]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(filtered):,}** reviews")
if city_filter != "All cities":
    n_hotels = filtered['Hotel_Name'].nunique()
    avg_score = filtered['Reviewer_Score'].mean()
    st.sidebar.metric("Hotels", n_hotels)
    st.sidebar.metric("Avg reviewer score", f"{avg_score:.2f} / 10")

# Page title
st.title("Macro Review Trends" + (f" — {city_filter}" if city_filter != "All cities" else ""))

# Global stat strip
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total reviews",  f"{len(filtered):,}")
c2.metric("Hotels",         f"{filtered['Hotel_Name'].nunique():,}")
c3.metric("Cities",         str(filtered['City'].nunique()))
c4.metric("Avg score",      f"{filtered['Reviewer_Score'].mean():.2f} / 10")
c5.metric("Nationalities",  f"{filtered['Reviewer_Nationality'].nunique():,}")

st.divider()

# TAB LAYOUT
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "City Comparison", "Topic Insights", "Review Trends", "Nationality", "Hotel Rankings", "Hypothesis Testing"
])


# TAB 1: City Comparison
with tab1:
    st.subheader("Average reviewer score by city")

    city_scores = (
        raw.groupby('City')['Reviewer_Score']
        .agg(['mean', 'count'])
        .reset_index()
        .rename(columns={'mean': 'Avg_Score', 'count': 'Reviews'})
        .sort_values('Avg_Score')
    )
    city_scores = city_scores[city_scores['City'].isin(CITIES)]
    colours_bar = [CITY_COLOURS.get(c, '#90A4AE') for c in city_scores['City']]

    score_fig = go.Figure(go.Bar(
        x=city_scores['Avg_Score'],
        y=city_scores['City'],
        orientation='h',
        marker_color=colours_bar,
        text=[f"{v:.2f}" for v in city_scores['Avg_Score']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Avg score: %{x:.2f}<extra></extra>',
    ))
    score_fig.update_layout(
        height=320, margin=dict(l=10, r=60, t=10, b=10),
        xaxis=dict(title='Mean reviewer score (out of 10)', range=[0, 10.5], showgrid=False),
        yaxis=dict(title=''),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(score_fig, width="stretch")
    st.caption("Average reviewer score per city out of 10. Higher = better rated hotels overall.")

    st.divider()

    if not city_bench.empty:
        st.subheader("Sentiment per topic across cities")
        st.caption("Sentiment scores from the BERTopic run on the 10k sample. Green = guests were positive, red = negative.")

        # Pivot to city × topic heatmap
        pivot = city_bench[city_bench['City'].isin(CITIES)].pivot(
            index='City', columns='Semantic_Label', values='City_Mean_Sentiment'
        )
        pivot = pivot.reindex(columns=[c for c in CATEGORIES if c in pivot.columns])

        heat_fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale='RdYlGn',
            zmid=0,
            text=[[f"{v:+.3f}" if not np.isnan(v) else "" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont=dict(size=11),
            hovertemplate='<b>%{y}</b> · %{x}<br>Sentiment: %{z:+.3f}<extra></extra>',
            colorbar=dict(title='Sentiment', tickformat='+.2f'),
        ))
        heat_fig.update_layout(
            height=350, margin=dict(l=10, r=10, t=10, b=120),
            xaxis=dict(tickangle=-35, title=''),
            yaxis=dict(title=''),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(heat_fig, width="stretch")
        st.caption("Darker green = more positive, darker red = more critical. Blank cells had no data.")

        st.divider()

        st.subheader("Sentiment by city")

        # Short tab labels to keep the tab bar readable
        TAB_SHORT = {
            "Staff Service":                "Staff",
            "Room Comfort & Quality":       "Rooms",
            "Cleanliness":                  "Cleanliness",
            "Location & Accessibility":     "Location",
            "Breakfast & Food":             "Breakfast",
            "Bathroom & Shower Experience": "Bathroom",
            "Noise & Sleep Disturbance":    "Noise",
            "Facilities & Amenities":       "Facilities",
            "Value for Money":              "Value",
            "Maintenance & Room Condition": "Maintenance",
        }

        tab_labels = ["Overall"] + [TAB_SHORT[c] for c in CATEGORIES]
        city_tabs  = st.tabs(tab_labels)

        def city_bar(data, caption_text):
            data = data.sort_values('sentiment')
            fig = go.Figure(go.Bar(
                x=data['sentiment'],
                y=data['City'],
                orientation='h',
                marker_color=[CITY_COLOURS.get(c, '#90A4AE') for c in data['City']],
                text=[f"{v:+.3f}" for v in data['sentiment']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Sentiment: %{x:+.3f}<extra></extra>',
            ))
            fig.add_vline(x=0, line_dash='dash', line_color='grey', line_width=1)
            fig.update_layout(
                height=300, margin=dict(l=10, r=60, t=10, b=10),
                xaxis=dict(title='Mean sentiment', range=[-1, 1], showgrid=False),
                yaxis=dict(title=''),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(caption_text)

        # Overall tab — mean sentiment across all topics per city
        with city_tabs[0]:
            overall = (
                city_bench[city_bench['City'].isin(CITIES)]
                .groupby('City')['City_Mean_Sentiment']
                .mean()
                .reset_index()
                .rename(columns={'City_Mean_Sentiment': 'sentiment'})
            )
            city_bar(overall, "Average sentiment across all 10 topics per city.")

        # One tab per category
        for i, cat in enumerate(CATEGORIES):
            with city_tabs[i + 1]:
                cat_data = (
                    city_bench[
                        (city_bench['City'].isin(CITIES)) &
                        (city_bench['Semantic_Label'] == cat)
                    ]
                    .rename(columns={'City_Mean_Sentiment': 'sentiment'})
                )
                if cat_data.empty:
                    st.info("No data for this topic.")
                else:
                    city_bar(cat_data, f"Sentiment for {cat} across cities. Right of zero is positive, left is negative.")


# TAB 2: Topic Insights
with tab2:
    if not corr_df.empty:
        st.subheader("Which topics most drive reviewer scores?")
        st.caption("Correlation between topic sentiment and the reviewer's own score (10k sample)")

        sig = corr_df[corr_df['Significant'] == True].copy() if 'Significant' in corr_df.columns else corr_df.copy()
        sig = sig.sort_values('Pearson_r')
        colours_corr = ['#F44336' if v < 0 else '#4CAF50' for v in sig['Pearson_r']]

        cr_fig = go.Figure(go.Bar(
            x=sig['Pearson_r'],
            y=sig['Topic'],
            orientation='h',
            marker_color=colours_corr,
            text=[f"{v:+.3f}" for v in sig['Pearson_r']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Pearson r: %{x:+.3f}<extra></extra>',
        ))
        cr_fig.add_vline(x=0, line_dash='dash', line_color='grey', line_width=1)
        cr_fig.update_layout(
            height=420, margin=dict(l=10, r=80, t=10, b=10),
            xaxis=dict(title='Pearson r', range=[-1, 1], showgrid=False),
            yaxis=dict(title=''),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(cr_fig, width="stretch")
        st.caption("Topics with a strong positive correlation are the ones most likely to push reviewer scores up.")
        st.divider()

    if not city_bench.empty:
        st.subheader("Topic fragment volume across cities")
        st.caption("Fragment counts per topic from the BERTopic run")

        vol_pivot = city_bench[city_bench['City'].isin(CITIES)].pivot(
            index='City', columns='Semantic_Label', values='City_Fragment_Count'
        ).fillna(0)
        vol_pivot = vol_pivot.reindex(columns=[c for c in CATEGORIES if c in vol_pivot.columns])

        vol_fig = go.Figure()
        for cat in vol_pivot.columns:
            vol_fig.add_trace(go.Bar(
                name=cat,
                x=list(vol_pivot.index),
                y=vol_pivot[cat].values,
                marker_color=CATEGORY_COLOURS.get(cat, '#90A4AE'),
            ))
        vol_fig.update_layout(
            barmode='stack',
            height=380, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title='City'),
            yaxis=dict(title='Fragment count'),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(size=10, color='white'), bgcolor='rgba(0,0,0,0.4)'),
        )
        st.plotly_chart(vol_fig, width="stretch")
        st.caption("Which topics guests write about most in each city.")


# TAB 3: Review Trends
with tab3:
    st.subheader("Review volume over time")

    monthly = (
        filtered.dropna(subset=['Year', 'Month'])
        .groupby(['Year', 'Month'])
        .size().reset_index(name='Count')
    )
    monthly['Date'] = pd.to_datetime(
        monthly['Year'].astype(int).astype(str) + '-' +
        monthly['Month'].astype(int).astype(str) + '-01'
    )
    monthly = monthly.sort_values('Date')

    if not monthly.empty:
        vol_time_fig = go.Figure(go.Scatter(
            x=monthly['Date'], y=monthly['Count'],
            mode='lines+markers',
            line=dict(color='#2196F3', width=2),
            marker=dict(size=4),
            hovertemplate='%{x|%b %Y}<br>Reviews: %{y:,}<extra></extra>',
        ))
        vol_time_fig.update_layout(
            height=350, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title='Date'),
            yaxis=dict(title='Number of reviews'),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(vol_time_fig, width="stretch")
        st.caption("Review volume by month. Gaps likely reflect seasonal travel drops.")

    st.divider()
    st.subheader("Reviewer score distribution")

    col_a, col_b = st.columns(2)
    with col_a:
        score_dist = filtered['Reviewer_Score'].dropna()
        hist_fig = go.Figure(go.Histogram(
            x=score_dist,
            nbinsx=20,
            marker_color='#9C27B0',
            opacity=0.8,
        ))
        hist_fig.update_layout(
            height=320, margin=dict(l=10, r=10, t=30, b=10),
            title='Score distribution (all hotels)',
            xaxis=dict(title='Reviewer score'),
            yaxis=dict(title='Count'),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(hist_fig, width="stretch")
        st.caption("Most reviewers score between 7-10 which is pretty typical for hotel booking sites.")

    with col_b:
        # Average score per city over time (yearly)
        city_year = (
            raw[raw['City'].isin(CITIES)]
            .dropna(subset=['Year'])
            .groupby(['Year', 'City'])['Reviewer_Score']
            .mean().reset_index()
        )
        cy_fig = go.Figure()
        for city in CITIES:
            sub = city_year[city_year['City'] == city].sort_values('Year')
            if not sub.empty:
                cy_fig.add_trace(go.Scatter(
                    x=sub['Year'], y=sub['Reviewer_Score'],
                    mode='lines+markers',
                    name=city,
                    line=dict(color=CITY_COLOURS.get(city, '#90A4AE'), width=2),
                    marker=dict(size=6),
                ))
        cy_fig.update_layout(
            height=320, margin=dict(l=10, r=10, t=30, b=10),
            title='Avg score per city by year',
            xaxis=dict(title='Year'),
            yaxis=dict(title='Mean reviewer score'),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(size=10, color='white'), bgcolor='rgba(0,0,0,0.4)'),
        )
        st.plotly_chart(cy_fig, width="stretch")
        st.caption("Average reviewer score per city over time.")

    st.divider()
    st.subheader("Review volume by season")

    season_counts = (
        filtered.dropna(subset=['Season'])
        .groupby('Season').size().reset_index(name='Count')
    )
    season_counts['Season'] = pd.Categorical(
        season_counts['Season'], categories=SEASON_ORDER, ordered=True
    )
    season_counts = season_counts.sort_values('Season')

    seas_fig = go.Figure(go.Bar(
        x=season_counts['Season'],
        y=season_counts['Count'],
        marker_color=['#4CAF50', '#FF9800', '#F44336', '#2196F3'],
        text=[f"{v:,}" for v in season_counts['Count']],
        textposition='outside',
    ))
    seas_fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title='Season'),
        yaxis=dict(title='Number of reviews'),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(seas_fig, width="stretch")
    st.caption("Review volume by season — summer is usually the busiest.")


# TAB 4: Nationality
with tab4:
    st.subheader("Top nationalities by review count")

    nat_data = (
        filtered.groupby('Reviewer_Nationality')
        .agg(Review_Count=('Reviewer_Score', 'count'),
             Avg_Score=('Reviewer_Score', 'mean'))
        .reset_index()
        .sort_values('Review_Count', ascending=False)
        .head(20)
    )

    col_n1, col_n2 = st.columns(2)

    with col_n1:
        nat_fig = go.Figure(go.Bar(
            x=nat_data['Review_Count'],
            y=nat_data['Reviewer_Nationality'],
            orientation='h',
            marker_color='#2196F3',
            opacity=0.8,
            text=[f"{v:,}" for v in nat_data['Review_Count']],
            textposition='outside',
        ))
        nat_fig.update_layout(
            height=540, margin=dict(l=10, r=60, t=10, b=10),
            title='Review count (top 20)',
            xaxis=dict(title='Reviews', showgrid=False),
            yaxis=dict(title='', autorange='reversed'),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(nat_fig, width="stretch")
        st.caption("Top 20 nationalities by review count.")

    with col_n2:
        nat_score_fig = go.Figure(go.Bar(
            x=nat_data['Avg_Score'],
            y=nat_data['Reviewer_Nationality'],
            orientation='h',
            marker_color='#FF9800',
            opacity=0.8,
            text=[f"{v:.2f}" for v in nat_data['Avg_Score']],
            textposition='outside',
        ))
        nat_score_fig.update_layout(
            height=540, margin=dict(l=10, r=60, t=10, b=10),
            title='Avg reviewer score (top 20 by volume)',
            xaxis=dict(title='Mean score', range=[0, 10.5], showgrid=False),
            yaxis=dict(title='', autorange='reversed'),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(nat_score_fig, width="stretch")
        st.caption("Average score per nationality. Worth noting some nationalities tend to score more harshly than others regardless of hotel quality.")

    st.divider()
    st.subheader("Nationality score: top vs bottom reviewers")
    st.caption("Nationalities with at least 100 reviews, sorted by score")

    nat_all = (
        filtered.groupby('Reviewer_Nationality')
        .agg(Review_Count=('Reviewer_Score', 'count'),
             Avg_Score=('Reviewer_Score', 'mean'))
        .reset_index()
    )
    nat_all = nat_all[nat_all['Review_Count'] >= 100]

    col_top, col_bot = st.columns(2)
    with col_top:
        top10 = nat_all.nlargest(10, 'Avg_Score')
        tf = go.Figure(go.Bar(
            x=top10['Avg_Score'], y=top10['Reviewer_Nationality'],
            orientation='h', marker_color='#4CAF50', opacity=0.85,
            text=[f"{v:.2f}" for v in top10['Avg_Score']], textposition='outside',
        ))
        tf.update_layout(
            height=320, margin=dict(l=10, r=60, t=30, b=10),
            title='Highest scoring nationalities',
            xaxis=dict(range=[0, 10.5], showgrid=False),
            yaxis=dict(title='', autorange='reversed'),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(tf, width="stretch")
        st.caption("Highest scoring nationalities (min 100 reviews).")

    with col_bot:
        bot10 = nat_all.nsmallest(10, 'Avg_Score')
        bf = go.Figure(go.Bar(
            x=bot10['Avg_Score'], y=bot10['Reviewer_Nationality'],
            orientation='h', marker_color='#F44336', opacity=0.85,
            text=[f"{v:.2f}" for v in bot10['Avg_Score']], textposition='outside',
        ))
        bf.update_layout(
            height=320, margin=dict(l=10, r=60, t=30, b=10),
            title='Lowest scoring nationalities',
            xaxis=dict(range=[0, 10.5], showgrid=False),
            yaxis=dict(title='', autorange='reversed'),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(bf, width="stretch")
        st.caption("Lowest scoring nationalities (min 100 reviews). These groups tend to score more critically.")


# TAB 5: Hotel Rankings
with tab5:
    st.subheader("Hotel rankings by reviewer score")

    hotel_agg = (
        filtered.groupby('Hotel_Name')
        .agg(Avg_Score=('Reviewer_Score', 'mean'),
             Review_Count=('Reviewer_Score', 'count'),
             City=('City', 'first'))
        .reset_index()
    )
    hotel_agg = hotel_agg[hotel_agg['Review_Count'] >= 50]

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        top_hotels = hotel_agg.nlargest(15, 'Avg_Score')
        th_fig = go.Figure(go.Bar(
            x=top_hotels['Avg_Score'],
            y=top_hotels['Hotel_Name'],
            orientation='h',
            marker_color=[CITY_COLOURS.get(c, '#90A4AE') for c in top_hotels['City']],
            text=[f"{v:.2f}" for v in top_hotels['Avg_Score']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Score: %{x:.2f}<extra></extra>',
        ))
        th_fig.update_layout(
            height=500, margin=dict(l=10, r=60, t=30, b=10),
            title='Top 15 hotels (min 50 reviews)',
            xaxis=dict(range=[0, 10.5], showgrid=False),
            yaxis=dict(title='', autorange='reversed'),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(th_fig, width="stretch")
        st.caption("Top 15 hotels by average reviewer score (min 50 reviews). Colour = city.")

    with col_r2:
        bot_hotels = hotel_agg.nsmallest(15, 'Avg_Score')
        bh_fig = go.Figure(go.Bar(
            x=bot_hotels['Avg_Score'],
            y=bot_hotels['Hotel_Name'],
            orientation='h',
            marker_color=[CITY_COLOURS.get(c, '#90A4AE') for c in bot_hotels['City']],
            text=[f"{v:.2f}" for v in bot_hotels['Avg_Score']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Score: %{x:.2f}<extra></extra>',
        ))
        bh_fig.update_layout(
            height=500, margin=dict(l=10, r=60, t=30, b=10),
            title='Bottom 15 hotels (min 50 reviews)',
            xaxis=dict(range=[0, 10.5], showgrid=False),
            yaxis=dict(title='', autorange='reversed'),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(bh_fig, width="stretch")
        st.caption("Bottom 15 hotels by average reviewer score (min 50 reviews).")

    if not htl_bench.empty:
        st.divider()
        st.subheader("Topic sentiment ranking — all hotels")
        st.caption("BERTopic sample, colour = city.")

        rank_cat = st.selectbox("Topic", CATEGORIES, key="rank_cat")
        rank_data = htl_bench[htl_bench['Semantic_Label'] == rank_cat].sort_values('Mean_Sentiment')
        if not rank_data.empty:
            rk_fig = go.Figure(go.Bar(
                x=rank_data['Mean_Sentiment'],
                y=rank_data['Hotel_Name'],
                orientation='h',
                marker_color=[CITY_COLOURS.get(c, '#90A4AE') for c in rank_data['City']],
                text=[f"{v:+.3f}" for v in rank_data['Mean_Sentiment']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Sentiment: %{x:+.3f}<extra></extra>',
            ))
            rk_fig.add_vline(x=0, line_dash='dash', line_color='grey', line_width=1)
            rk_fig.update_layout(
                height=max(400, len(rank_data) * 22),
                margin=dict(l=10, r=80, t=10, b=10),
                xaxis=dict(title='Mean sentiment', range=[-1.2, 1.2], showgrid=False),
                yaxis=dict(title='', autorange='reversed'),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(rk_fig, width="stretch")
            st.caption(f"All hotels ranked by sentiment for {rank_cat}. Right of zero is positive. Colour = city.")

            # Colour legend for cities
            legend_html = " &nbsp; ".join(
                f"<span style='color:{CITY_COLOURS[c]};font-weight:bold'>■ {c}</span>"
                for c in CITIES
            )
            st.markdown(legend_html, unsafe_allow_html=True)

# TAB 6: Hypothesis Testing
with tab6:
    st.subheader("2-Group Hypothesis Test")
    st.caption(
        "Compare the sentiment of two groups on a specific topic. "
        "Uses Welch's t-test (unequal variance) with Mann-Whitney U as a non-parametric check. "
        "Based on the BERTopic sample (~36k reviews, 53k fragments)."
    )

    if frag_df.empty:
        st.warning("BERTopic fragment data not found. Run Main_Full_Pipeline.ipynb first.")
    else:
        # Step 1: grouping variable
        GROUP_COL_MAP = {
            "Nationality": "Reviewer_Nationality",
            "City":        "City",
            "Trip Type":   "Trip_Type",
            "Group Type":  "Group_Type",
        }
        group_by = st.selectbox("**Step 1** — Compare groups by", list(GROUP_COL_MAP.keys()))
        col_name = GROUP_COL_MAP[group_by]

        # Build list of valid options (≥30 fragments), ordered by count descending
        group_counts = frag_df[col_name].value_counts()
        valid_groups = group_counts[group_counts >= 30].index.tolist()  # already sorted by count desc

        # Display labels include the fragment count
        def fmt(name):
            return f"{name} ({group_counts[name]:,})"

        valid_labels = [fmt(g) for g in valid_groups]
        # Map display label → raw group name for filtering
        label_to_name = {fmt(g): g for g in valid_groups}

        if len(valid_groups) < 2:
            st.warning("Not enough groups with sufficient data for comparison.")
        else:
            # Step 2: pick two groups
            st.markdown("**Step 2** — Choose two groups to compare")
            col_a, col_b = st.columns(2)
            with col_a:
                label_a = st.selectbox("Group A", valid_labels, index=0)
                group_a = label_to_name[label_a]
            with col_b:
                remaining_labels = [l for l in valid_labels if l != label_a]
                label_b = st.selectbox("Group B", remaining_labels, index=0)
                group_b = label_to_name[label_b]

            # Step 3: pick topic
            topic = st.selectbox("**Step 3** — On topic", CATEGORIES)

            # Step 4: run
            if st.button("Run hypothesis test", type="primary"):
                topic_frags = frag_df[frag_df['Semantic_Label'] == topic]
                a_scores = topic_frags.loc[topic_frags[col_name] == group_a, 'Sentiment_Score'].dropna().values
                b_scores = topic_frags.loc[topic_frags[col_name] == group_b, 'Sentiment_Score'].dropna().values

                if len(a_scores) == 0:
                    st.error(f"No fragments found for **{group_a}** on topic **{topic}**.")
                elif len(b_scores) == 0:
                    st.error(f"No fragments found for **{group_b}** on topic **{topic}**.")
                else:
                    # Warnings for small samples
                    MIN_N = 30
                    if len(a_scores) < MIN_N:
                        st.warning(f"⚠️ {group_a} only has {len(a_scores)} fragments for this topic — interpret with caution.")
                    if len(b_scores) < MIN_N:
                        st.warning(f"⚠️ {group_b} only has {len(b_scores)} fragments for this topic — interpret with caution.")

                    # Statistical tests
                    t_stat, p_ttest = stats.ttest_ind(a_scores, b_scores, equal_var=False)
                    _, p_mann       = stats.mannwhitneyu(a_scores, b_scores, alternative='two-sided')

                    # Cohen's d
                    pooled_std = np.sqrt((a_scores.std()**2 + b_scores.std()**2) / 2)
                    cohens_d   = (a_scores.mean() - b_scores.mean()) / pooled_std if pooled_std > 0 else 0.0
                    d_label = (
                        "negligible" if abs(cohens_d) < 0.2 else
                        "small"      if abs(cohens_d) < 0.5 else
                        "medium"     if abs(cohens_d) < 0.8 else
                        "large"
                    )

                    # Results header
                    st.divider()
                    m1, m2, m3 = st.columns(3)
                    m1.metric("t-statistic (Welch)", f"{t_stat:+.3f}")
                    m2.metric("p-value",             f"{p_ttest:.4f}")
                    m3.metric("Cohen's d",           f"{cohens_d:+.3f} ({d_label})")

                    # Plain English verdict
                    if p_ttest < 0.05:
                        direction = group_a if a_scores.mean() > b_scores.mean() else group_b
                        st.success(
                            f"**Significant difference detected** (p = {p_ttest:.4f} < 0.05). "
                            f"**{direction}** reviewers express more positive sentiment on **{topic}**. "
                            f"Effect size is {d_label} (Cohen's d = {cohens_d:+.3f})."
                        )
                    else:
                        st.info(
                            f"**No significant difference found** (p = {p_ttest:.4f} ≥ 0.05). "
                            f"The sentiment of **{group_a}** and **{group_b}** reviewers "
                            f"on **{topic}** is not statistically distinguishable."
                        )
                    st.caption(f"Mann-Whitney U check: p = {p_mann:.4f} — {'consistent with t-test' if (p_mann < 0.05) == (p_ttest < 0.05) else 'differs from t-test (check distribution)'}")

                    # Distribution histograms (stacked vertically)
                    bins        = np.linspace(-1, 1, 21)
                    bin_centres = (bins[:-1] + bins[1:]) / 2
                    bin_labels  = [f"{c:+.1f}" for c in bin_centres]

                    a_counts, _ = np.histogram(a_scores, bins=bins)
                    b_counts, _ = np.histogram(b_scores, bins=bins)
                    a_freq = a_counts / len(a_scores)
                    b_freq = b_counts / len(b_scores)
                    y_max = max(a_freq.max(), b_freq.max()) * 1.15

                    def make_hist(scores, freq, label, colour):
                        fig = go.Figure(go.Bar(
                            x=bin_labels, y=freq,
                            name=label,
                            marker_color=colour, opacity=0.85,
                            hovertemplate='Bin %{x}<br>Proportion: %{y:.1%}<extra></extra>',
                        ))
                        fig.add_vline(
                            x=scores.mean(), line_dash='dash', line_color='white', line_width=2,
                            annotation_text=f"mean {scores.mean():+.3f}",
                            annotation_position="top right",
                            annotation_font=dict(color='white', size=10),
                        )
                        fig.update_layout(
                            bargap=0,
                            height=260, margin=dict(l=10, r=10, t=40, b=10),
                            title=f"{label} (n={len(scores):,})",
                            title_font=dict(color=colour),
                            xaxis=dict(title='', showgrid=False, range=[-1.05, 1.05]),
                            yaxis=dict(title='Proportion', tickformat='.0%', range=[0, y_max]),
                            showlegend=False,
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        )
                        return fig

                    st.plotly_chart(make_hist(a_scores, a_freq, group_a, '#2196F3'), width="stretch")
                    st.plotly_chart(make_hist(b_scores, b_freq, group_b, '#FF9800'), width="stretch")
                    st.caption("Sentiment distributions for both groups. Dashed line = mean. The more they overlap, the less different the groups actually are.")

