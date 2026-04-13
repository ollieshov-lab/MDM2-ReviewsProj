# CLAUDE START
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import ast

st.set_page_config(page_title="Hotel Review Analysis", layout="wide")

DATA_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BERTModelRawOutputs")
ENRICHED_FILE  = os.path.join(DATA_DIR, "Hotel_Enriched.csv")
COORDS_FILE    = os.path.join(DATA_DIR, "Hotel_Topic_Coords.csv")
TOPIC_FILE     = os.path.join(DATA_DIR, "Hotel_Topic_Info.csv")
LABELS_FILE    = os.path.join(DATA_DIR, "Hotel_Topic_Labels.csv")
SAMPLE_FILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "Datasets", "Hotel_Reviews_StratSamp_10000.csv")

CATEGORIES = [
    "Staff Service", "Room Comfort & Quality", "Cleanliness",
    "Location & Accessibility", "Breakfast & Food",
    "Bathroom & Shower Experience", "Noise & Sleep Disturbance",
    "Facilities & Amenities", "Value for Money", "Maintenance & Room Condition",
]
CATEGORY_COLOURS = {
    "Staff Service":               "#2196F3",
    "Room Comfort & Quality":      "#9C27B0",
    "Cleanliness":                 "#4CAF50",
    "Location & Accessibility":    "#FF9800",
    "Breakfast & Food":            "#F44336",
    "Bathroom & Shower Experience":"#00BCD4",
    "Noise & Sleep Disturbance":   "#795548",
    "Facilities & Amenities":      "#607D8B",
    "Value for Money":             "#FFEB3B",
    "Maintenance & Room Condition":"#E91E63",
}
MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}


@st.cache_data
def load_data():
    df = pd.read_csv(ENRICHED_FILE)
    if os.path.exists(SAMPLE_FILE):
        samp = pd.read_csv(SAMPLE_FILE, usecols=['ID', 'Negative_Review', 'Positive_Review'])
        df = df.merge(samp, left_on='Person_id', right_on='ID', how='left')
        drop_cols = [c for c in ['ID', 'ID_x', 'ID_y'] if c in df.columns]
        df = df.drop(columns=drop_cols)
    else:
        df['Negative_Review'] = ''
        df['Positive_Review'] = ''
    coords = pd.read_csv(COORDS_FILE)
    topics = pd.read_csv(TOPIC_FILE)
    topics = topics[topics['Topic'] != -1].reset_index(drop=True)
    topics['Representation'] = topics['Representation'].apply(
        lambda r: ast.literal_eval(r) if isinstance(r, str) else r
    )
    label_map = {}
    if os.path.exists(LABELS_FILE):
        label_map = pd.read_csv(LABELS_FILE).set_index('Name')['Label'].to_dict()
    coords = coords[coords['label'].astype(str) != '-1'].reset_index(drop=True)
    coords['semantic'] = coords['label'].map(label_map).fillna('Other')
    coords['colour']   = coords['semantic'].map(CATEGORY_COLOURS).fillna('#90A4AE')
    return df, coords, topics

if not os.path.exists(ENRICHED_FILE):
    st.warning("Run Main2.ipynb first to generate the data.")
    st.stop()

df, coords_df, topics_df = load_data()
hotels = sorted(df['Hotel_Name'].unique())

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Hotel Selector")
selected_hotel = st.sidebar.selectbox("Choose a hotel", hotels)
hotel_df = df[df['Hotel_Name'] == selected_hotel].copy()

# ── Session state ─────────────────────────────────────────────────────────────
if "selected" not in st.session_state:
    st.session_state.selected = None
if "selected_review" not in st.session_state:
    st.session_state.selected_review = None


# ── Build inter-topic distance map ────────────────────────────────────────────
def build_map(coords):
    sizes = coords['size'].values.astype(float)
    norm_sizes = 8 + 42 * (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-9)
    size_threshold = np.sort(sizes)[-25] if len(sizes) > 25 else 0

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
            marker=dict(size=nsub, color=colour, opacity=0.80,
                        line=dict(width=1, color='white')),
            customdata=sub['label'],
            hovertemplate='<b>%{customdata}</b><extra>' + category + '</extra>',
        ))

    labelled = coords[sizes >= size_threshold]
    fig.add_trace(go.Scatter(
        x=labelled['x'], y=labelled['y'],
        mode='text',
        text=labelled['semantic'],
        textfont=dict(size=9, color='white', family='Arial Black'),
        showlegend=False, hoverinfo='skip',
        customdata=labelled['label'],
    ))

    fig.update_layout(
        height=580,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(title='Category', itemsizing='constant',
                    bgcolor='rgba(0,0,0,0.4)', font=dict(color='white')),
        clickmode='event+select',
    )
    return fig


# ── Detail view for a clicked topic ──────────────────────────────────────────
def build_detail(raw_label, hotel_df, topics_df):
    row = topics_df[topics_df['Name'] == raw_label]
    if row.empty:
        st.warning("Topic data not found.")
        return
    row      = row.iloc[0]
    keywords = row['Representation']
    count    = int(row['Count'])
    semantic = row.get('Semantic_Label') if pd.notna(row.get('Semantic_Label', None)) else None
    colour   = CATEGORY_COLOURS.get(semantic, '#90A4AE')

    # Fragment count + hotel sentiment summary
    topic_hotel = hotel_df[hotel_df['Name'] == raw_label]
    cat_hotel   = hotel_df[hotel_df['Semantic_Label'] == semantic] if semantic else pd.DataFrame()
    hotel_sent  = topic_hotel['Sentiment_Score'].mean()
    hotel_frags = len(topic_hotel)

    st.markdown(
        f"Category: <span style='color:{colour};font-weight:bold'>{semantic or raw_label}</span> &nbsp;·&nbsp; "
        f"**{count:,}** total fragments &nbsp;·&nbsp; "
        f"**{hotel_frags}** from this hotel &nbsp;·&nbsp; "
        f"Hotel sentiment: **{hotel_sent:+.3f}**" if hotel_frags > 0 else
        f"Category: <span style='color:{colour};font-weight:bold'>{semantic or raw_label}</span>",
        unsafe_allow_html=True
    )

    st.divider()

    # ── Row 1: Keywords + Topic sentiment over time ───────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top keywords**")
        n = len(keywords)
        kw_fig = go.Figure(go.Bar(
            x=list(range(n, 0, -1)), y=keywords,
            orientation='h', marker_color=colour, opacity=0.85,
        ))
        kw_fig.update_layout(
            height=300, margin=dict(l=10, r=20, t=10, b=10),
            xaxis=dict(showgrid=False, showticklabels=False, title=''),
            yaxis=dict(autorange='reversed'),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(kw_fig, width="stretch")

    with col2:
        st.markdown("**Topic sentiment over time — this hotel**")
        if not topic_hotel.empty and topic_hotel['Month'].notna().sum() >= 2:
            t_time = (
                topic_hotel[topic_hotel['Month'].notna()]
                .groupby('Month')['Sentiment_Score']
                .agg(['mean', 'count'])
                .reset_index()
            )
            t_time = t_time[t_time['count'] >= 2]
            t_time['Month_Name'] = t_time['Month'].map(MONTH_NAMES)
            t_time = t_time.sort_values('Month')

            tt_fig = go.Figure()
            tt_fig.add_trace(go.Scatter(
                x=t_time['Month_Name'], y=t_time['mean'],
                mode='lines+markers',
                line=dict(color=colour, width=2),
                marker=dict(size=8),
                name='This topic',
            ))
            # Also show category average as a faint reference line
            if not cat_hotel.empty:
                c_time = (
                    cat_hotel[cat_hotel['Month'].notna()]
                    .groupby('Month')['Sentiment_Score'].mean()
                    .reset_index()
                )
                c_time['Month_Name'] = c_time['Month'].map(MONTH_NAMES)
                c_time = c_time.sort_values('Month')
                tt_fig.add_trace(go.Scatter(
                    x=c_time['Month_Name'], y=c_time['Sentiment_Score'],
                    mode='lines', line=dict(color=colour, width=1, dash='dot'),
                    opacity=0.4, name='Category avg',
                ))
            topic_mean = t_time['mean'].mean()
            tt_fig.add_hline(y=topic_mean, line_dash='dot', line_color=colour,
                             line_width=1.5, opacity=0.6,
                             annotation_text=f"Topic avg {topic_mean:+.3f}",
                             annotation_position="top right",
                             annotation_font=dict(size=10, color=colour))
            tt_fig.add_hline(y=0, line_dash='dash', line_color='grey', line_width=1)
            tt_fig.update_layout(
                height=420, margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(title='Month', categoryorder='array',
                           categoryarray=list(MONTH_NAMES.values())),
                yaxis=dict(title='Mean Sentiment', autorange=True),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
            )
            st.plotly_chart(tt_fig, width="stretch")
        else:
            st.info("Not enough monthly data for this topic at this hotel.")

    st.divider()

    # ── Row 2: Category sentiment over time (all topics in category) ──────────
    if not cat_hotel.empty and semantic:
        st.markdown(f"**{semantic} — all topics sentiment over time**")
        cat_time = (
            cat_hotel[cat_hotel['Month'].notna()]
            .groupby(['Month', 'Name'])['Sentiment_Score']
            .mean().reset_index()
        )
        cat_time['Month_Name'] = cat_time['Month'].map(MONTH_NAMES)
        cat_time = cat_time.sort_values('Month')

        if not cat_time.empty:
            ct_fig = go.Figure()
            for topic_name, grp in cat_time.groupby('Name'):
                if len(grp) >= 2:
                    short = topic_name.split('_', 1)[-1].replace('_', ' ')[:30]
                    ct_fig.add_trace(go.Scatter(
                        x=grp['Month_Name'], y=grp['Sentiment_Score'],
                        mode='lines+markers', name=short,
                        line=dict(width=1.5), marker=dict(size=5), opacity=0.8,
                    ))

            # Category average line across all topics per month
            cat_avg = (
                cat_hotel[cat_hotel['Month'].notna()]
                .groupby('Month')['Sentiment_Score'].mean().reset_index()
            )
            cat_avg['Month_Name'] = cat_avg['Month'].map(MONTH_NAMES)
            cat_avg = cat_avg.sort_values('Month')
            cat_mean = cat_avg['Sentiment_Score'].mean()
            ct_fig.add_trace(go.Scatter(
                x=cat_avg['Month_Name'], y=cat_avg['Sentiment_Score'],
                mode='lines+markers', name='Category average',
                line=dict(color=colour, width=3),
                marker=dict(size=8),
            ))
            ct_fig.add_hline(y=cat_mean, line_dash='dot', line_color=colour,
                             line_width=1.5, opacity=0.6,
                             annotation_text=f"Overall avg {cat_mean:+.3f}",
                             annotation_position="top right",
                             annotation_font=dict(size=10, color=colour))
            ct_fig.add_hline(y=0, line_dash='dash', line_color='grey', line_width=1)
            ct_fig.update_layout(
                height=480, margin=dict(l=10, r=10, t=30, b=10),
                xaxis=dict(title='Month', categoryorder='array',
                           categoryarray=list(MONTH_NAMES.values())),
                yaxis=dict(title='Mean Sentiment', autorange=True),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(
                    font=dict(size=9, color='white'),
                    bgcolor='rgba(0,0,0,0.4)',
                    itemclick='toggleothers',
                    itemdoubleclick='toggle',
                ),
                annotations=[dict(
                    text="Click legend to isolate · Double-click to add/remove",
                    xref='paper', yref='paper', x=0, y=1.06,
                    showarrow=False, font=dict(size=10, color='grey'), align='left',
                )],
            )
            st.plotly_chart(ct_fig, width="stretch")

    st.divider()

    # ── Row 3: Trip type + Group type heatmaps ────────────────────────────────
    st.markdown(f"**{semantic} — guest type breakdown**")
    col3, col4 = st.columns(2)

    def cat_heatmap(data, group_col, title, min_count=5):
        sub = data[data[group_col].notna() & (data[group_col] != 'Other')]
        pivot = (
            sub.groupby(group_col)['Sentiment_Score']
            .agg(['mean', 'count']).reset_index()
        )
        pivot = pivot[pivot['count'] >= min_count]
        if pivot.empty:
            return None
        fig = go.Figure(go.Bar(
            x=pivot['mean'], y=pivot[group_col],
            orientation='h', marker_color=colour, opacity=0.8,
            text=[f"{v:+.3f}" for v in pivot['mean']],
            textposition='outside',
        ))
        fig.add_vline(x=0, line_dash='dash', line_color='grey', line_width=1)
        fig.update_layout(
            title=title, height=250,
            margin=dict(l=10, r=60, t=30, b=10),
            xaxis=dict(range=[-0.7, 0.7], showgrid=False, title=''),
            yaxis=dict(title=''),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        )
        return fig

    with col3:
        fig_trip = cat_heatmap(cat_hotel, 'Trip_Type', 'By Trip Type')
        if fig_trip:
            st.plotly_chart(fig_trip, width="stretch")
        else:
            st.info("Not enough trip type data.")

    with col4:
        fig_group = cat_heatmap(cat_hotel, 'Group_Type', 'By Group Type')
        if fig_group:
            st.plotly_chart(fig_group, width="stretch")
        else:
            st.info("Not enough group type data.")

    st.divider()

    # ── Row 4: Nationality + Score correlation ────────────────────────────────
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("**By Reviewer Nationality**")
        nat_counts = cat_hotel['Reviewer_Nationality'].value_counts()
        top_nats   = nat_counts[nat_counts >= 5].head(8).index
        nat_df = (
            cat_hotel[cat_hotel['Reviewer_Nationality'].isin(top_nats)]
            .groupby('Reviewer_Nationality')['Sentiment_Score']
            .agg(['mean', 'count']).reset_index()
        )
        nat_df = nat_df[nat_df['count'] >= 5].sort_values('mean')
        if not nat_df.empty:
            nat_fig = go.Figure(go.Bar(
                x=nat_df['mean'], y=nat_df['Reviewer_Nationality'],
                orientation='h', marker_color=colour, opacity=0.8,
                text=[f"{v:+.3f}" for v in nat_df['mean']],
                textposition='outside',
            ))
            nat_fig.add_vline(x=0, line_dash='dash', line_color='grey', line_width=1)
            nat_fig.update_layout(
                height=300, margin=dict(l=10, r=60, t=10, b=10),
                xaxis=dict(range=[-0.7, 0.7], showgrid=False, title=''),
                yaxis=dict(title=''),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(nat_fig, width="stretch")
        else:
            st.info("Not enough nationality data.")

    with col6:
        st.markdown("**Sentiment vs Reviewer Score** — click a point to read the review")
        score_df = cat_hotel[cat_hotel['Reviewer_Score'].notna()].copy()
        if len(score_df) >= 10:
            rev_cols = [c for c in ['Negative_Review', 'Positive_Review'] if c in score_df.columns]
            agg_spec = {
                'Sentiment_Score': ('Sentiment_Score', 'mean'),
                'Reviewer_Score':  ('Reviewer_Score',  'first'),
                'Reviewer_Nationality': ('Reviewer_Nationality', 'first'),
                'Trip_Type':  ('Trip_Type',  'first'),
                'Group_Type': ('Group_Type', 'first'),
            }
            for c in rev_cols:
                agg_spec[c] = (c, 'first')
            agg_df = score_df.groupby('Person_id').agg(**agg_spec).reset_index()
            for c in ['Negative_Review', 'Positive_Review']:
                if c not in agg_df.columns:
                    agg_df[c] = ''

            cd_cols = ['Person_id', 'Reviewer_Score', 'Reviewer_Nationality',
                       'Trip_Type', 'Group_Type', 'Negative_Review', 'Positive_Review']
            sc_fig = go.Figure()
            sc_fig.add_trace(go.Scatter(
                x=agg_df['Reviewer_Score'], y=agg_df['Sentiment_Score'],
                mode='markers',
                marker=dict(color=colour, opacity=0.55, size=8,
                            line=dict(width=0.5, color='white')),
                customdata=agg_df[cd_cols].values,
                hovertemplate=(
                    'Score: %{x}/10<br>Sentiment: %{y:.3f}<br>'
                    '%{customdata[2]}<extra></extra>'
                ),
                name='Reviews',
            ))
            # OLS trendline via numpy
            valid = agg_df[agg_df['Sentiment_Score'].notna()]
            if len(valid) >= 2:
                m, b = np.polyfit(valid['Reviewer_Score'], valid['Sentiment_Score'], 1)
                x_rng = np.linspace(valid['Reviewer_Score'].min(), valid['Reviewer_Score'].max(), 60)
                sc_fig.add_trace(go.Scatter(
                    x=x_rng, y=m * x_rng + b,
                    mode='lines', line=dict(color=colour, width=2),
                    name='Trend', showlegend=False, hoverinfo='skip',
                ))
            sc_fig.add_hline(y=0, line_dash='dash', line_color='grey', line_width=1)
            sc_fig.update_layout(
                height=300, margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(title='Guest Score (/10)',
                           showgrid=False),
                yaxis=dict(title='Our Sentiment'),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
            )
            sc_event = st.plotly_chart(
                sc_fig, width="stretch",
                on_select="rerun", selection_mode="points",
                key="scatter_review",
            )
            if sc_event and sc_event.get("selection", {}).get("points"):
                pt = sc_event["selection"]["points"][0]
                cd = pt.get("customdata")
                if cd is not None:
                    st.session_state.selected_review = list(cd)
        else:
            st.info("Not enough score data.")

    # ── Review panel ──────────────────────────────────────────────────────────
    if st.session_state.selected_review:
        cd = st.session_state.selected_review
        person_id, rev_score, nationality, trip_type, group_type, neg_rev, pos_rev = cd
        st.divider()
        rcol_h, rcol_x = st.columns([8, 1])
        with rcol_h:
            st.markdown(
                f"**Review #{int(person_id) if str(person_id).replace('.','',1).isdigit() else person_id}**"
                f"&nbsp;·&nbsp; Score: **{float(rev_score):.1f}/10**"
                f"&nbsp;·&nbsp; {nationality}"
                f"&nbsp;·&nbsp; {trip_type}"
                f"&nbsp;·&nbsp; {group_type}",
                unsafe_allow_html=True,
            )
        with rcol_x:
            if st.button("✕ Clear", key="clear_review"):
                st.session_state.selected_review = None
                st.rerun()

        pos_text = str(pos_rev) if pos_rev and str(pos_rev) not in ('nan', '') else '—'
        neg_text = str(neg_rev) if neg_rev and str(neg_rev) not in ('nan', '') else '—'
        st.markdown(
            f"<div style='background:rgba(255,255,255,0.05);border-radius:8px;padding:12px 16px;'>"
            f"<p style='margin:0 0 6px 0'><strong>✅ Positive</strong></p>"
            f"<p style='margin:0 0 12px 0;font-style:italic'>{pos_text}</p>"
            f"<p style='margin:0 0 6px 0'><strong>❌ Negative</strong></p>"
            f"<p style='margin:0;font-style:italic'>{neg_text}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Page layout ───────────────────────────────────────────────────────────────
st.title(selected_hotel)

n_reviews = hotel_df['Person_id'].nunique()
avg_sent  = hotel_df['Sentiment_Score'].mean()
avg_score = hotel_df['Reviewer_Score'].mean()
st.caption(
    f"{n_reviews} reviews · Mean sentiment: **{avg_sent:+.3f}** · "
    f"Mean reviewer score: **{avg_score:.1f}/10**"
)

if st.session_state.selected is None:
    st.caption(
        "Each bubble is a topic discovered from hotel reviews. "
        "Colour = category · Size = fragment count · "
        "Click a bubble to explore that topic for this hotel."
    )

    # Legend strip
    cols = st.columns(len(CATEGORY_COLOURS))
    for col, (cat, colour) in zip(cols, CATEGORY_COLOURS.items()):
        col.markdown(
            f"<span style='background:{colour};color:white;padding:2px 6px;"
            f"border-radius:4px;font-size:10px'>{cat}</span>",
            unsafe_allow_html=True
        )
    st.write("")

    fig = build_map(coords_df)
    event = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points")

    if event and event.get("selection", {}).get("points"):
        pt = event["selection"]["points"][0]
        clicked = pt.get("customdata")
        if clicked:
            st.session_state.selected = clicked
            st.rerun()

    # ── Overall hotel sentiment over time ─────────────────────────────────────
    st.divider()
    st.markdown("**Overall hotel sentiment over time**")

    overall_time = (
        hotel_df[hotel_df['Month'].notna() & hotel_df['Semantic_Label'].isin(CATEGORIES)]
        .groupby(['Month', 'Semantic_Label'])['Sentiment_Score']
        .mean().reset_index()
    )
    overall_time['Month_Name'] = overall_time['Month'].map(MONTH_NAMES)
    overall_time = overall_time.sort_values('Month')

    hotel_monthly_avg = (
        hotel_df[hotel_df['Month'].notna()]
        .groupby('Month')['Sentiment_Score'].mean().reset_index()
    )
    hotel_monthly_avg['Month_Name'] = hotel_monthly_avg['Month'].map(MONTH_NAMES)
    hotel_monthly_avg = hotel_monthly_avg.sort_values('Month')

    if not overall_time.empty:
        ov_fig = go.Figure()

        for cat in CATEGORIES:
            sub = overall_time[overall_time['Semantic_Label'] == cat]
            if len(sub) >= 2:
                ov_fig.add_trace(go.Scatter(
                    x=sub['Month_Name'], y=sub['Sentiment_Score'],
                    mode='lines+markers', name=cat,
                    line=dict(color=CATEGORY_COLOURS.get(cat, '#90A4AE'), width=1.5),
                    marker=dict(size=5), opacity=0.8,
                ))

        # Bold overall average line on top
        ov_fig.add_trace(go.Scatter(
            x=hotel_monthly_avg['Month_Name'], y=hotel_monthly_avg['Sentiment_Score'],
            mode='lines+markers', name='Hotel average',
            line=dict(color='white', width=3),
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
            height=480, margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(title='Month', categoryorder='array',
                       categoryarray=list(MONTH_NAMES.values())),
            yaxis=dict(title='Mean Sentiment', autorange=True),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                font=dict(size=10, color='white'),
                bgcolor='rgba(0,0,0,0.4)',
                itemclick='toggleothers',      # single click isolates that line
                itemdoubleclick='toggle',      # double click adds/removes a line
            ),
            annotations=[dict(
                text="Click legend to isolate · Double-click to add/remove",
                xref='paper', yref='paper', x=0, y=1.04,
                showarrow=False, font=dict(size=10, color='grey'),
                align='left',
            )],
        )
        st.plotly_chart(ov_fig, width="stretch")
    else:
        st.info("Not enough monthly data for this hotel.")

else:
    row = topics_df[topics_df['Name'] == st.session_state.selected]
    semantic = row.iloc[0].get('Semantic_Label', '') if not row.empty else ''
    colour   = CATEGORY_COLOURS.get(semantic, '#90A4AE')

    col_title, col_back = st.columns([5, 1])
    with col_title:
        st.subheader(
            f"{'  ●  '}{st.session_state.selected.split('_', 1)[-1].replace('_', ' ').title()}"
        )
    with col_back:
        if st.button("← Back to map"):
            st.session_state.selected = None
            st.session_state.selected_review = None
            st.rerun()

    build_detail(st.session_state.selected, hotel_df, topics_df)
# CLAUDE END
