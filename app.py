from html import escape
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -------------------- Configuration --------------------

st.set_page_config(
    page_title="Customer Segmentation AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "customer_segmentation_model.joblib"
DATA_PATH = BASE_DIR / "data" / "Mall_Customers.csv"

INK = "#0F172A"
MUTED = "#64748B"
LINE = "#E6E8EC"
BRAND = "#0F766E"

SEGMENT_COLORS = {
    "High Income - High Spending": "#2563EB",
    "High Income - Low Spending": "#F59E0B",
    "Low Income - High Spending": "#DB2777",
    "Low Income - Low Spending": "#64748B",
    "Average Income - Average Spending": "#0F766E",
}
FALLBACK_COLORS = ["#7C3AED", "#EA580C", "#0891B2", "#65A30D", "#BE123C"]

SEGMENT_ACTIONS = {
    "High Income - High Spending": "Loyalty programs and premium offers.",
    "High Income - Low Spending": "Personalized campaigns. Look into what holds them back from buying.",
    "Low Income - High Spending": "Test affordable offers and targeted promotions.",
    "Low Income - Low Spending": "Value-oriented campaigns.",
    "Average Income - Average Spending": "Broad engagement and retention campaigns.",
}

# -------------------- Styling --------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

    html, body, [class*="st-"], .stApp, button, input, select, textarea {
        font-family: 'Manrope', system-ui, -apple-system, 'Segoe UI', sans-serif;
    }
    .stApp { background: #F6F7F9; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"], [data-testid="stDecoration"],
    #MainMenu, footer { display: none !important; }
    [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }

    .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }

    /* Header */
    .hero { padding: 0.5rem 0 1.6rem 0; }
    .hero h1 {
        font-size: 2.5rem; font-weight: 800; letter-spacing: -0.03em;
        color: #0F172A; margin: 0 0 0.5rem 0; padding: 0; line-height: 1.1;
    }
    .hero p {
        font-size: 1.08rem; color: #475569; margin: 0 0 1.1rem 0;
        max-width: 640px; line-height: 1.55;
    }
    .chips { display: flex; flex-wrap: wrap; gap: 0.5rem; }
    .chip {
        font-size: 0.82rem; font-weight: 600; color: #0F172A;
        background: #FFFFFF; border: 1px solid #E6E8EC;
        padding: 0.3rem 0.75rem; border-radius: 999px;
    }

    /* Section titles */
    .section-title {
        font-size: 1.25rem; font-weight: 700; color: #0F172A;
        letter-spacing: -0.01em; margin: 2.6rem 0 0.25rem 0;
    }
    .section-sub { color: #64748B; font-size: 0.95rem; margin: 0 0 1rem 0; }
    .panel-title { font-size: 1.05rem; font-weight: 700; color: #0F172A; margin: 0 0 0.15rem 0; }
    .panel-sub { font-size: 0.88rem; color: #64748B; margin: 0 0 0.9rem 0; }

    /* Bordered containers */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF;
        border: 1px solid #E6E8EC !important;
        border-radius: 18px !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }

    /* Sliders and labels */
    [data-testid="stSlider"] label p, .stSelectbox label p {
        font-weight: 600; color: #0F172A; font-size: 0.92rem;
    }

    /* Prediction result */
    .result {
        margin-top: 1.1rem; padding: 1.1rem 1.2rem;
        border-radius: 14px; background: #F8FAFC;
        border: 1px solid #E6E8EC; border-left: 6px solid var(--seg);
    }
    .result-label { font-size: 0.85rem; color: #64748B; font-weight: 600; }
    .result-name {
        font-size: 1.35rem; font-weight: 800; color: #0F172A;
        letter-spacing: -0.02em; margin: 0.2rem 0 0.15rem 0; line-height: 1.25;
    }
    .result-meta { font-size: 0.9rem; color: #475569; margin-bottom: 0.7rem; }
    .result-action { font-size: 0.92rem; color: #0F172A; line-height: 1.5; }

    /* KPI row */
    .kpis {
        display: grid; gap: 1rem; margin-top: 1.4rem;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    }
    .kpi {
        background: #FFFFFF; border: 1px solid #E6E8EC; border-radius: 16px;
        padding: 1.1rem 1.3rem; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .kpi-label { font-size: 0.88rem; color: #64748B; font-weight: 600; }
    .kpi-value {
        font-size: 2rem; font-weight: 800; color: #0F172A;
        letter-spacing: -0.03em; margin-top: 0.2rem;
    }

    /* Segment cards */
    .segs {
        display: grid; gap: 1rem;
        grid-template-columns: repeat(auto-fit, minmax(205px, 1fr));
    }
    .seg {
        background: #FFFFFF; border: 1px solid #E6E8EC; border-radius: 16px;
        padding: 1.15rem 1.2rem 1.2rem 1.2rem; border-top: 5px solid var(--seg);
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        display: flex; flex-direction: column; gap: 0.2rem;
    }
    .seg-name { font-size: 0.98rem; font-weight: 700; color: #0F172A; line-height: 1.3; min-height: 2.6em; }
    .seg-share {
        font-size: 2.1rem; font-weight: 800; color: var(--seg);
        letter-spacing: -0.03em; margin-top: 0.3rem; line-height: 1.1;
    }
    .seg-count { font-size: 0.85rem; color: #64748B; margin-bottom: 0.7rem; }
    .seg-row {
        display: flex; justify-content: space-between; font-size: 0.88rem;
        padding: 0.4rem 0; border-top: 1px solid #EEF0F3; color: #475569;
    }
    .seg-row b { color: #0F172A; font-weight: 700; }
    .seg-action { font-size: 0.88rem; color: #334155; line-height: 1.5; margin-top: 0.6rem; }

    .note {
        margin-top: 2.4rem; padding: 1rem 1.2rem; border-radius: 14px;
        background: #FFFFFF; border: 1px solid #E6E8EC; color: #475569;
        font-size: 0.9rem; line-height: 1.55;
    }
    .foot { margin-top: 1.2rem; color: #94A3B8; font-size: 0.82rem; }

    @media (max-width: 640px) {
        .hero h1 { font-size: 1.9rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def html_block(markup: str) -> str:
    """Flatten markup so Streamlit's markdown never treats it as a code block."""
    return "".join(line.strip() for line in markup.splitlines())


# -------------------- Load Model --------------------

@st.cache_resource
def load_artifact(path):
    return joblib.load(path)


if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

try:
    artifact = load_artifact(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

pipeline = artifact["pipeline"]
features = artifact["features"]
cluster_names = artifact["cluster_names"]

# -------------------- Load Dataset --------------------

if not DATA_PATH.exists():
    st.error(
        "Dataset not found. Place Mall_Customers.csv "
        "inside the project's data folder."
    )
    st.stop()

try:
    data = pd.read_csv(DATA_PATH)
except Exception as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()

required_columns = [
    "CustomerID",
    "Gender",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
]
missing_columns = [c for c in required_columns if c not in data.columns]
if missing_columns:
    st.error(f"Dataset is missing columns: {missing_columns}")
    st.stop()

# -------------------- Cluster Assignments --------------------

try:
    X = data[features].copy()
    data["Cluster"] = pipeline.predict(X).astype(int)
    data["Segment"] = data["Cluster"].map(cluster_names)
except Exception as exc:
    st.error(f"Could not assign customer segments: {exc}")
    st.stop()

# -------------------- Derived values --------------------

total_customers = len(data)
profiles = (
    data.groupby(["Cluster", "Segment"], as_index=False)
    .agg(
        Customers=("CustomerID", "count"),
        Average_Income=("Annual Income (k$)", "mean"),
        Average_Spending=("Spending Score (1-100)", "mean"),
    )
    .sort_values("Cluster")
)
profiles["Share"] = profiles["Customers"] / total_customers * 100

segment_names = profiles["Segment"].tolist()
color_map = {}
fallback = iter(FALLBACK_COLORS)
for name in segment_names:
    color_map[name] = SEGMENT_COLORS.get(name) or next(fallback, MUTED)

# -------------------- Header --------------------

st.markdown(
    html_block(
        f"""
        <div class="hero">
            <h1>Customer Segmentation AI</h1>
            <p>See where any customer sits on the segment map and get a
            suggested marketing approach for them.</p>
            <div class="chips">
                <span class="chip">K-Means model</span>
                <span class="chip">{profiles.shape[0]} customer segments</span>
                <span class="chip">{total_customers:,} customers analysed</span>
                <span class="chip">Silhouette score {artifact['silhouette_score']:.2f}</span>
            </div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

# -------------------- Predictor + Segment map --------------------

inc_min = int(data["Annual Income (k$)"].min())
inc_max = int(data["Annual Income (k$)"].max())
spend_min = int(data["Spending Score (1-100)"].min())
spend_max = int(data["Spending Score (1-100)"].max())

left, right = st.columns([1, 1.7], gap="large")

with left:
    with st.container(border=True):
        st.markdown(
            '<div class="panel-title">Customer profile</div>'
            '<div class="panel-sub">Move the sliders to describe a customer.</div>',
            unsafe_allow_html=True,
        )
        income = st.slider(
            "Annual income (k$)",
            min_value=inc_min,
            max_value=max(inc_max, 137),
            value=60,
        )
        spending_score = st.slider(
            "Spending score (1-100)",
            min_value=min(spend_min, 1),
            max_value=max(spend_max, 99),
            value=50,
        )

        customer = pd.DataFrame([[income, spending_score]], columns=features)
        predicted_cluster = int(pipeline.predict(customer)[0])
        predicted_segment = cluster_names[predicted_cluster]
        seg_color = color_map.get(predicted_segment, BRAND)
        seg_row = profiles[profiles["Cluster"] == predicted_cluster].iloc[0]
        action = SEGMENT_ACTIONS.get(
            predicted_segment, "Validate with campaign data before acting."
        )

        st.markdown(
            html_block(
                f"""
                <div class="result" style="--seg:{seg_color}">
                    <div class="result-label">Predicted segment</div>
                    <div class="result-name">{escape(str(predicted_segment))}</div>
                    <div class="result-meta">
                        Cluster {predicted_cluster}, {seg_row['Share']:.1f}% of customers
                    </div>
                    <div class="result-action"><b>Suggested approach:</b> {escape(action)}</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

with right:
    with st.container(border=True):
        st.markdown(
            '<div class="panel-title">Segment map</div>'
            '<div class="panel-sub">Each dot is a customer. The dark marker is the profile you selected.</div>',
            unsafe_allow_html=True,
        )

        fig = px.scatter(
            data,
            x="Annual Income (k$)",
            y="Spending Score (1-100)",
            color="Segment",
            color_discrete_map=color_map,
            category_orders={"Segment": segment_names},
            hover_data=["CustomerID", "Gender", "Age", "Cluster"],
            labels={
                "Annual Income (k$)": "Annual income (k$)",
                "Spending Score (1-100)": "Spending score",
            },
        )
        fig.update_traces(
            marker=dict(size=9, opacity=0.85, line=dict(width=1, color="white"))
        )
        fig.add_trace(
            go.Scatter(
                x=[income],
                y=[spending_score],
                mode="markers",
                name="Selected profile",
                marker=dict(
                    symbol="star",
                    size=20,
                    color=INK,
                    line=dict(width=2, color="white"),
                ),
                hovertemplate=(
                    "Selected profile<br>Income: %{x}k<br>"
                    "Spending score: %{y}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            height=470,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="white",
            font=dict(family="Manrope, sans-serif", color=INK, size=13),
            legend=dict(
                title_text="",
                orientation="h",
                yanchor="top",
                y=-0.18,
                xanchor="left",
                x=0,
            ),
            xaxis=dict(gridcolor="#EEF0F3", zeroline=False, linecolor=LINE),
            yaxis=dict(gridcolor="#EEF0F3", zeroline=False, linecolor=LINE),
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

# -------------------- Dataset overview --------------------

average_income = data["Annual Income (k$)"].mean()
average_spending = data["Spending Score (1-100)"].mean()

st.markdown(
    html_block(
        f"""
        <div class="kpis">
            <div class="kpi"><div class="kpi-label">Total customers</div>
                <div class="kpi-value">{total_customers:,}</div></div>
            <div class="kpi"><div class="kpi-label">Segments found</div>
                <div class="kpi-value">{data['Cluster'].nunique()}</div></div>
            <div class="kpi"><div class="kpi-label">Average annual income</div>
                <div class="kpi-value">${average_income:.1f}k</div></div>
            <div class="kpi"><div class="kpi-label">Average spending score</div>
                <div class="kpi-value">{average_spending:.1f}</div></div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

# -------------------- Segment profiles --------------------

st.markdown(
    '<div class="section-title">Segment profiles</div>'
    '<div class="section-sub">Who is in each segment and how to approach them.</div>',
    unsafe_allow_html=True,
)

cards = []
for _, row in profiles.iterrows():
    name = str(row["Segment"])
    color = color_map.get(name, MUTED)
    cards.append(
        f"""
        <div class="seg" style="--seg:{color}">
            <div class="seg-name">{escape(name)}</div>
            <div class="seg-share">{row['Share']:.1f}%</div>
            <div class="seg-count">{int(row['Customers'])} customers</div>
            <div class="seg-row"><span>Avg. income</span><b>${row['Average_Income']:.1f}k</b></div>
            <div class="seg-row"><span>Avg. spending</span><b>{row['Average_Spending']:.1f}</b></div>
            <div class="seg-action">{escape(SEGMENT_ACTIONS.get(name, ''))}</div>
        </div>
        """
    )

st.markdown(
    html_block(f'<div class="segs">{"".join(cards)}</div>'),
    unsafe_allow_html=True,
)

# -------------------- Customer records --------------------

st.markdown(
    '<div class="section-title">Customer records</div>'
    '<div class="section-sub">Filter by segment and export the result.</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    f1, f2 = st.columns([3, 1], vertical_alignment="bottom")
    with f1:
        selected_segment = st.selectbox(
            "Segment",
            ["All segments"] + sorted(data["Segment"].unique().tolist()),
        )

    filtered_data = data.copy()
    if selected_segment != "All segments":
        filtered_data = filtered_data[filtered_data["Segment"] == selected_segment]

    with f2:
        st.download_button(
            "Download CSV",
            data=filtered_data.to_csv(index=False).encode("utf-8"),
            file_name="customer_segments.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.dataframe(filtered_data, use_container_width=True, hide_index=True)

# -------------------- Note + footer --------------------

st.markdown(
    '<div class="note">These are suggested marketing hypotheses, not proven '
    "outcomes. Validate them with your own campaign and sales data before "
    "acting on them.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="foot">Model: K-Means with five clusters, trained on annual '
    "income and spending score.</div>",
    unsafe_allow_html=True,
)
