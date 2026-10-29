from html import escape
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Customer Segmentation AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "customer_segmentation_model.joblib"
DATA_PATH = BASE_DIR / "data" / "Mall_Customers.csv"

INK, MUTED, LINE = "#0F172A", "#64748B", "#E7E9F0"

# name -> (persona, color, goal, action, KPI to watch)
SEGMENTS = {
    "High Income - High Spending": ("Champions", "#E11D48", "Protect and grow",
        "Loyalty programs and premium offers.", "Repeat purchase rate"),
    "High Income - Low Spending": ("Untapped Potential", "#F59E0B", "Win more of their wallet",
        "Personalized campaigns. Look into what holds them back from buying.", "Conversion rate"),
    "Low Income - High Spending": ("Eager Spenders", "#9333EA", "Reward without over-discounting",
        "Test affordable offers and targeted promotions.", "Promo response rate"),
    "Low Income - Low Spending": ("Budget Conscious", "#64748B", "Serve efficiently",
        "Value-oriented campaigns.", "Cost per conversion"),
    "Average Income - Average Spending": ("Core Shoppers", "#0D9488", "Keep engaged",
        "Broad engagement and retention campaigns.", "Retention rate"),
}
FALLBACK = ["#7C3AED", "#EA580C", "#0891B2", "#65A30D", "#BE123C"]

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="st-"], .stApp, button, input, select, textarea {
  font-family: 'Manrope', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.stApp, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(900px 520px at 4% -5%, rgba(251,146,60,.28), transparent 60%),
    radial-gradient(800px 520px at 100% 8%, rgba(244,63,94,.18), transparent 60%),
    radial-gradient(900px 600px at 55% 112%, rgba(168,85,247,.14), transparent 60%),
    linear-gradient(180deg, #FFF6EE 0%, #FFF1F2 52%, #FAF5FF 100%);
  background-attachment: fixed; }
[data-testid="stMain"], [data-testid="stMainBlockContainer"], .main { background: transparent; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer,
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
.block-container { max-width: 1240px; padding-top: 1.2rem; padding-bottom: 4rem; }

.topbar { display:flex; justify-content:space-between; align-items:center; padding:.4rem 0 1rem 0; }
.brand { display:flex; align-items:center; gap:.65rem; font-weight:800; font-size:1.05rem; color:#0F172A; letter-spacing:-.01em; }
.status { display:flex; align-items:center; gap:.45rem; font-size:.85rem; font-weight:600; color:#047857;
  background:#ECFDF5; border:1px solid #A7F3D0; padding:.3rem .8rem; border-radius:999px; }
.status i { width:8px; height:8px; border-radius:50%; background:#10B981; display:inline-block; }

.hero { border-radius:24px; padding:2.6rem 2.6rem 2.2rem 2.6rem; color:#fff; margin-bottom:1.6rem;
  background: radial-gradient(800px 340px at 92% -20%, rgba(255,255,255,.26), transparent 60%),
              radial-gradient(600px 300px at 0% 120%, rgba(251,191,36,.5), transparent 60%),
              linear-gradient(120deg, #EA580C 0%, #E11D48 52%, #A21CAF 100%); }
.hero h1 { color:#fff; font-size:2.7rem; font-weight:800; letter-spacing:-.035em; line-height:1.1;
  margin:0 0 .8rem 0; padding:0; max-width:760px; }
.hero p { color:#FFEDD5; font-size:1.08rem; line-height:1.55; max-width:620px; margin:0 0 1.8rem 0; }
.hero-stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:1rem; }
.hero-stat { background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.3);
  border-radius:16px; padding:.9rem 1.1rem; }
.hero-stat span { display:block; font-size:.82rem; color:#FFEDD5; font-weight:600; }
.hero-stat b { display:block; font-size:1.75rem; font-weight:800; letter-spacing:-.03em; margin-top:.15rem; }
.hero-note { margin-top:1.1rem; font-size:.82rem; color:#FFEDD5; }

[data-testid="stTabs"] button[role="tab"] { padding:.7rem 1.1rem; }
[data-testid="stTabs"] button[role="tab"] p { font-weight:700; font-size:.98rem; }
[data-testid="stTabs"] [role="tablist"] { gap:.4rem; }

[data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border:1px solid #E7E9F0 !important;
  border-radius:20px !important; box-shadow:0 1px 2px rgba(15,23,42,.04), 0 8px 24px rgba(15,23,42,.04); }
[data-testid="stSlider"] label p, .stSelectbox label p, .stMultiSelect label p { font-weight:700; color:#0F172A; font-size:.92rem; }
span[data-baseweb="tag"] { background:#FFE4E6 !important; color:#9F1239 !important; }
.stDownloadButton button { background:#0F172A; color:#fff; border:0; border-radius:12px; font-weight:700; padding:.55rem 1rem; }
.stDownloadButton button:hover { background:#E11D48; color:#fff; }

.panel-title { font-size:1.08rem; font-weight:800; color:#0F172A; letter-spacing:-.01em; margin:0 0 .15rem 0; }
.panel-sub { font-size:.88rem; color:#64748B; margin:0 0 1rem 0; }

.result { margin-top:1rem; padding:1.15rem 1.2rem; border-radius:16px; color:#fff;
  background:linear-gradient(135deg, var(--seg), color-mix(in srgb, var(--seg) 70%, #0B1220)); }
.result small { font-size:.82rem; font-weight:600; opacity:.85; }
.result h3 { color:#fff; font-size:1.6rem; font-weight:800; letter-spacing:-.025em; margin:.15rem 0 0 0; padding:0; line-height:1.15; }
.result .seg { font-size:.9rem; opacity:.9; margin:.2rem 0 .9rem 0; }
.result .row { display:flex; justify-content:space-between; font-size:.88rem; padding:.4rem 0; border-top:1px solid rgba(255,255,255,.22); }
.result .act { margin-top:.7rem; font-size:.92rem; line-height:1.5; background:rgba(255,255,255,.16); padding:.7rem .85rem; border-radius:12px; }

.segs { display:grid; gap:1rem; grid-template-columns:repeat(5,1fr); }
@media (max-width:1100px) { .segs { grid-template-columns:repeat(3,1fr); } }
@media (max-width:720px)  { .segs { grid-template-columns:repeat(1,1fr); } .hero h1 { font-size:2rem; } .hero { padding:1.6rem; } }
.seg-card { background:#fff; border:1px solid #E7E9F0; border-radius:18px; padding:1.1rem 1.15rem 1.2rem 1.15rem;
  position:relative; overflow:hidden; box-shadow:0 1px 2px rgba(15,23,42,.04); }
.seg-card:before { content:""; position:absolute; inset:0 0 auto 0; height:5px; background:var(--seg); }
.seg-card h4 { font-size:1.08rem; font-weight:800; color:#0F172A; margin:.35rem 0 0 0; padding:0; letter-spacing:-.01em; }
.seg-card .orig { font-size:.8rem; color:#64748B; min-height:2.3em; margin-top:.15rem; }
.seg-card .share { font-size:2.2rem; font-weight:800; color:var(--seg); letter-spacing:-.03em; line-height:1.1; margin-top:.4rem; }
.seg-card .cnt { font-size:.84rem; color:#64748B; margin-bottom:.7rem; }
.seg-card .r { display:flex; justify-content:space-between; font-size:.87rem; color:#475569; padding:.4rem 0; border-top:1px solid #EEF0F5; }
.seg-card .r b { color:#0F172A; }
.seg-card .a { font-size:.86rem; color:#334155; line-height:1.5; margin-top:.6rem; }

table.play { width:100%; border-collapse:collapse; font-size:.92rem; }
table.play th { text-align:left; color:#64748B; font-weight:700; font-size:.82rem; padding:.7rem .8rem; border-bottom:1px solid #E7E9F0; }
table.play td { padding:.95rem .8rem; border-bottom:1px solid #F0F1F6; color:#334155; vertical-align:top; line-height:1.45; }
table.play tr:last-child td { border-bottom:0; }
.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:.5rem; background:var(--seg); }
.pname { font-weight:800; color:#0F172A; white-space:nowrap; }
.note { margin-top:1.4rem; padding:1rem 1.2rem; border-radius:14px; background:#FFFBEB; border:1px solid #FDE68A;
  color:#78350F; font-size:.9rem; line-height:1.55; }
.foot { margin-top:2rem; text-align:center; color:#94A3B8; font-size:.82rem; }
</style>
""",
    unsafe_allow_html=True,
)


def block(markup: str) -> str:
    """Flatten markup so Streamlit never treats indented HTML as a code block."""
    return " ".join(line.strip() for line in markup.splitlines())


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

pipeline, features, cluster_names = (
    artifact["pipeline"], artifact["features"], artifact["cluster_names"],
)

if not DATA_PATH.exists():
    st.error("Dataset not found. Place Mall_Customers.csv inside the project's data folder.")
    st.stop()
try:
    data = pd.read_csv(DATA_PATH)
except Exception as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()

required = ["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
missing = [c for c in required if c not in data.columns]
if missing:
    st.error(f"Dataset is missing columns: {missing}")
    st.stop()

try:
    data["Cluster"] = pipeline.predict(data[features].copy()).astype(int)
    data["Segment"] = data["Cluster"].map(cluster_names)
except Exception as exc:
    st.error(f"Could not assign customer segments: {exc}")
    st.stop()

# -------------------- Derived values --------------------

fallback_iter = iter(FALLBACK)


def meta(name):
    if name in SEGMENTS:
        return SEGMENTS[name]
    return (name, next(fallback_iter, MUTED), "Review", "Validate with campaign data.", "Conversion rate")


META = {n: meta(n) for n in data["Segment"].unique()}
data["Persona"] = data["Segment"].map(lambda n: META[n][0])
COLORS = {META[n][0]: META[n][1] for n in META}

total = len(data)
profiles = (
    data.groupby(["Cluster", "Segment", "Persona"], as_index=False)
    .agg(
        Customers=("CustomerID", "count"),
        Income=("Annual Income (k$)", "mean"),
        Spending=("Spending Score (1-100)", "mean"),
        Age=("Age", "mean"),
    )
    .sort_values("Cluster")
)
profiles["Share"] = profiles["Customers"] / total * 100
order = profiles["Persona"].tolist()

avg_income = data["Annual Income (k$)"].mean()
avg_spend = data["Spending Score (1-100)"].mean()

# -------------------- Top bar + hero --------------------

st.markdown(
    block(
        f"""
<div class="topbar">
  <div class="brand">
    <svg width="30" height="30" viewBox="0 0 30 30"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#F97316"/><stop offset="1" stop-color="#E11D48"/></linearGradient></defs>
    <rect width="30" height="30" rx="9" fill="url(#g)"/><circle cx="10" cy="11" r="3" fill="#fff"/>
    <circle cx="20" cy="10" r="2.2" fill="#fff" fill-opacity=".8"/><circle cx="15" cy="20" r="3.4" fill="#fff" fill-opacity=".9"/></svg>
    Customer Segmentation AI
  </div>
  <div class="status"><i></i>Model ready</div>
</div>
<div class="hero">
  <h1>Know which customers to reward, win back, or keep warm.</h1>
  <p>Place any customer on the segment map and get the marketing play that fits them.</p>
  <div class="hero-stats">
    <div class="hero-stat"><span>Customers analysed</span><b>{total:,}</b></div>
    <div class="hero-stat"><span>Segments found</span><b>{data['Cluster'].nunique()}</b></div>
    <div class="hero-stat"><span>Average annual income</span><b>${avg_income:.1f}k</b></div>
    <div class="hero-stat"><span>Average spending score</span><b>{avg_spend:.1f}</b></div>
  </div>
  <div class="hero-note">K-Means model, silhouette score {artifact['silhouette_score']:.2f}</div>
</div>
"""
    ),
    unsafe_allow_html=True,
)


def style_fig(fig, height):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        font=dict(family="Manrope, sans-serif", color=INK, size=13),
        legend=dict(title_text="", orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0),
    )
    return fig


tab_predict, tab_segments, tab_customers, tab_playbook = st.tabs(
    ["Predictor", "Segments", "Customers", "Playbook"]
)

# -------------------- Predictor --------------------

with tab_predict:
    left, right = st.columns([1, 1.75], gap="large")

    with left:
        with st.container(border=True):
            st.markdown(
                '<div class="panel-title">Customer profile</div>'
                '<div class="panel-sub">Move the sliders to describe a customer.</div>',
                unsafe_allow_html=True,
            )
            income = st.slider("Annual income (k$)", 15, 137, 60)
            score = st.slider("Spending score (1-100)", 1, 99, 50)

            cust = pd.DataFrame([[income, score]], columns=features)
            cid = int(pipeline.predict(cust)[0])
            seg = cluster_names[cid]
            persona, color, goal, action, _ = META.get(seg, meta(seg))
            row = profiles[profiles["Cluster"] == cid].iloc[0]

            st.markdown(
                block(
                    f"""
<div class="result" style="--seg:{color}">
  <small>Predicted segment</small>
  <h3>{escape(persona)}</h3>
  <div class="seg">{escape(str(seg))}, {row['Share']:.1f}% of customers</div>
  <div class="row"><span>Segment avg. income</span><b>${row['Income']:.1f}k</b></div>
  <div class="row"><span>Segment avg. spending</span><b>{row['Spending']:.1f}</b></div>
  <div class="act"><b>Suggested approach:</b> {escape(action)}</div>
</div>
"""
                ),
                unsafe_allow_html=True,
            )

    with right:
        with st.container(border=True):
            st.markdown(
                '<div class="panel-title">Segment map</div>'
                '<div class="panel-sub">Each dot is a customer. The star is the profile you selected.</div>',
                unsafe_allow_html=True,
            )
            fig = go.Figure()
            for p in order:
                d = data[data["Persona"] == p]
                fig.add_trace(
                    go.Scatter(
                        x=d["Annual Income (k$)"], y=d["Spending Score (1-100)"],
                        mode="markers", name=p,
                        marker=dict(size=10, color=COLORS[p], opacity=0.85, line=dict(width=1, color="white")),
                        customdata=d[["CustomerID", "Gender", "Age"]],
                        hovertemplate=p + "<br>Customer %{customdata[0]}, %{customdata[1]}, age %{customdata[2]}"
                        "<br>Income %{x}k, spending %{y}<extra></extra>",
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=[income], y=[score], mode="markers", name="Selected profile",
                    marker=dict(symbol="star", size=22, color=INK, line=dict(width=2, color="white")),
                    hovertemplate="Selected profile<br>Income %{x}k, spending %{y}<extra></extra>",
                )
            )
            style_fig(fig, 500)
            fig.update_xaxes(title="Annual income (k$)", gridcolor="#EEF0F5", zeroline=False, linecolor=LINE)
            fig.update_yaxes(title="Spending score", gridcolor="#EEF0F5", zeroline=False, linecolor=LINE)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# -------------------- Segments --------------------

with tab_segments:
    cards = []
    for _, r in profiles.iterrows():
        _, col, _, act, _ = META[r["Segment"]]
        cards.append(
            f"""
<div class="seg-card" style="--seg:{col}">
  <h4>{escape(r['Persona'])}</h4>
  <div class="orig">{escape(str(r['Segment']))}</div>
  <div class="share">{r['Share']:.1f}%</div>
  <div class="cnt">{int(r['Customers'])} customers</div>
  <div class="r"><span>Avg. income</span><b>${r['Income']:.1f}k</b></div>
  <div class="r"><span>Avg. spending</span><b>{r['Spending']:.1f}</b></div>
  <div class="r"><span>Avg. age</span><b>{r['Age']:.0f}</b></div>
  <div class="a">{escape(act)}</div>
</div>"""
        )
    st.markdown(block(f'<div class="segs">{"".join(cards)}</div>'), unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.container(border=True):
            st.markdown(
                '<div class="panel-title">Customer share</div>'
                '<div class="panel-sub">How the customer base splits across segments.</div>',
                unsafe_allow_html=True,
            )
            donut = go.Figure(
                go.Pie(
                    labels=profiles["Persona"], values=profiles["Customers"], hole=0.64, sort=False,
                    marker=dict(colors=[COLORS[p] for p in profiles["Persona"]], line=dict(color="white", width=3)),
                    textinfo="percent", textfont=dict(color="white", size=13),
                )
            )
            donut.add_annotation(text=f"<b>{total}</b><br>customers", showarrow=False, font=dict(size=16, color=INK))
            style_fig(donut, 380)
            st.plotly_chart(donut, use_container_width=True, config={"displayModeBar": False})
    with c2:
        with st.container(border=True):
            st.markdown(
                '<div class="panel-title">Income and spending by segment</div>'
                '<div class="panel-sub">Average annual income (k$) next to average spending score.</div>',
                unsafe_allow_html=True,
            )
            bars = go.Figure()
            bars.add_trace(go.Bar(x=profiles["Persona"], y=profiles["Income"].round(1), name="Avg. income (k$)", marker_color="#EA580C"))
            bars.add_trace(go.Bar(x=profiles["Persona"], y=profiles["Spending"].round(1), name="Avg. spending score", marker_color="#DB2777"))
            style_fig(bars, 380)
            bars.update_layout(barmode="group", bargap=0.3)
            bars.update_yaxes(gridcolor="#EEF0F5", zeroline=False)
            st.plotly_chart(bars, use_container_width=True, config={"displayModeBar": False})

# -------------------- Customers --------------------

with tab_customers:
    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([2, 2, 2, 1.2], vertical_alignment="bottom")
        with f1:
            pick = st.multiselect("Segments", order, default=[], placeholder="All segments")
        with f2:
            genders = st.multiselect("Gender", sorted(data["Gender"].unique()), default=[], placeholder="All")
        with f3:
            lo, hi = int(data["Age"].min()), int(data["Age"].max())
            ages = st.slider("Age range", lo, hi, (lo, hi))

        view = data.copy()
        if pick:
            view = view[view["Persona"].isin(pick)]
        if genders:
            view = view[view["Gender"].isin(genders)]
        view = view[view["Age"].between(*ages)]

        with f4:
            st.download_button(
                "Download CSV",
                data=view.to_csv(index=False).encode("utf-8"),
                file_name="customer_segments.csv",
                mime="text/csv",
                use_container_width=True,
            )
        st.markdown(
            f'<div class="panel-sub" style="margin:.8rem 0 .6rem 0">Showing {len(view)} of {total} customers</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            view[["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)", "Persona", "Segment"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Annual Income (k$)": st.column_config.NumberColumn("Income", format="$%dk"),
                "Spending Score (1-100)": st.column_config.ProgressColumn(
                    "Spending score", min_value=0, max_value=100, format="%d"
                ),
            },
        )

# -------------------- Playbook --------------------

with tab_playbook:
    with st.container(border=True):
        st.markdown(
            '<div class="panel-title">Marketing playbook</div>'
            '<div class="panel-sub">A starting point for each segment and the metric to watch.</div>',
            unsafe_allow_html=True,
        )
        rows = ""
        for _, r in profiles.iterrows():
            persona, col, goal, act, kpi = META[r["Segment"]]
            rows += (
                f'<tr style="--seg:{col}"><td class="pname"><span class="dot"></span>{escape(persona)}</td>'
                f'<td>{int(r["Customers"])} ({r["Share"]:.1f}%)</td><td>{escape(goal)}</td>'
                f'<td>{escape(act)}</td><td>{escape(kpi)}</td></tr>'
            )
        st.markdown(
            block(
                f'<table class="play"><tr><th>Segment</th><th>Customers</th><th>Goal</th>'
                f"<th>Suggested approach</th><th>Metric to track</th></tr>{rows}</table>"
            ),
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="note">These are suggested marketing hypotheses, not proven outcomes. '
        "Validate them with your own campaign and sales data before acting on them.</div>",
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="foot">Customer Segmentation AI. K-Means with five clusters, trained on annual income and spending score.</div>',
    unsafe_allow_html=True,
)
