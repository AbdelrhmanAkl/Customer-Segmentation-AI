
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# -------------------- Configuration --------------------

st.set_page_config(
    page_title="Customer Segmentation AI",
    page_icon="",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "customer_segmentation_model.joblib"
DATA_PATH = BASE_DIR / "data" / "Mall_Customers.csv"

st.title("Customer Segmentation AI")
st.caption(
    "Machine Learning dashboard for customer profiling "
    "and data-driven marketing decisions."
)

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

missing_columns = [
    column for column in required_columns
    if column not in data.columns
]

if missing_columns:
    st.error(f"Dataset is missing columns: {missing_columns}")
    st.stop()

# -------------------- Generate Cluster Assignments --------------------

try:
    X = data[features].copy()
    data["Cluster"] = pipeline.predict(X).astype(int)
    data["Segment"] = data["Cluster"].map(cluster_names)
except Exception as exc:
    st.error(f"Could not assign customer segments: {exc}")
    st.stop()

# -------------------- Sidebar --------------------

st.sidebar.header("Customer Profile")

income = st.sidebar.slider(
    "Annual Income (k$)",
    min_value=15,
    max_value=137,
    value=60,
)

spending_score = st.sidebar.slider(
    "Spending Score (1-100)",
    min_value=1,
    max_value=99,
    value=50,
)

# -------------------- Customer Prediction --------------------

st.subheader("Predict Customer Segment")

customer = pd.DataFrame(
    [[income, spending_score]],
    columns=features,
)

predicted_cluster = int(pipeline.predict(customer)[0])
predicted_segment = cluster_names[predicted_cluster]

col1, col2, col3 = st.columns(3)

col1.metric("Annual Income", f"${income}k")
col2.metric("Spending Score", spending_score)
col3.metric("Predicted Cluster", predicted_cluster)

st.success(f"Predicted Customer Segment: {predicted_segment}")

# -------------------- Dataset Overview --------------------

st.divider()
st.subheader("Dataset Overview")

total_customers = len(data)
number_of_clusters = data["Cluster"].nunique()
average_income = data["Annual Income (k$)"].mean()
average_spending = data["Spending Score (1-100)"].mean()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Customers", f"{total_customers:,}")
c2.metric("Clusters Represented", number_of_clusters)
c3.metric("Average Annual Income", f"${average_income:.1f}k")
c4.metric("Average Spending Score", f"{average_spending:.1f}")

# -------------------- Cluster Visualization --------------------

st.divider()
st.subheader("Customer Segments")

fig = px.scatter(
    data,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color="Segment",
    hover_data=["CustomerID", "Gender", "Age", "Cluster"],
    title="Customer Segments by Income and Spending",
    labels={
        "Annual Income (k$)": "Annual Income (thousands)",
        "Spending Score (1-100)": "Spending Score",
    },
)

fig.update_layout(
    legend_title_text="Customer Segment",
    height=560,
)

st.plotly_chart(fig, use_container_width=True)

# -------------------- Segment Profiles --------------------

st.subheader("Segment Profiles")

profiles = (
    data.groupby(["Cluster", "Segment"], as_index=False)
    .agg(
        Customers=("CustomerID", "count"),
        Average_Income=("Annual Income (k$)", "mean"),
        Average_Spending=("Spending Score (1-100)", "mean"),
    )
)

profiles["Customer Share (%)"] = (
    profiles["Customers"] / total_customers * 100
).round(1)

profiles = profiles.rename(
    columns={
        "Cluster": "Cluster ID",
        "Average_Income": "Average Income (k$)",
        "Average_Spending": "Average Spending Score",
    }
)

profiles["Average Income (k$)"] = (
    profiles["Average Income (k$)"].round(2)
)
profiles["Average Spending Score"] = (
    profiles["Average Spending Score"].round(2)
)

st.dataframe(
    profiles,
    use_container_width=True,
    hide_index=True,
)

# -------------------- Customer Data --------------------

with st.expander("Explore Customer Records"):
    selected_segment = st.selectbox(
        "Filter by Segment",
        ["All Segments"] + sorted(data["Segment"].unique().tolist()),
    )

    filtered_data = data.copy()

    if selected_segment != "All Segments":
        filtered_data = filtered_data[
            filtered_data["Segment"] == selected_segment
        ]

    st.dataframe(
        filtered_data,
        use_container_width=True,
        hide_index=True,
    )

# -------------------- Business Interpretation --------------------

st.divider()
st.subheader("Business Interpretation")

st.markdown(
    """
    - **High income, high spending:** Potential candidates for loyalty
      programs and premium offers.
    - **High income, low spending:** Consider personalized campaigns
      and investigate barriers to purchase.
    - **Low income, high spending:** Test affordable offers and
      relevant promotions.
    - **Low income, low spending:** Focus on value-oriented campaigns.
    - **Average income, average spending:** Consider broad engagement
      and retention campaigns.
    
    These are suggested marketing hypotheses, not proven outcomes.
    Validate them using actual campaign and sales data.
    """
)

st.caption(
    "Model: K-Means | Five clusters | "
    f"Training silhouette score: {artifact['silhouette_score']:.4f}"
)
