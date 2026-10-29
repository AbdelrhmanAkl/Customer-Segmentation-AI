# Customer Segmentation AI

**An End-to-End Machine Learning Project for Customer Behavior Analysis**

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Deployed-success)](https://customer-segmentation-ai1.streamlit.app/)

**Live Demo:** [Launch Customer Segmentation AI](https://customer-segmentation-ai1.streamlit.app/)

**GitHub Repository:** [AbdelrhmanAkl/customer-segmentation-ai](https://github.com/AbdelrhmanAkl/customer-segmentation-ai)

## Overview

Customer Segmentation AI is an end-to-end unsupervised machine learning project that groups customers according to their annual income and spending behavior.

The solution uses **K-Means clustering** with feature standardization to identify distinct customer segments. An interactive Streamlit dashboard enables users to explore customer distributions, inspect segment profiles, and generate predictions for new customer profiles.

The project demonstrates a practical machine learning workflow, from data preparation and clustering evaluation to model serialization, application development, and cloud deployment.

## Key Features

- **Customer Segmentation:** Groups customers into five clusters using K-Means.
- **Feature Standardization:** Applies `StandardScaler` through a reusable machine learning pipeline.
- **Model Evaluation:** Uses the silhouette score to assess clustering quality.
- **Interactive Predictions:** Assigns a customer profile to a learned cluster using annual income and spending score.
- **Data Visualization:** Explores customer distributions and cluster patterns through interactive charts.
- **Segment Profiling:** Displays segment characteristics to support customer behavior analysis.
- **Business Insights:** Connects clustering results with interpretable customer segment descriptions.
- **Web Deployment:** Provides an interactive interface through Streamlit Community Cloud.

## Machine Learning Approach

### Dataset

The project uses the Mall Customers dataset, containing **200 customer records**.

| Feature | Description |
|---|---|
| CustomerID | Customer identifier; not used as a clustering feature |
| Gender | Customer gender |
| Age | Customer age |
| Annual Income (k$) | Annual income in thousands of dollars |
| Spending Score (1-100) | Spending score on a scale from 1 to 100 |

The clustering model uses two features:

- Annual Income (k$)
- Spending Score (1-100)

### Workflow

1. Load and inspect the customer dataset.
2. Select the relevant numerical features.
3. Standardize the features using `StandardScaler`.
4. Train a K-Means clustering model with five clusters.
5. Evaluate clustering quality using the silhouette score.
6. Save the fitted pipeline and segment metadata using Joblib.
7. Integrate the trained model into a Streamlit application.
8. Deploy the application for interactive use.

### Model Configuration and Results

| Item | Result |
|---|---|
| Algorithm | K-Means |
| Number of clusters | 5 |
| Feature scaling | StandardScaler |
| Random state | 42 |
| Initialization | 10 runs (`n_init=10`) |
| Dataset size | 200 customers |
| Silhouette score | 0.5547 |

The silhouette score of **0.5547** indicates meaningful separation among the identified clusters for this dataset. It is an internal clustering metric, not a classification accuracy score.

## Customer Segments

The five clusters can be interpreted using their average annual income and spending scores.

| Segment | Customers | Avg. Annual Income (k$) | Avg. Spending Score |
|---|---:|---:|---:|
| Average Income - Average Spending | 81 | 55.30 | 49.52 |
| High Income - High Spending | 39 | 86.54 | 82.13 |
| Low Income - High Spending | 22 | 25.73 | 79.36 |
| High Income - Low Spending | 35 | 88.20 | 17.11 |
| Low Income - Low Spending | 23 | 26.30 | 20.91 |

These labels provide a business-oriented interpretation of the cluster profiles. They are descriptive categories derived from the observed averages, not predefined target labels.

### Potential Business Applications

- **High Income - High Spending:** Explore loyalty programs and premium customer experiences.
- **High Income - Low Spending:** Investigate barriers to spending and opportunities for engagement.
- **Low Income - High Spending:** Explore suitable promotions and value-oriented offers.
- **Low Income - Low Spending:** Consider budget-conscious campaigns and targeted engagement.
- **Average Income - Average Spending:** Develop retention strategies and monitor changes in spending behavior.

These are potential applications of the segmentation results, not measured business outcomes.

## Interactive Dashboard

The deployed application allows users to:

- Explore the dataset and customer segments.
- Visualize customer distributions.
- Review segment-level statistics.
- Adjust income and spending inputs to predict a customer segment.
- Interpret the predicted segment using its descriptive label.

**Try the application:** [Customer Segmentation AI — Live Demo](https://customer-segmentation-ai1.streamlit.app/)

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Feature scaling and K-Means clustering |
| Joblib | Model serialization and loading |
| Plotly | Interactive visualizations |
| Streamlit | Interactive web application |
| Git & GitHub | Version control and source code hosting |
| Streamlit Community Cloud | Application deployment |

## Project Structure

```text
customer-segmentation-ai/
├── app.py
├── requirements.txt
├── .gitignore
├── data/
│   └── Mall_Customers.csv
└── models/
    └── customer_segmentation_model.joblib
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/AbdelrhmanAkl/customer-segmentation-ai.git
cd customer-segmentation-ai
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Launch the application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL where you can interact with the dashboard.

## Model Inference

The saved model artifact contains the fitted preprocessing and clustering pipeline, feature names, cluster descriptions, cluster profiles, and evaluation metadata.

Example of loading the artifact and predicting a customer segment:

```python
import joblib
import pandas as pd

artifact = joblib.load(
    "models/customer_segmentation_model.joblib"
)

customer = pd.DataFrame(
    [[60, 50]],
    columns=artifact["features"]
)

cluster_id = int(
    artifact["pipeline"].predict(customer)[0]
)

segment_name = artifact["cluster_names"][cluster_id]

print("Cluster ID:", cluster_id)
print("Customer Segment:", segment_name)
```

The example uses an annual income of 60 (thousand dollars) and a spending score of 50.

## Limitations

- The results depend on the dataset and selected features.
- K-Means requires the number of clusters to be specified in advance.
- Cluster assignments describe similarities in the selected features; they do not establish customer intent or future purchasing behavior.
- Segment labels are interpretations of cluster profiles and may need adjustment for a different business context.
- The results should be validated against real business data before being used for operational decisions.

## Future Improvements

- Add additional customer attributes where appropriate.
- Compare clustering configurations using multiple evaluation metrics.
- Add more detailed segment-level visualizations.
- Introduce exportable customer segment reports.
- Evaluate segmentation stability on additional datasets.

## Author

**Abdelrahman Akl**

AI Engineer | Machine Learning | Data Science | Generative AI

- **GitHub:** [@AbdelrhmanAkl](https://github.com/AbdelrhmanAkl)
- **LinkedIn:** [Abdelrahman Akl](https://www.linkedin.com/in/abdelrahmanakl/)

---

*Built as a practical machine learning project demonstrating customer segmentation, model evaluation, interactive visualization, and deployment.*