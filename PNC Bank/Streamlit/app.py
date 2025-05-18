import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="NLP Anomaly Detection Dashboard",
)
from tabs import st_cluster_combo #, sentiment_analysis
st.title("Interactive Multi-Module Dashboard")

# Description of the dashboard
st.markdown("""
### 🔍 Overview
This Streamlit dashboard integrates multiple data science modules designed for deep insights:
- **Sentiment Analysis**: Anomaly detection via unsupervised + rule-based NLP modeling.
- **Clustering with BERT**: Cluster user queries via sentence embeddings.


Each module is designed for interactivity and real-time scoring. Built to scale.
""")

# Sidebar tab selector
tab_options = {
   # "Sentiment Analysis": sentiment_analysis.render,
    "Cluster Bert": st_cluster_combo.render,
}
selected_tab = st.sidebar.radio("📂 Choose a dashboard tab", list(tab_options.keys()))

# Run selected tab
tab_options[selected_tab]()
