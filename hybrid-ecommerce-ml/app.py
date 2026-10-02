# app.py
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Unsupervised+Supervised Prediction Model",
    page_icon="🛍️",
    layout="wide"
)

@st.cache_resource
def load_pipeline_artifact():
    pipeline_path = "models/ecommerce_pipeline.pkl"
    if not os.path.exists(pipeline_path):
        st.error("Serialized pipeline artifact 'models/ecommerce_pipeline.pkl' not found. Please execute 'python train.py' first.")
        st.stop()
    return joblib.load(pipeline_path)

pipeline = load_pipeline_artifact()

st.title("🛍️ Unsupervised+Supervised Prediction Model")
st.caption("E-Commerce Session Purchase Conversion Prediction Engine")

st.markdown("---")
st.subheader("Interactive Session Parameter Input")
st.caption("Adjust real-time session behavioral signals to evaluate conversion probability.")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Session Activity**")
    admin = st.number_input("Administrative Pages", 0, 30, 2)
    admin_dur = st.number_input("Admin Duration (s)", 0.0, 3000.0, 45.0)
    info = st.number_input("Informational Pages", 0, 20, 0)
    info_dur = st.number_input("Informational Duration (s)", 0.0, 2000.0, 0.0)
    prod = st.number_input("Product Related Pages", 0, 500, 25)
    prod_dur = st.number_input("Product Duration (s)", 0.0, 20000.0, 850.0)

with col2:
    st.markdown("**Behavioral & Temporal**")
    bounce = st.slider("Bounce Rate", 0.0, 0.2, 0.01, step=0.005)
    exit_rate = st.slider("Exit Rate", 0.0, 0.2, 0.03, step=0.005)
    page_val = st.number_input("Page Value Score", 0.0, 400.0, 22.5)
    special_day = st.slider("Special Day Proximity", 0.0, 1.0, 0.0, step=0.1)
    month = st.selectbox("Month", ['Jan', 'Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], index=8)

with col3:
    st.markdown("**User Environment**")
    operating_systems = st.selectbox("Operating System ID", [1, 2, 3, 4, 5, 6, 7, 8], index=1)
    browser = st.selectbox("Browser ID", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], index=1)
    region = st.selectbox("Region ID", [1, 2, 3, 4, 5, 6, 7, 8, 9], index=0)
    traffic_type = st.selectbox("Traffic Type ID", list(range(1, 21)), index=1)
    visitor_type = st.selectbox("Visitor Type", ["Returning_Visitor", "New_Visitor", "Other"], index=0)
    weekend = st.checkbox("Weekend Session", value=False)

if st.button("Predict Purchase Intent", type="primary", use_container_width=True):
    input_dict = {
        'Administrative': admin,
        'Administrative_Duration': admin_dur,
        'Informational': info,
        'Informational_Duration': info_dur,
        'ProductRelated': prod,
        'ProductRelated_Duration': prod_dur,
        'BounceRates': bounce,
        'ExitRates': exit_rate,
        'PageValues': page_val,
        'SpecialDay': special_day,
        'Month': month,
        'OperatingSystems': operating_systems,
        'Browser': browser,
        'Region': region,
        'TrafficType': traffic_type,
        'VisitorType': visitor_type,
        'Weekend': weekend
    }
    
    res = pipeline.predict_session(input_dict)
    
    st.divider()
    st.subheader("Inference & Feature Augmentation Output")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Purchase Probability", f"{res['purchase_probability']*100:.1f}%")
    m2.metric("Prediction", res['prediction_label'])
    m3.metric("Behavioral Cluster", f"{res['cluster_name']} (ID: {res['cluster_id']})")
    m4.metric("DBSCAN Anomaly", res['anomaly_status'])
    
    st.markdown("---")
    st.markdown("### 🏷️ Simulated Catalogue Mapping & Similarity Analysis")
    st.info("⚠️ **Academic Notice**: The product recommendation below is generated from an isolated simulated catalogue using TF-IDF cosine similarity over session behavioral profiles.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Similarity Score:** `{res['similarity_score']:.4f}`")
        st.write(f"**Matched Product Title:** {res['matched_product']['Title']}")
        st.write(f"**Matched Category:** {res['matched_product']['Category']}")
    with c2:
        st.write(f"**Base Price:** ${res['matched_product']['Base_Price']}")
        st.write(f"**Catalogue Item Description:** {res['matched_product']['Description']}")