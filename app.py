import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import os

# Load saved models
rf_reg = joblib.load("models/rf_regressor.pkl")
rf_clf = joblib.load("models/rf_classifier.pkl")
scaler_h = joblib.load("models/scaler_housing.pkl")
kmeans = joblib.load("models/kmeans_clustering.pkl")  # Optional

# Try to load visualisation data safely
if os.path.exists("data/housing_scaled.csv"):
    housing_scaled = pd.read_csv("data/housing_scaled.csv")
else:
    housing_scaled = pd.DataFrame({
        'longitude': np.random.uniform(-124, -114, 100),
        'latitude': np.random.uniform(32, 42, 100),
        'median_income': np.random.uniform(1, 10, 100),
        'cluster': np.random.randint(0, 3, 100)
    })

if os.path.exists("data/housing_clean.csv"):
    housing_clean = pd.read_csv("data/housing_clean.csv")
else:
    housing_clean = pd.DataFrame({
        'median_income': np.random.uniform(1, 10, 100),
        'median_house_value': np.random.uniform(50000, 500000, 100),
        'ocean_proximity': np.random.choice(["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"], 100)
    })

# Page config
st.set_page_config(page_title="California Housing Explorer", layout="wide")
st.title("California Housing Price & Tier Prediction")

# Sidebar user input
st.sidebar.header("Input Features")
income = st.sidebar.slider("Median Income (×10,000)", 1.0, 10.0, 3.5)
latitude = st.sidebar.slider("Latitude", 32.5, 42.0, 36.0)
longitude = st.sidebar.slider("Longitude", -124.0, -114.0, -119.0)
ocean_prox = st.sidebar.selectbox("Ocean Proximity (Encoded)", [0, 1, 2, 3, 4], format_func=lambda x: ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"][x])

# Combine input
input_data = np.array([[longitude, latitude, income, ocean_prox]])
input_scaled = scaler_h.transform(input_data)

# Predict
pred_price_scaled = rf_reg.predict(input_scaled)[0]
pred_tier = rf_clf.predict(input_scaled)[0]
cluster = kmeans.predict(input_scaled)[0]

# Display predictions
st.subheader("Predicted Values")
pred_price = scaler_h.inverse_transform(
    np.hstack([np.zeros((1, 3)), np.array([[pred_price_scaled]])])
)[0][-1]
pred_price = max(0, pred_price)
st.metric("Predicted Median House Price", f"${pred_price:,.0f}")
st.metric("Price Tier", ["Low", "Mid", "High"][pred_tier])
st.metric("Market Segment (Cluster)", f"Cluster {cluster}")

# Explanation
with st.expander("Model Explanation"):
    st.markdown("""
    **How predictions are made:**
    - A Random Forest Regressor estimates house prices based on location, income, and proximity to the ocean.
    - A separate classifier predicts the price tier.
    - Clustering reveals market segments with similar patterns.

    **Most important features:**
    - Median Income
    - Latitude & Longitude (location)
    - Ocean proximity

    **Why visualisation and explanation matter:**
    - Visuals like maps, scatter plots, and heatmaps help uncover hidden patterns and relationships in the data.
    - 3D and interactive graphics help non-technical users understand complex trends.
    - Interpretation blocks guide users to understand why a prediction was made, not just what it is.
    - These explanations help build trust in the analytics process and support better decision-making.
    """)

# Interpretation Example
with st.expander("Example Interpretation of the Results"):
    st.markdown(f"""
    ### 📍 Input Summary
    - **Latitude:** 34 (Southern California)
    - **Longitude:** -121 (near coast)
    - **Median Income:** 7.5 ($75,000)
    - **Ocean Proximity:** <1H OCEAN (code 0)

    ### 📈 Model Prediction
    - **Predicted Median House Value:** ~${pred_price:,.0f}
    - **Predicted Tier:** {['Low', 'Mid', 'High'][pred_tier]}
    - **Assigned Cluster:** {cluster}

    ### 🔍 Interpretation
    - This region has high income and is close to the coast, which typically increases housing prices.
    - The model assigns the home to a high value range and places it among similar geographic/economic clusters.
    - This reflects how well the model incorporates both spatial and economic dimensions.

    ### 💡 What this tells us
    - The prediction aligns with domain knowledge: income and location are top price drivers.
    - Users can understand *why* the prediction makes sense — and trust the logic behind it.
    """)

# Visualize feature importance
with st.expander("Feature Importance (Classifier)"):
    importances = rf_clf.feature_importances_
    features = ["longitude", "latitude", "median_income", "ocean_proximity_cat"]
    sorted_idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(6,4))
    plt.title("Feature Importances")
    plt.bar(range(len(features)), importances[sorted_idx])
    plt.xticks(range(len(features)), np.array(features)[sorted_idx], rotation=45)
    st.pyplot(plt)

# Optional Visualizations
with st.expander("Visualizations"):
    viz_type = st.selectbox("Choose a visualization", ["Cluster Map", "Income vs Price Scatter", "Boxplot by Ocean Proximity", "Correlation Heatmap", "3D Cluster Scatter"])

    if viz_type == "Cluster Map":
        df = housing_scaled[['longitude', 'latitude', 'cluster']]
        fig = px.scatter_mapbox(
            df, lat='latitude', lon='longitude', color='cluster',
            mapbox_style="carto-positron", zoom=5, height=500
        )
        st.plotly_chart(fig)

    elif viz_type == "Income vs Price Scatter":
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(x='median_income', y='median_house_value', data=housing_clean, alpha=0.3, ax=ax)
        ax.set_title("Income vs. House Value")
        st.pyplot(fig)

    elif viz_type == "Boxplot by Ocean Proximity":
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='ocean_proximity', y='median_house_value', data=housing_clean, ax=ax)
        ax.set_title("House Value by Ocean Proximity")
        st.pyplot(fig)

    elif viz_type == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(housing_scaled.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Heatmap (Housing Features)")
        st.pyplot(fig)

    elif viz_type == "3D Cluster Scatter":
        df = housing_scaled[['longitude', 'latitude', 'median_income']].copy()
        df['cluster'] = housing_scaled['cluster'] if 'cluster' in housing_scaled else kmeans.labels_
        fig = px.scatter_3d(df, x='longitude', y='latitude', z='median_income', color='cluster',
                            title="3D Housing Market Clusters")
        st.plotly_chart(fig)

# Footer for feedback
st.markdown("""
---
### 💬 Feedback
Let us know what worked well or could be improved in this app. Your input helps us make it better!
""")
