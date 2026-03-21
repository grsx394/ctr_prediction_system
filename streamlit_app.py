# streamlit_app.py

"""
Streamlit Demo for CTR Prediction System.

Run locally with:
    streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import json

# API endpoint - uses your deployed API
API_URL = "https://ctr-prediction-api.onrender.com"

# Page config
st.set_page_config(
    page_title="CTR Prediction",
    page_icon="🎯",
    layout="wide"
)

# Title and description
st.title("🎯 Click-Through Rate Prediction")
st.markdown("""
This demo predicts the probability that a user will click on an advertisement.
Adjust the features below and click **Predict** to see the result.
""")

st.divider()

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("📱 Device Features")

    device_type = st.selectbox(
        "Device Type",
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: {
            0: "Unknown",
            1: "Mobile",
            2: "Desktop",
            3: "Tablet",
            4: "Other"
        }[x],
        index=1
    )

    device_conn_type = st.selectbox(
        "Connection Type",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "Unknown",
            1: "WiFi",
            2: "Cellular",
            3: "Other"
        }[x],
        index=1
    )

    st.subheader("⏰ Time Features")

    hour = st.slider("Hour of Day", 0, 23, 14)
    day_of_week = st.selectbox(
        "Day of Week",
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
        index=2
    )

with col2:
    st.subheader("📍 Ad Features")

    banner_pos = st.selectbox(
        "Banner Position",
        options=[0, 1, 2, 3, 4, 5],
        format_func=lambda x: {
            0: "Position 0 (Top)",
            1: "Position 1",
            2: "Position 2",
            3: "Position 3",
            4: "Position 4",
            5: "Position 5 (Bottom)"
        }[x],
        index=0
    )

    st.subheader("🔢 Anonymous Features")

    C14 = st.number_input("C14 (Ad ID)", value=15706, step=1)
    C1 = st.number_input("C1", value=1005, step=1)
    C15 = st.number_input("C15", value=320, step=1)
    C16 = st.number_input("C16", value=50, step=1)
    C17 = st.number_input("C17", value=1722, step=1)
    C18 = st.number_input("C18", value=0, step=1)
    C19 = st.number_input("C19", value=35, step=1)
    C21 = st.number_input("C21", value=79, step=1)

st.divider()

# Predict button
if st.button("🔮 Predict CTR", type="primary", use_container_width=True):

    # Prepare request payload
    payload = {
        "C14": int(C14),
        "C1": int(C1),
        "C15": int(C15),
        "C16": int(C16),
        "C17": int(C17),
        "C18": int(C18),
        "C19": int(C19),
        "C21": int(C21),
        "device_type": device_type,
        "device_conn_type": device_conn_type,
        "banner_pos": banner_pos,
        "hour": hour,
        "day_of_week": day_of_week,
        "site_category": "unknown",
        "app_category": "unknown",
        "site_domain": "unknown",
        "app_domain": "unknown"
    }

    try:
        with st.spinner("Getting prediction..."):
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30
            )

        if response.status_code == 200:
            result = response.json()

            # Display results
            st.success("Prediction Complete!")

            # Create three columns for metrics
            m1, m2, m3 = st.columns(3)

            with m1:
                ctr = result["click_probability"]
                st.metric(
                    label="Click Probability",
                    value=f"{ctr:.2%}"
                )

            with m2:
                st.metric(
                    label="Model Version",
                    value=result["model_version"]
                )

            with m3:
                st.metric(
                    label="Latency",
                    value=f"{result['latency_ms']:.0f} ms"
                )

            # Visual indicator
            st.divider()
            st.subheader("Prediction Interpretation")

            if ctr < 0.05:
                st.warning("📉 Low CTR — This ad is unlikely to be clicked.")
            elif ctr < 0.15:
                st.info("📊 Moderate CTR — Average click probability.")
            else:
                st.success("📈 High CTR — This ad has good click potential!")

            # Show raw response
            with st.expander("View Raw API Response"):
                st.json(result)

        else:
            st.error(f"API Error: {response.status_code}")
            st.text(response.text)

    except requests.exceptions.Timeout:
        st.error("Request timed out. The API might be waking up (free tier). Try again in 30 seconds.")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Footer
st.divider()
st.markdown("""
---
**About this project:**
- Built with XGBoost achieving 0.72 AUC-ROC
- Deployed on Render with FastAPI
- [View GitHub Repository](https://github.com/grsx394/ctr_prediction_system)
""")