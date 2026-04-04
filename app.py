import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
import time

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="SmartEstate AI Predictor",
    page_icon="🏡",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #e0f7fa, #f8f9fa);
}

/* Container */
.block-container {
    padding: 2rem;
    background: rgba(255,255,255,0.9);
    border-radius: 20px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
}

/* Button */
div.stButton > button {
    background: linear-gradient(90deg, #007bff, #00c6ff);
    color: white;
    border-radius: 12px;
    height: 55px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

/* Header */
.header-container {
    background: linear-gradient(135deg, #007bff, #00c6ff, #6f42c1);
    padding: 35px;
    border-radius: 20px;
    text-align: center;
    color: white;
    margin-bottom: 25px;
}

/* Prediction Card */
.prediction-card {
    background: white;
    padding: 40px;
    border-radius: 20px;
    border-left: 8px solid #007bff;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# --- PATHS ---
model_path = "house_model.pkl"
history_path = "user_predictions_log.csv"

# --- SIDEBAR (NEW INTERACTIVE PANEL) ---
st.sidebar.title("📊 Property Insights")

st.sidebar.info("💡 Enter values to see smart insights")

# --- LOAD MODEL ---
if os.path.exists(model_path):
    model = joblib.load(model_path)

    # --- HEADER ---
    st.markdown("""
    <div class="header-container">
        <h1>🏡 SmartEstate AI Predictor</h1>
        <p>✨ Know Your Property’s True Worth in Seconds</p>
    </div>
    """, unsafe_allow_html=True)

    # --- INPUT ---
    col1, col2 = st.columns(2)

    with col1:
        sq_ft = st.slider("📐 Area (Sq Ft)", 100, 10000, 2500)
        rooms = st.slider("🏠 Rooms", 1, 15, 4)

    with col2:
        dist = st.slider("📍 Distance (km)", 0.0, 100.0, 5.0)
        age = st.slider("📅 Property Age", 0, 100, 10)

    predict_btn = st.button("🚀 Generate Valuation")

    # --- SIDEBAR LIVE INSIGHTS ---
    score = int((sq_ft/10000)*40 + (rooms/15)*30 + (1 - dist/100)*20 + (1 - age/100)*10)

    if score > 75:
        category = "🔥 Luxury Property"
    elif score > 50:
        category = "🏡 Premium Property"
    else:
        category = "💰 Budget Property"

    st.sidebar.metric("🏆 Property Score", f"{score}/100")
    st.sidebar.success(category)

    if dist > 20:
        st.sidebar.warning("⚠️ Far from city → price may drop")

    if age > 30:
        st.sidebar.warning("⚠️ Old property → lower value")

    if rooms > 6:
        st.sidebar.info("🏠 Spacious home increases value")

    # --- PREDICTION ---
    if predict_btn:
        with st.spinner("🤖 AI analyzing..."):
            time.sleep(1)

            input_data = pd.DataFrame([{
                'square_feet': sq_ft,
                'num_rooms': rooms,
                'age': age,
                'distance_to_city(km)': dist
            }])

            prediction = model.predict(input_data)[0]

        # --- RESULT ---
        st.markdown(f"""
        <div class="prediction-card">
            <h4>ESTIMATED VALUE</h4>
            <h1 style='color:#007bff;'>₹{prediction:,.0f}</h1>
            <p style='color:green;'>✔ AI Prediction Complete</p>
        </div>
        """, unsafe_allow_html=True)

        # --- METRICS ---
        c1, c2, c3 = st.columns(3)
        c1.metric("Area", f"{sq_ft} sqft")
        c2.metric("Rooms", rooms)
        c3.metric("Distance", f"{dist} km")

        st.balloons()

        # --- SAVE ---
        new_entry = {
            'Time': datetime.now().strftime("%H:%M:%S"),
            'Area': sq_ft,
            'Rooms': rooms,
            'Age': age,
            'Distance': dist,
            'Price': f"₹{prediction:,.0f}"
        }

        df = pd.DataFrame([new_entry])

        if not os.path.exists(history_path):
            df.to_csv(history_path, index=False)
        else:
            df.to_csv(history_path, mode='a', header=False, index=False)

    # --- HISTORY ---
    st.divider()
    st.subheader("📜 Recent Predictions")

    if os.path.exists(history_path):
        hist = pd.read_csv(history_path)
        st.dataframe(hist.tail(5).iloc[::-1], use_container_width=True)
    else:
        st.info("No predictions yet.")

else:
    st.error("Model not found")