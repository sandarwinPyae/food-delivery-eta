import streamlit as st
import pandas as pd
import joblib
import os
import gdown

from utils.helper import calculate_distance, prepare_features

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="ETA Predictor",
    page_icon="🍕",
    layout="centered"
)

# -----------------------------
# Simple Header
# -----------------------------
st.markdown("<h2 style='text-align:center;'>🍕 ETA Predictor</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Delivery time estimation</p>", unsafe_allow_html=True)
st.write("")

# -----------------------------
# Load Model (same as yours)
# -----------------------------
file_id = "1mHXp-yZx-970tf7CsVyTI_c9SJ34Tbym"
model_path = "model/multi_spectral_stack_model.pkl"
features_path = "model/features.pkl"

@st.cache_resource
def load_assets():
    os.makedirs("model", exist_ok=True)

    if not os.path.exists(model_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=True)

    if os.path.exists(model_path) and os.path.exists(features_path):
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        return model, features

    return None, None

model, feature_list = load_assets()

if model is None:
    st.error("Model files not found.")
    st.stop()

# -----------------------------
# Minimal Form Card
# -----------------------------
with st.form("eta_form"):

    st.subheader("Basic Info")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 60, 30)
        rating = st.slider("Rating", 1.0, 5.0, 4.5, 0.1)
        traffic = st.selectbox("Traffic", ["Low", "Medium", "High", "Jam"])

    with col2:
        weather = st.selectbox("Weather", ["Sunny", "Stormy", "Sandstorms", "Cloudy", "Fog", "Windy"])
        vehicle = st.selectbox("Vehicle", [0, 1, 2])
        deliveries = st.selectbox("Deliveries", [0, 1, 2, 3])

    st.write("")
    festival = st.checkbox("Festival")
    city = st.selectbox("City", ["Urban", "Semi-Urban", "Metropolitian"])

    st.write("")
    st.subheader("Location")

    res_lat = st.number_input("Restaurant Latitude", value=12.976, format="%.6f")
    res_lon = st.number_input("Restaurant Longitude", value=80.2219, format="%.6f")
    del_lat = st.number_input("Delivery Latitude", value=13.006, format="%.6f")
    del_lon = st.number_input("Delivery Longitude", value=80.2519, format="%.6f")

    order_time = st.time_input("Order Time")

    st.write("")
    submit = st.form_submit_button("Predict", use_container_width=True)

# -----------------------------
# Prediction
# -----------------------------
if submit:

    data = {
        'Delivery_person_Age': age,
        'Delivery_person_Ratings': rating,
        'Restaurant_latitude': res_lat,
        'Restaurant_longitude': res_lon,
        'Delivery_location_latitude': del_lat,
        'Delivery_location_longitude': del_lon,
        'Road_traffic_density': traffic,
        'Weatherconditions': weather,
        'Vehicle_condition': vehicle,
        'multiple_deliveries': deliveries,
        'Festival': 1 if festival else 0,
        'Time_Orderd': str(order_time),
        'City': city
    }

    df = pd.DataFrame([data])

    df['distance_km'] = calculate_distance(res_lat, res_lon, del_lat, del_lon)

    traffic_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Jam': 3}
    weather_map = {'Sunny': 0, 'Stormy': 1, 'Sandstorms': 2, 'Cloudy': 3, 'Fog': 4, 'Windy': 5}
    city_map = {'Urban': 0, 'Semi-Urban': 1, 'Metropolitian': 2}

    df['Road_traffic_density'] = df['Road_traffic_density'].map(traffic_map)
    df['Weatherconditions'] = df['Weatherconditions'].map(weather_map)
    df['City'] = df['City'].map(city_map)

    temp_time = pd.to_timedelta(df['Time_Orderd'])
    df['Order_Hour'] = temp_time.dt.components['hours']

    df_prepared = prepare_features(df)
    df_final = df_prepared.reindex(columns=feature_list, fill_value=0)

    prediction = model.predict(df_final)

    # Minimal Result Display
    st.write("")
    st.success(f"Estimated Time: {prediction[0]:.1f} minutes")
    st.caption(f"Distance: {df['distance_km'].values[0]:.2f} km")