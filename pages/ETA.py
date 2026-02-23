import streamlit as st
import pandas as pd
import joblib
import os
import gdown
from datetime import datetime
from utils.helper import calculate_distance

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Precise ETA", page_icon="📍", layout="wide")

# --------------------------------------------------
# MINIMALIST UI DESIGN
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #F4F6F9;
}

.block-container {
    padding-top: 2rem;
}

h1 {
    font-weight: 700;
    letter-spacing: -1px;
}

.section-card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

.stButton > button {
    background-color: black;
    color: white;
    border-radius: 12px;
    height: 48px;
    font-weight: 600;
    border: none;
    transition: 0.3s ease;
}

.stButton > button:hover {
    background-color: #333333;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# GOOGLE DRIVE MODEL IDS
# --------------------------------------------------
MODEL_CONFIG = {
    "Random Forest": "15GVW6CWnYxUJOS07cTcj8mX7tW2Cm2dp",
    "LightGBM": "1YYhkY9IjmaClSHDREgxRUg7_Ib6bFUlr",
    "Multi-Stacking": "1VsZuT8OPc4NAtUcBTQCDUecq2UTn632Z"
}
# https://drive.google.com/file/d/15GVW6CWnYxUJOS07cTcj8mX7tW2Cm2dp/view?usp=drive_link
FEATURES_FILE_ID = "1HWECz5MaglgZ7bzyMmmfodz8utgKzb8u"

# --------------------------------------------------
# SAFE GOOGLE DRIVE DOWNLOAD
# --------------------------------------------------
def download_from_drive(file_id, output_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    gdown.download(url, output_path, quiet=False)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
@st.cache_resource
def load_models():
    os.makedirs("model", exist_ok=True)
    models = {}

    for name, file_id in MODEL_CONFIG.items():
        filename = name.lower().replace(" ", "_") + ".pkl"
        path = os.path.join("model", filename)

        if not os.path.exists(path):
            download_from_drive(file_id, path)

        try:
            model_obj = joblib.load(path)

            # If accidentally downloaded HTML
            if isinstance(model_obj, str):
                st.error(f"{name} downloaded incorrectly (HTML file).")
                continue

            # If list
            if isinstance(model_obj, list) and len(model_obj) > 0:
                model_obj = model_obj[0]

            # If dict
            if isinstance(model_obj, dict):
                for key in ["model", "estimator", "best_estimator_", "classifier", "regressor"]:
                    if key in model_obj:
                        model_obj = model_obj[key]
                        break

            if hasattr(model_obj, "predict"):
                models[name] = model_obj
            else:
                st.warning(f"{name} loaded but has no predict() method.")

        except Exception as e:
            st.error(f"Error loading {name}: {e}")

    # Load feature list
    features_path = os.path.join("model", "features.pkl")

    if not os.path.exists(features_path):
        download_from_drive(FEATURES_FILE_ID, features_path)

    try:
        feature_list = joblib.load(features_path)
    except Exception as e:
        st.error(f"Failed to load features.pkl: {e}")
        feature_list = []

    if not models:
        st.error("❌ No valid models loaded. Check Google Drive files.")
        st.stop()

    return models, feature_list


models, feature_list = load_models()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Model Configuration")
    selected_model_name = st.selectbox(
        "Select Model",
        list(models.keys())
    )

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("<h1 style='margin-bottom:0;'>Precise ETA</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:gray; margin-top:0;'>Real-time delivery prediction powered by Machine Learning</p>", unsafe_allow_html=True)

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📍 Location Details")

    res_lat = st.number_input("Restaurant Latitude", value=12.976, format="%.4f")
    res_lon = st.number_input("Restaurant Longitude", value=80.2219, format="%.4f")
    del_lat = st.number_input("Delivery Latitude", value=13.006, format="%.4f")
    del_lon = st.number_input("Delivery Longitude", value=80.2519, format="%.4f")

    traffic = st.selectbox("Traffic Density", ["Low", "Medium", "High", "Jam"])
    weather = st.selectbox("Weather Condition", ["Sunny", "Stormy", "Sandstorms", "Cloudy", "Fog", "Windy"])
    city = st.selectbox("City Type", ["Urban", "Semi-Urban", "Metropolitian"])

with col2:
    st.markdown("### 🛵 Courier Details")

    age = st.number_input("Delivery Person Age", 18, 60, 25)
    rating = st.slider("Delivery Rating", 1.0, 5.0, 4.8)
    vehicle = st.selectbox("Vehicle Condition", [0, 1, 2])

    st.markdown("### ⏰ Order Time")

    time_col1, time_col2 = st.columns(2)

    with time_col1:
        order_hour = st.selectbox(
            "Hour",
            list(range(0, 24)),
            index=datetime.now().hour
        )

    with time_col2:
        order_minute = st.selectbox(
            "Minute",
            list(range(0, 60, 5)),
            index=datetime.now().minute // 5
        )

    festival = st.toggle("Festival Mode")

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if st.button("Generate Prediction"):

    data = {
        "Delivery_person_Age": age,
        "Delivery_person_Ratings": rating,
        "Restaurant_latitude": res_lat,
        "Restaurant_longitude": res_lon,
        "Delivery_location_latitude": del_lat,
        "Delivery_location_longitude": del_lon,
        "Vehicle_condition": vehicle,
        "multiple_deliveries": 1,
        "Order_Hour": order_hour,
        "Road_traffic_density": traffic,
        "Weatherconditions": weather,
        "Festival": "Yes" if festival else "No",
        "City": city
    }

    df = pd.DataFrame([data])
    df["distance_km"] = calculate_distance(res_lat, res_lon, del_lat, del_lon)

    df_ohe = pd.get_dummies(df)
    df_final = df_ohe.reindex(columns=feature_list, fill_value=0)

    model = models[selected_model_name]
    prediction = model.predict(df_final)[0]

    st.divider()
    c1, c2, c3 = st.columns(3)

    c1.metric("Expected Arrival", f"{int(prediction)} min")
    c2.metric("Travel Distance", f"{df['distance_km'].values[0]:.2f} km")
    c3.metric("Active Model", selected_model_name)