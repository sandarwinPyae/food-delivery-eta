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
# FIXED DIMENSION UI DESIGN (CSS)
# --------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #F8F9FB; }

    div[data-testid="stVerticalBlock"] > div:has(div.stSubheader) {
        background: white;
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        border: 1px solid #EDF0F5;
        min-height: 550px; 
    }

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 52px;
        background-color: #000000;
        color: white;
        font-weight: 600;
        border: none;
        transition: 0.3s ease;
    }
    .stButton > button:hover { background-color: #333; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD ASSETS
# --------------------------------------------------
MODEL_CONFIG = {
    "Random Forest": "15GVW6CWnYxUJOS07cTcj8mX7tW2Cm2dp",
    "LightGBM": "1YYhkY9IjmaClSHDREgxRUg7_Ib6bFUlr",
    "Multi-Stacking": "1VsZuT8OPc4NAtUcBTQCDUecq2UTn632Z"
}
FEATURES_FILE_ID = "1HWECz5MaglgZ7bzyMmmfodz8utgKzb8u"


@st.cache_resource
def load_assets():
    os.makedirs("model", exist_ok=True)
    models = {}
    for name, file_id in MODEL_CONFIG.items():
        path = os.path.join("model", f"{name.lower().replace(' ', '_')}.pkl")
        if not os.path.exists(path):
            gdown.download(id=file_id, output=path, quiet=True)
        try:
            obj = joblib.load(path)
            if isinstance(obj, list): obj = obj[0]
            if isinstance(obj, dict):
                for k in ["model", "best_estimator_", "regressor"]:
                    if k in obj: obj = obj[k]; break
            if hasattr(obj, "predict"): models[name] = obj
        except:
            pass

    f_path = "model/features.pkl"
    if not os.path.exists(f_path): gdown.download(id=FEATURES_FILE_ID, output=f_path, quiet=True)
    return models, joblib.load(f_path)


models, feature_list = load_assets()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    selected_model_name = st.selectbox("Prediction Model", options=list(models.keys()))
    st.divider()

# --------------------------------------------------
# MAIN UI
# --------------------------------------------------
st.title("Precise ETA.")
st.markdown("<p style='color:#86868B; margin-top:-15px;'>Smart logistics, powered by neural intelligence.</p>",
            unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    with st.container():
        st.subheader("📍 Journey & Order")
        c1, c2 = st.columns(2)
        with c1:
            res_lat = st.number_input("Restaurant Lat", value=12.976, format="%.4f")
            res_lon = st.number_input("Restaurant Lon", value=80.2219, format="%.4f")
        with c2:
            del_lat = st.number_input("Delivery Lat", value=13.006, format="%.4f")
            del_lon = st.number_input("Delivery Lon", value=80.2519, format="%.4f")

        food_type = st.selectbox("Type of Food", ["Buffet", "Drinks", "Meal", "Snack"])
        st.write("---")
        st.subheader("☁️ Environment")
        e1, e2 = st.columns(2)
        with e1: traffic = st.selectbox("Traffic Density", ["Low", "Medium", "High", "Jam"])
        with e2: weather = st.selectbox("Weather", ["Sunny", "Stormy", "Sandstorms", "Cloudy", "Fog", "Windy"])
        city = st.selectbox("City Zone", ["Urban", "Semi-Urban", "Metropolitian"])

with col_right:
    with st.container():
        st.subheader("🛵 Personnel & Timing")
        age = st.number_input("Courier Age", 18, 60, 25)
        rating = st.slider("Rating Score", 1.0, 5.0, 4.8)
        vehicle_condition = st.select_slider("Vehicle Condition", options=[0, 1, 2], value=1)
        vehicle_type = st.selectbox("Type of Vehicle", ["Bicycle", "Electric Scooter", "Scooter", "Motorcycle"])
        st.write("---")
        st.write("**⏰ Order Timestamp**")
        t1, t2 = st.columns(2)
        with t1: order_hour = st.number_input("Hour (0-23)", 0, 23, datetime.now().hour)
        with t2: order_min = st.number_input("Minute (0-59)", 0, 59, datetime.now().minute)
        festival = st.toggle("Festival Impact Mode")

# --------------------------------------------------
# PREDICTION LOGIC WITH GEOGRAPHICAL VALIDATION
# --------------------------------------------------
st.write("")
if st.button("Generate Intelligence Report"):

    # 1. Check for Negative Values
    coords = [res_lat, res_lon, del_lat, del_lon]
    has_negative = any(c < 0 for c in coords)

    # 2. Check Latitude range (6 to 37) and Longitude range (68 to 97)
    valid_lat = all(6 <= l <= 37 for l in [res_lat, del_lat])
    valid_lon = all(68 <= l <= 97 for l in [res_lon, del_lon])

    if has_negative:
        st.error("🚨 **Negative Values Not Allowed:** Coordinates cannot be negative numbers.")
    elif not valid_lat:
        st.error("🚨 **Invalid Latitude:** Values must be between **6.0 and 37.0**.")
    elif not valid_lon:
        st.error("🚨 **Invalid Longitude:** Values must be between **68.0 and 97.0**.")
    else:
        with st.spinner("Processing Logistics Data..."):
            # ... (Rest of your data dictionary and prediction logic remains the same)
            data = {
                "Delivery_person_Age": age,
                "Delivery_person_Ratings": rating,
                "Restaurant_latitude": res_lat,
                "Restaurant_longitude": res_lon,
                "Delivery_location_latitude": del_lat,
                "Delivery_location_longitude": del_lon,
                "Vehicle_condition": vehicle_condition,
                "multiple_deliveries": 1,
                "Order_Hour": order_hour,
                "Road_traffic_density": traffic,
                "Weatherconditions": weather,
                "Festival": "Yes" if festival else "No",
                "City": city
            }

            df = pd.DataFrame([data])
            dist = calculate_distance(res_lat, res_lon, del_lat, del_lon)
            df["distance_km"] = dist

            df_final = pd.get_dummies(df).reindex(columns=feature_list, fill_value=0)

            selected_model = models[selected_model_name]
            prediction = selected_model.predict(df_final)[0]

            # Results Display
            st.divider()
            res_c1, res_c2, res_c3 = st.columns(3)
            res_c1.metric("ESTIMATED ARRIVAL", f"{int(prediction)} MIN")
            res_c2.metric("TOTAL DISTANCE", f"{dist:.2f} KM")
            res_c3.metric("SELECTED ENGINE", selected_model_name)

            st.info(f"Summary: Delivering {food_type} via {vehicle_type}")