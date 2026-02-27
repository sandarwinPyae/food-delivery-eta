import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import analysis

# ==============================
# 1️⃣ Page Configuration
# ==============================
st.set_page_config(page_title="Delivery Analytics", page_icon="📊", layout="wide")


# ==============================
# 2️⃣ Optimized Data Loading & Processing
# ==============================
@st.cache_data
def load_and_clean_data():
    try:
        df = analysis.load_data()
    except:
        df = pd.read_csv("train_cleaned_data.csv")

    if 'City' in df.columns:
        df['City'] = df['City'].astype(str).str.strip()

    # Move heavy time processing here so it only happens ONCE
    if 'Order_Date' in df.columns:
        df['Order_Date'] = pd.to_datetime(df['Order_Date'])
        df['Day_Name'] = df['Order_Date'].dt.day_name()

    if 'Order_Hour' in df.columns:
        df['Order_Hour'] = pd.to_numeric(df['Order_Hour'], errors='coerce').fillna(0).astype(int)

    return df


# Cache the Spearman calculation specifically
@st.cache_data
def calculate_spearman(df_subset):
    numeric_df = df_subset.select_dtypes(include=np.number)
    return numeric_df.corr(method='spearman')


df = load_and_clean_data()

# ==============================
# 3️⃣ Sidebar Filters
# ==============================
with st.sidebar:
    # st.title("⚙️ Settings")

    selected_city = st.multiselect(
        "Filter by City",
        options=sorted(df['City'].unique()),
        default=df['City'].unique()
    )

    st.divider()

    if st.checkbox("Show Raw Data Preview"):
        # Optimized with modern width parameter
        st.dataframe(df[df["City"].isin(selected_city)].head(10), width="stretch")

# Create filtered copy
filtered_df = df[df["City"].isin(selected_city)].copy()

# ==============================
# Custom Metric Card Styling
# ==============================
st.markdown("""
    <style>
    [data-testid="stMetric"] {
        border: 1px solid #e6e9ef;
        padding: 15px;
        border-radius: 10px;
        background-color: #f8f9fa;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# 4️⃣ Main Dashboard
# ==============================
st.title("📊 Food Delivery Insights")

# KPI Section
kpi_cols = st.columns(3)
kpi_cols[0].metric("Orders", f"{len(filtered_df):,}")
kpi_cols[1].metric("Avg Time Taken", f"{filtered_df['Time_taken(min)'].mean():.1f} m")
kpi_cols[2].metric("Avg Rating", f"{filtered_df['Delivery_person_Ratings'].mean():.2f} ⭐")

st.divider()

# ==============================
# 5️⃣ Tabs
# ==============================
tab_viz, tab_stats, tab_info = st.tabs(
    ["📈 Visual Analysis", "🔢 Statistics", "🧬 Data Schema"]
)

# ==============================
# 📈 VISUAL TAB (Optimized)
# ==============================
with tab_viz:
    st.subheader("📊 Feature Distribution")

    # Identify numeric columns
    numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()

    # Exclude technical columns for a cleaner dropdown
    display_cols = [c for c in numeric_cols if 'latitude' not in c.lower() and 'longitude' not in c.lower()]

    # Create a layout with a selector on the left and the chart filling the rest
    col_select, col_chart = st.columns([1, 3])

    with col_select:
        target = st.selectbox("Select Feature", display_cols,
                              index=display_cols.index('Time_taken(min)') if 'Time_taken(min)' in display_cols else 0)

        # Adding some summary stats specifically for this feature in the sidebar col
        st.info(f"""
            **Quick Stats:**
            - Mean: {filtered_df[target].mean():.2f}
            - Median: {filtered_df[target].median():.2f}
            - Max: {filtered_df[target].max():.2f}
            """)

    with col_chart:
        # Create a responsive Plotly Histogram with a Box plot marginal
        # Using a professional Blue (#3498db) instead of Green
        fig_dist = px.histogram(
            filtered_df,
            x=target,
            marginal="box",  # Shows outliers clearly at the top
            nbins=40,
            template="simple_white",
            color_discrete_sequence=['#3498db'],
            opacity=0.8
        )

        fig_dist.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=450,
            showlegend=False,
            xaxis_title=f"{target} Value",
            yaxis_title="Frequency"
        )

        st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()

    # Peak Hours & Weekly Trend
    col_h1, col_h2 = st.columns(2)

    with col_h1:
        st.subheader("🕒 Peak Delivery Hours")
        fig_hour, ax_hour = plt.subplots(figsize=(6, 4))
        sns.countplot(data=filtered_df, x='Order_Hour', hue='Order_Hour', palette="viridis", ax=ax_hour, legend=False)
        sns.despine()
        st.pyplot(fig_hour)

    with col_h2:
        st.subheader("📅 Weekly Order Trend")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_counts = filtered_df['Day_Name'].value_counts().reindex(day_order)

        fig_week, ax_week = plt.subplots(figsize=(6, 4))
        weekly_counts.plot(kind='line', marker='o', color='#2ecc71', ax=ax_week)
        ax_week.set_ylabel("Number of Orders")
        sns.despine()
        st.pyplot(fig_week)

    st.divider()

    st.subheader("📍 Top 10 High-Density Restaurant Hubs")
    try:
        top_10 = pd.read_csv("data/top_10_restaurant_hubs.csv")
        fig_hub, ax_hub = plt.subplots(figsize=(10, 6))

        # Create the Lollipop Chart
        ax_hub.hlines(y=top_10['City_Name'], xmin=0, xmax=top_10['Restaurant_Count'],
                      color='#D3D3D3', alpha=0.6, linewidth=1.5)
        ax_hub.scatter(top_10['Restaurant_Count'], top_10['City_Name'],
                       color='#E74C3C', s=100, zorder=3)

        # Adding the labels you requested
        ax_hub.set_xlabel("Number of Restaurants", fontsize=10, fontweight='bold', labelpad=10)
        ax_hub.set_ylabel("City Name", fontsize=10, fontweight='bold', labelpad=10)

        # Add text annotations for each point
        for _, row in top_10.iterrows():
            ax_hub.text(row['Restaurant_Count'] + 0.3, row['City_Name'],
                        f"{int(row['Restaurant_Count'])}", va='center',
                        fontsize=9, fontweight='bold')

        sns.despine(top=True, right=True, left=True)

        # Adjust layout to make sure labels aren't cut off
        plt.tight_layout()

        st.pyplot(fig_hub)
    except Exception as e:
        st.warning(f"Hub data file not found or error occurred: {e}")
# ==============================
# 🔢 STATISTICS TAB
# ==============================
with tab_stats:
    # Filter for useful numbers only
    exclude_cols = ['Order_Hour']
    final_stats_cols = [c for c in numeric_cols if c not in exclude_cols]

    st.subheader("📊 Descriptive Summary (4 Decimal Places)")
    # Applying the 4-decimal precision
    stats_df = filtered_df[final_stats_cols].describe().T.round(4)
    st.dataframe(stats_df, use_container_width=True)

    st.divider()

    st.subheader("📉 Outlier Analysis")
    feat = st.selectbox("Select Feature to Inspect", final_stats_cols)
    fig_box = px.box(filtered_df, y=feat, template="simple_white", color_discrete_sequence=['#3498db'])
    st.plotly_chart(fig_box, use_container_width=True)


with tab_info:
    st.subheader("🧬 Data Architecture")
    st.write("Overview of the features used to calculate Delivery ETA and performance metrics.")

    # Create two clean columns for high-level info
    meta1, meta2, meta3 = st.columns(3)
    meta1.metric("Total Features", f"{len(filtered_df.columns)}")
    meta2.metric("Categorical Cols", f"{len(filtered_df.select_dtypes(include=['object']).columns)}")
    meta3.metric("Numerical Cols", f"{len(filtered_df.select_dtypes(include=np.number).columns)}")

    st.divider()

    # --- Section 1: Feature Definitions ---
    st.markdown("### 📋 Feature Glossary")

    # We define the categories for the user
    schema_data = {
        "Category": [
            "Identity", "Identity", "Identity", "Identity",
            "Location", "Location", "Location", "Location",
            "Temporal", "Temporal", "Temporal", "Temporal", "Temporal",
            "Environmental", "Environmental", "Environmental",
            "Logistics", "Logistics", "Logistics", "Logistics",
            "Logistics", "Logistics",
            "Target", "Logistics"
        ],
        "Column Name": [
            "ID", "Delivery_person_ID", "Delivery_person_Age", "Delivery_person_Ratings",
            "Restaurant_latitude", "Restaurant_longitude", "Delivery_location_latitude", "Delivery_location_longitude",
            "Order_Date", "Time_Orderd", "Time_Order_picked", "Order_Hour", "Day_of_Week",
            "Weatherconditions", "Road_traffic_density", "Festival",
            "distance_km", "Type_of_order", "Type_of_vehicle", "Vehicle_condition",
            "multiple_deliveries", "City",
            "Time_taken(min)", "Prep_Time_Min"
        ],
        "Description": [
            "Unique Order ID", "Unique Courier ID", "Age of the delivery partner",
            "Average rating of the delivery partner",
            "GPS Latitude of the restaurant", "GPS Longitude of the restaurant", "GPS Latitude of the delivery point",
            "GPS Longitude of the delivery point",
            "Date the order was placed", "Exact timestamp of order placement", "Timestamp when courier picked up food",
            "Hour of day (0-23)", "Day name (Monday-Sunday)",
            "Weather state (Sunny, Stormy, etc.)", "Traffic level (Low, Medium, High, Jam)",
            "Binary indicator if it is a public festival",
            "Haversine distance between restaurant and delivery point", "Category of food (Snack, Meal, etc.)",
            "Mode of transport used", "Maintenance rank of vehicle (0-2)",
            "Number of other orders being delivered simultaneously", "Type of area (Urban, Metropolitian, Semi-Urban)",
            "Total duration from order to delivery (Primary Goal)",
            "Calculated time taken by the kitchen to prepare food"
        ]
    }
    st.table(pd.DataFrame(schema_data))

    st.divider()

    # --- Section 2: Technical Summary Table ---
    st.markdown("### ⚙️ Technical Structure")

    # Generate the technical summary dynamically
    tech_info = []
    for col in filtered_df.columns:
        tech_info.append({
            "Column": col,
            "Type": str(filtered_df[col].dtype).replace('object', 'String/Text').replace('float64', 'Decimal').replace(
                'int64', 'Integer'),
            "Completeness": f"{(1 - filtered_df[col].isnull().sum() / len(filtered_df)) * 100:.1f}%",
            "Unique Values": filtered_df[col].nunique()
        })

    st.dataframe(pd.DataFrame(tech_info), use_container_width=True, hide_index=True)