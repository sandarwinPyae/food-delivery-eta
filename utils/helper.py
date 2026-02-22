import pandas as pd
import numpy as np


def calculate_distance(rest_lat1, rest_lon1, deli_lat2, deli_lon2):
    Earth_Radius = 6371  # Earth radius in km
    d_lat = np.radians(deli_lat2 - rest_lat1)
    d_lon = np.radians(deli_lon2 - rest_lon1)
    a = np.sin(d_lat / 2) ** 2 + np.cos(np.radians(rest_lat1)) * np.cos(np.radians(deli_lat2)) * np.sin(d_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return Earth_Radius * c


def prepare_features(data):
    df_subset = data.copy()
    for col in df_subset.columns:
        if pd.api.types.is_timedelta64_dtype(df_subset[col]):
            df_subset[col] = df_subset[col].dt.total_seconds() / 60.0

    df_subset = df_subset.select_dtypes(include=['number', 'bool']).copy()
    for col in df_subset.columns:
        if df_subset[col].dtype == 'bool':
            df_subset[col] = df_subset[col].astype(int)
    return df_subset