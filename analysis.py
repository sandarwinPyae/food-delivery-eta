import pandas as pd
import numpy as np

def load_data():
    return pd.read_csv("data/train_cleaned_data.csv")


def get_basic_statistics(df):
    return df.describe()


def get_skewness(df):
    numeric_cols = df.select_dtypes(include=np.number)
    return numeric_cols.skew()


def get_correlation(df):
    numeric_cols = df.select_dtypes(include=np.number)
    return numeric_cols.corr()