import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path)
    df = df.replace("?", np.nan)
    return df

def clean_data(df):

    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df