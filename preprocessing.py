import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df, target_col):
    df = df.dropna()
    X = df.drop(columns=[target_col])
    y = df[target_col]

    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y.astype(str))

    return train_test_split(X, y, test_size=0.2, random_state=42)
