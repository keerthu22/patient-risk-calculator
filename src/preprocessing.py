#src/preprocessing.py

import pandas as pd

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()  # Just in case
    df.dropna(subset=['MemberID'], inplace=True)  # ✅ Only drop if MemberID is missing
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Gender'] = df['Gender'].fillna('Unknown')
    return df
