import preprocess
import featureEngineer
import model
import pandas as pd
from sklearn.utils import resample
import joblib

df = pd.read_csv('data/patient_data.csv')

pre = preprocess.preprocess(df)

final = featureEngineer.featureEngineer(pre)

# Balancing Classes
target_count = 100

# Store balanced data
balanced_df = pd.DataFrame()

# Process each class
for score in sorted(final['RiskScore'].unique()):
    subset = final[final['RiskScore'] == score]
    if len(subset) < target_count:
        upsampled = resample(subset, replace=True, n_samples=target_count, random_state=42)
        balanced_df = pd.concat([balanced_df, upsampled])
    else:
        downsampled = resample(subset, replace=False, n_samples=target_count, random_state=42)
        balanced_df = pd.concat([balanced_df, downsampled])

# Shuffle and save
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("balanced_df:", balanced_df.head())

risk_counts = balanced_df['RiskScore'].value_counts().sort_index()
print("RC:", risk_counts)

model.model(balanced_df)