import preprocess
import featureEngineer
import model
import pandas as pd
from sklearn.utils import resample
import joblib

df = pd.read_csv('data/initial_datasets/patient_data.csv')

pre = preprocess.preprocess(df)

final = featureEngineer.featureEngineer(pre)[0]

print("shape of: ", final.shape)

# RULE-BASED METHOF FOR RISK SCORE CALCULATION
def assign_risk_score(row):
    # Add weighted scoring logic
    score = 0
        
    # Severity-based weighting
    score += row['MaxCriticality'] * 1.5
    score += row['AvgCriticality'] * 1.2
        
    # Volume of disease burden
    score += row['UniqueDiseases'] * 0.8
    score += row['ChronicCount'] * 2.0  # Higher weight for chronicity
    score += min(row['NumClaims'], 10) * 0.3  # Cap claims impact

    # Assign RiskScore based on ranges
    if score >= 20:
        return 5
    elif score >= 15:
        return 4
    elif score >= 10:
        return 3
    elif score >= 6:
        return 2
    else:
        return 1
        
final['RiskScore'] = final.apply(assign_risk_score, axis=1)

risk_counts = final['RiskScore'].value_counts().sort_index()
print("Risk Score Counts:\n", risk_counts)

# Balancing Classes
target_count = final.shape[0]//5

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
print("Risk Score Count after balancing:\n", risk_counts)


model.model(balanced_df)