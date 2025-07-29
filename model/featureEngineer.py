import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def featureEngineer(df):
    # # loading the preprocessed DS
    # df = pd.read_csv('data/preprocessed.csv')

    # Slicing the ICD Prefix
    df['ICD_Prefix'] = df['DiagnosisCode'].str[:3]

    # Creating a custom ICD map to disease name
    icd_prefix_map = {
        'E10': 'Type 1 Diabetes',
        'E11': 'Type 2 Diabetes',
        'E13': 'Other Diabetes',
        'I10': 'Hypertension',
        'I25': 'Heart Disease',
        'I50': 'Heart Failure',
        'J44': 'COPD',
        'J45': 'Asthma',
        'J18': 'Pneumonia',
        'N18': 'Kidney Disease',
        'C50': 'Breast Cancer',
        'C34': 'Lung Cancer',
        'C18': 'Colon Cancer',
        'F32': 'Depression',
        'F20': 'Schizophrenia',
        'G30': 'Alzheimer\'s Disease',
        'M54': 'Back Pain',
        'K21': 'Acid Reflux',
        'K50': 'Crohn\'s Disease',
        'R51': 'Headache',
        'R53': 'Fatigue',
        'Z00': 'General Check-up',
        'Z51': 'Palliative Care'
    }

    # Creating new col in DS
    df['DiseaseName'] = df['ICD_Prefix'].map(icd_prefix_map)

    # Map for Disease to criticality
    disease_criticality = {
        'Type 1 Diabetes': 4,
        'Type 2 Diabetes': 4,
        'Other Diabetes': 4,
        'Hypertension': 3,
        'Heart Disease': 5,
        'Heart Failure': 5,
        'COPD': 4,
        'Asthma': 3,
        'Pneumonia': 4,
        'Kidney Disease': 4,
        'Breast Cancer': 5,
        'Lung Cancer': 5,
        'Colon Cancer': 5,
        'Depression': 3,
        'Schizophrenia': 4,
        'Alzheimer\'s Disease': 4,
        'Back Pain': 2,
        'Acid Reflux': 2,
        'Crohn\'s Disease': 3,
        'Headache': 2,
        'Fatigue': 2,
        'General Check-up': 1,
        'Palliative Care': 5
    }

    # Creating col for Criticality
    df['DiseaseCriticality'] = df['DiseaseName'].map(disease_criticality)

    # temp df for calculating risk after removing null valued records
    df_risk = df.dropna(subset=['DiseaseName', 'DiseaseCriticality'])

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

    # Risk Score Calculation dor each Member
    risk_features = df_risk.groupby('MemberID').agg(
        NumClaims=('ClaimID', 'count'),
        UniqueDiseases=('DiseaseName', 'nunique'),
        AvgCriticality=('DiseaseCriticality', 'mean'),
        MaxCriticality=('DiseaseCriticality', 'max'),
        ChronicCount=('DiseaseCriticality', lambda x: (x >= 4).sum())
    ).reset_index()

    risk_features['RiskScore'] = risk_features.apply(assign_risk_score, axis=1)

    # Preventive care map
    preventive_care_map = {
        'Type 1 Diabetes': "Regular insulin monitoring and diet control",
        'Type 2 Diabetes': "Weight management, avoid sugar, regular A1C tests",
        'Other Diabetes': "Monitor glucose, exercise, low-carb diet",
        'Hypertension': "Reduce salt, regular BP check, avoid stress",
        'Heart Disease': "Cardiologist visit, ECG, physical activity",
        'Heart Failure': "Low sodium diet, daily weight check, medications",
        'COPD': "Avoid smoking, pulmonary rehab, flu vaccination",
        'Asthma': "Use inhalers regularly, avoid triggers, annual checkups",
        'Pneumonia': "Vaccination, hygiene, timely antibiotics",
        'Kidney Disease': "Limit protein/sodium, regular blood/urine tests",
        'Breast Cancer': "Mammograms, regular check-ups, self-exams",
        'Lung Cancer': "Quit smoking, screening for early detection",
        'Colon Cancer': "Colonoscopy, high-fiber diet, routine screening",
        'Depression': "Therapy, medication, regular mental health checks",
        'Schizophrenia': "Psychiatric care, medication adherence",
        'Back Pain': "Stretching, physiotherapy, avoid heavy lifting",
        'Acid Reflux': "Avoid spicy food, small meals, raise head while sleeping",
        'Crohn\'s Disease': "Anti-inflammatory drugs, avoid trigger foods",
        'Alzheimer\'s Disease': "Memory exercises, caregiver support",
        'Headache': "Hydration, sleep, limit screen time",
        'Fatigue': "Balanced diet, sleep, manage stress",
        'General Check-up': "Routine health screening and lifestyle advice",
        'Palliative Care': "Comfort-focused care and emotional support"
    }

    # handling null valued records
    df_prevent = df_risk[['MemberID', 'DiseaseName']].dropna()
    df_prevent['Recommendation'] = df_prevent['DiseaseName'].map(preventive_care_map)

    # creating Recommendation Column
    recommendation_df = df_prevent.groupby('MemberID')['Recommendation'].unique().reset_index()
    recommendation_df['PreventiveCareAdvice'] = recommendation_df['Recommendation'].apply(lambda x: "; ".join(x))
    recommendation_df = recommendation_df.drop(columns=['Recommendation'])

    final_patient_profile = pd.merge(risk_features, recommendation_df, on='MemberID', how='left')

    # # Storing the Feature Engineered DS
    # balanced_df.to_csv("data/final_patient_profile.csv", index=False)

    claims_df = df
    profile_df = final_patient_profile

    # print("claims:", claims_df.head())
    # print("profile:",profile_df.head())

    claims_df = claims_df.drop_duplicates(subset=['ClaimID'])

    # Merge to bring RiskScore to each row
    df = claims_df.merge(profile_df[['MemberID', 'PreventiveCareAdvice', 'NumClaims', 'UniqueDiseases', 'AvgCriticality', 'MaxCriticality', 'ChronicCount', 'RiskScore']], on='MemberID', how='inner')

    df = df.drop_duplicates()

    df.reset_index(drop=True)

    # print("df:", df.head())

    # Drop missing data
    df.dropna(subset=['Age', 'Gender',  'NumClaims', 'UniqueDiseases', 'AvgCriticality', 'MaxCriticality', 'ChronicCount', 'DiagnosisCode', 'ProcedureCode', 'RiskScore'], inplace=True)

    # Normalize gender values
    df['Gender'] = df['Gender'].str.strip().str.upper()  # 'Male' → 'MALE'
    df['Gender'] = df['Gender'].replace({'MALE': 'M', 'FEMALE': 'F'})

    # Encode categorical columns
    le_gender = LabelEncoder()
    df['Gender'] = le_gender.fit_transform(df['Gender'])

    le_diag = LabelEncoder()
    df['DiagnosisCode'] = le_diag.fit_transform(df['DiagnosisCode'])

    le_proc = LabelEncoder()
    df['ProcedureCode'] = le_proc.fit_transform(df['ProcedureCode'])

    joblib.dump(le_gender, 'output/le_gender.pkl')
    joblib.dump(le_diag, 'output/le_diag.pkl')
    joblib.dump(le_proc, 'output/le_proc.pkl')

    print("Feature Engineering completed successfully!!!")

    return df