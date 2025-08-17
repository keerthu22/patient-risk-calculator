import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def featureEngineer(df):
    # # loading the preprocessed DS
    # df = pd.read_csv('data/preprocessed.csv')

    # Slicing the ICD Prefix
    # df['ICD_Prefix'] = df['DiagnosisCode'].str[:3]

    # Load the map from CSV
    icd_map = pd.read_csv('data/final_datasets/icd_disease_criticality_preventivecare.csv')

    # Create a dictionary from ICD prefix to (disease, criticality, PreventiveCareAdvice)
    icd_dict = icd_map.set_index('ICD_Prefix')[['Criticality', 'PreventiveCareAdvice']].to_dict(orient='index')
    
    # Map function using only first 3 letters of DiagnosisCode
    def get_disease_info(code):
        info = icd_dict.get(code[:3], {'Criticality': 1, 'PreventiveCareAdvice':'Consult the Doctor'})
        return pd.Series([info['Criticality'], info['PreventiveCareAdvice']])
    
    # Apply the function to create two new columns
    df[['DiseaseCriticality', 'PreventiveCareAdvice']] = df['DiagnosisCode'].apply(get_disease_info)

    icd_df = pd.read_csv('data/final_datasets/final_ICD.csv')

    code_to_desc = icd_df.set_index('Code')['Desc'].to_dict()
    group_to_cat = icd_df.set_index('Group')['Category'].to_dict()

    def get_disease_name(code):
        code = code.replace(".", "")
        if code in code_to_desc:
            return code_to_desc[code]
        # 3-char group fallback
        prefix = code[:3]
        return group_to_cat.get(prefix, 'Unknown')

    df['DiseaseName'] = df['DiagnosisCode'].apply(get_disease_name)

    # Risk Score Calculation for each Member
    # Step 1: Find max criticality rows per MemberID
    critical_rows = df.loc[df.groupby('MemberID')['DiseaseCriticality'].idxmax()]

    # Step 2: Build risk features with extra fields
    risk_features = df.groupby('MemberID').agg(
        Age=('Age','first'),
        Gender=('Gender','first'),
        NumClaims=('ClaimID', 'count'),
        UniqueDiseases=('DiagnosisCode', 'nunique'),
        AvgCriticality=('DiseaseCriticality', 'mean'),
        MaxCriticality=('DiseaseCriticality', 'max'),
        ChronicCount=('DiseaseCriticality', lambda x: (x >= 4).sum()),
        TotalAmountBilled=('AmountBilled','sum'),
    ).reset_index()

    # Step 3: Merge with critical DiagnosisCode and ProcedureCode
    critical_codes = critical_rows[['MemberID', 'DiagnosisCode', 'ProcedureCode']]
    risk_features = risk_features.merge(critical_codes, on='MemberID', how='left')

    # risk_features['RiskScore'] = risk_features.apply(assign_risk_score, axis=1)

    # Normalize gender values
    df['Gender'] = df['Gender'].str.strip().str.upper()  # 'Male' → 'MALE'
    df['Gender'] = df['Gender'].replace({'MALE': 'M', 'FEMALE': 'F'})
    risk_features['Gender'] = risk_features['Gender'].str.strip().str.upper()  # 'Male' → 'MALE'
    risk_features['Gender'] = risk_features['Gender'].replace({'MALE': 'M', 'FEMALE': 'F'})

    le_gender = LabelEncoder()
    risk_features['Gender'] = le_gender.fit_transform(risk_features['Gender'])

    le_diag = LabelEncoder()
    risk_features['DiagnosisCode'] = le_diag.fit_transform(risk_features['DiagnosisCode'])

    le_proc = LabelEncoder()
    risk_features['ProcedureCode'] = le_proc.fit_transform(risk_features['ProcedureCode'])

    joblib.dump(le_gender, 'output/le_gender.pkl')
    joblib.dump(le_diag, 'output/le_diag.pkl')
    joblib.dump(le_proc, 'output/le_proc.pkl')

    risk_features.to_csv('data/temp/Summary.csv')
    df.to_csv('data/temp/Report.csv')

    print("Feature Engineering completed successfully!!!")

    return risk_features, df