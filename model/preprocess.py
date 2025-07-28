import pandas as pd

def preprocess(df):
    # # loading the synthetic dataset
    # df = pd.read_csv('data/patient_data.csv')

    # dropping rows where 'DiagnosisCode', 'MemberID' are null
    df = df.dropna(subset=['DiagnosisCode', 'MemberID'])

    # converting the following cols to str type
    df['DiagnosisCode'] = df['DiagnosisCode'].astype(str)
    df['ProcedureCode'] = df['ProcedureCode'].astype(str)

    # separating records if a single record has multiple values separated by ';'
    df['DiagnosisCode'] = df['DiagnosisCode'].str.split(';')
    df['ProcedureCode'] = df['ProcedureCode'].str.split(';')
    df = df.explode('DiagnosisCode')
    df = df.explode('ProcedureCode')

    # # storing the preprocessed DS
    # df.to_csv("data/preprocessed.csv", index=False)

    print("Preprocessing completed!")

    return df