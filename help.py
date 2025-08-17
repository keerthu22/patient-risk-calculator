import pandas as pd

df = pd.read_csv('data/patient_data.csv')

print(*df['DiagnosisCode'].unique())