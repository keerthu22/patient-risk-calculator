#src/model_loader.py

import joblib

def load_model(model_path='output/patient_risk_model.pkl'):
    try:
        model = joblib.load(model_path)
        le_gender = joblib.load('output/le_gender.pkl')
        le_diagnosis = joblib.load('output/le_diag.pkl')  # for DiagnosisCode
        le_procedure = joblib.load('output/le_proc.pkl')  # for ProcedureCode

        return {
            'model': model,
            'le_gender': le_gender,
            'le_diagnosis': le_diagnosis,
            'le_procedure': le_procedure
        }
    except Exception as e:
        print(f"Failed to load model or encoders: {e}")
        return None
