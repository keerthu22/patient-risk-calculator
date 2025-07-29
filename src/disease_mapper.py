# src/disease_mapper.py

def map_diseases(code_str):
    """
    Maps ICD-10 code to a general disease category based on the first character.
    """
    code_map = {
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
    
    code_str = code_str.strip().upper()
    return code_map.get(code_str[:3], 'Unknown')