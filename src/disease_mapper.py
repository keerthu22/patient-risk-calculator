# src/disease_mapper.py

def map_diseases(code_str):
    """
    Maps ICD-10 code to a general disease category based on the first character.
    """
    code_map = {
        'A': 'Infectious diseases',
        'B': 'Infectious diseases',
        'C': 'Cancer',
        'D': 'Blood disorders',
        'E': 'Endocrine/Metabolic',
        'F': 'Mental disorders',
        'G': 'Neurological disorders',
        'H': 'Eye/Ear diseases',
        'I': 'Cardiovascular',
        'J': 'Respiratory',
        'K': 'Digestive',
        'L': 'Skin diseases',
        'M': 'Musculoskeletal',
        'N': 'Genitourinary',
        'O': 'Pregnancy-related',
        'P': 'Perinatal conditions',
        'Q': 'Congenital anomalies',
        'R': 'General symptoms',
        'S': 'Injuries and poisonings',
        'T': 'Injuries and poisonings',
        'Z': 'General health status'
    }

    if not code_str or not isinstance(code_str, str):
        return 'Unknown'

    code_str = code_str.strip().upper()
    return code_map.get(code_str[0], 'Unknown')
