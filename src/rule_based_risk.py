#src/rule_based_risk.py

def calculate_rule_based_risk(row):
    score = 0

    if row['Age'] > 60:
        score += 2
    elif row['Age'] > 45:
        score += 1

    if row['PrimaryDisease'] in ['Cardiovascular', 'Diabetes']:
        score += 3
    elif row['PrimaryDisease'] in ['Hypertension', 'Asthma']:
        score += 2

    if row['AmountBilled'] > 10000:
        score += 2
    elif row['AmountBilled'] > 5000:
        score += 1

    # if row['PreventiveGap'] == 'Yes':
    #     score += 2

    return score
