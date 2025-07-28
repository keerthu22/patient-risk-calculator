from src.preprocessing import load_and_clean_data
from src.disease_mapping import get_max_risk_level
from src.risk_scoring import calculate_risk_score
from src.preventive_engine import generate_recommendations
from src.model_training import train_risk_model
import os

if __name__ == "__main__":
    os.makedirs('output', exist_ok=True)

    # Load and preprocess the data
    df = load_and_clean_data('data/synthetic_patient_claims.csv')

    # Map diseases to base risk levels
    df['BaseRiskLevel'] = df['PrimaryDisease'].apply(get_max_risk_level)

    # Calculate final risk scores
    df['FinalRiskScore'] = df.apply(calculate_risk_score, axis=1)

    # Generate preventive care recommendations
    df['Recommendation'] = df.apply(generate_recommendations, axis=1)

    # Train and save a model (for future ML prediction)
    train_risk_model(df)

    # Save the processed output
    df.to_csv('output/patient_risk_results.csv', index=False)

    print("✅ Risk report generated: output/patient_risk_results.csv")
