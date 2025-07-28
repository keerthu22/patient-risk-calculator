import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

def model(df):
    # Define Features & Target
    features = ['Age', 'Gender', 'NumClaims', 'UniqueDiseases', 'AvgCriticality', 'MaxCriticality', 'ChronicCount', 'DiagnosisCode', 'ProcedureCode', 'AmountBilled']
    X = df[features]
    y = df['RiskScore'].astype(int)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

    # Train Model
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )

    cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
    print("Cross-validation scores:", cv_scores)
    print("Mean CV Accuracy:", np.mean(cv_scores))

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Evaluation
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # # Save Model and Encoders
    joblib.dump(rf, 'output/patient_risk_model.pkl')
    # joblib.dump(le_gender, 'output/le_gender.pkl')
    # joblib.dump(le_diag, 'output/le_diag.pkl')
    # joblib.dump(le_proc, 'output/le_proc.pkl')

    print("Models are saved successfully!")
