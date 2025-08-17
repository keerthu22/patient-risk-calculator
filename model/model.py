import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.model_selection import StratifiedKFold

def model(df):
    # Define Features & Target

    print("Shape of training dataset: ", df.shape)

    features = ['Age', 'Gender', 'NumClaims', 'UniqueDiseases', 'AvgCriticality', 'MaxCriticality', 'ChronicCount', 'TotalAmountBilled', 'DiagnosisCode', 'ProcedureCode']

    print("Features used for training: ", features)

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

    strat_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=strat_kfold)

    # cv_scores = cross_val_score(rf, X_train, y_train, cv=5)

    print("Cross-validation scores:", cv_scores)
    print("Mean CV Accuracy:", np.mean(cv_scores))

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Evaluation
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save Model and Encoders
    joblib.dump(rf, 'output/patient_risk_model.pkl')

    print("Models are saved successfully!")
