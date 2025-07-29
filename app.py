import streamlit as st
import pandas as pd
import joblib
from model.preprocess import preprocess
from model.featureEngineer import featureEngineer
import joblib
from src.disease_mapper import map_diseases
from docx import Document
from docx.shared import Inches
import io

def generate_patient_report(df):
    patient_docs = {}
    grouped = df.groupby("MemberID")

    for member_id, group in grouped:
        doc = Document()
        doc.add_heading("🧠 Patient Risk Report", level=0)
        doc.add_paragraph(f"Member ID: {member_id}")
        doc.add_paragraph(f"Age: {group['Age'].iloc[0]}")
        doc.add_paragraph(f"Gender: {'Female' if le_gender.inverse_transform([group['Gender'].iloc[0]])[0]=='F' else 'Male'}")
        doc.add_paragraph(f"Total Claims: {len(group)}")

        doc.add_heading("📄 Claim Details", level=1)
        for i, row in group.iterrows():
            doc.add_paragraph(
                f"• Claim ID: {row['ClaimID']}\n"
                f"  - Diagnosis Code: {le_diag.inverse_transform([row['DiagnosisCode']])[0]} ({map_diseases(le_diag.inverse_transform([row['DiagnosisCode']])[0])})\n"
                f"  - Procedure Code: {le_proc.inverse_transform([row['ProcedureCode']])[0]}\n"
                f"  - Amount Billed: ${row['AmountBilled']}\n",
                style='List Bullet'
            )

        doc.add_paragraph(f"  Avg Criticality: {row['AvgCriticality']}\n"
                          f"  Max Criticality: {row['MaxCriticality']}\n"
                          f"  Chronic Count: {row['ChronicCount']}\n"
                          f"  Risk Score: {row['PredictedRisk']}\n"
                          f"  Recommendation: {generate_recommendation(row['PredictedRisk'])}\n"
                          f"  Preventive Care Advice: {row['PreventiveCareAdvice']}\n")

        # Save to bytes
        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        patient_docs[member_id] = buffer

    return patient_docs

st.set_page_config(page_title="🧠 Patient Risk Analyzer", layout="wide")

# Title and subtitle
st.markdown("<h1 style='text-align: center; color: white;'>🧠 Patient Risk Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white;'>Predict and monitor health risks using AI-powered analysis of patient claim data.</p>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
        /* Full app background */
        .stApp {
            background: linear-gradient(to bottom right, #accbee, #e7f0fd);  /* pastel blue gradient */
            background-size: cover;
        }

        /* Global text styling */
        html, body, [class*="css"] {
            color: #1e1e1e;  /* Dark gray text */
            font-family: 'Segoe UI', sans-serif;
        }

        /* Headings and paragraph text */
        h1, h2, h3, h4, h5, h6, p {
            color: #1a1a1a;
        }

        /* Data table */
        .stDataFrameContainer {
            background-color: rgba(255, 255, 255, 0.96) !important;
            border-radius: 12px;
            padding: 12px;
        }

        /* Buttons */
        .stButton>button, .stDownloadButton>button {
            border-radius: 8px;
            background-color: #0d47a1;
            color: white;
            font-weight: 500;
        }

        .stDownloadButton>button:hover, .stButton>button:hover {
            background-color: #08306b;
            color: white;
        }

        /* Tabs background and text color */
        .stTabs [role="tablist"] {
            background-color: #dbeafe;  /* Soft blue tab background */
            border-radius: 12px;
            padding: 6px;
        }

        .stTabs [role="tab"] {
            color: #1e3a8a;
            font-weight: 600;
        }

        .stTabs [aria-selected="true"] {
            background-color: #c7d2fe !important; /* Active tab */
            border-radius: 8px;
        }

        /* File uploader text */
        .stFileUploader label {
            color: #1a1a1a !important;
        }

    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# Tabs
tab1, tab2 = st.tabs(["📁 Upload & Analyze", "📊 About & Insights"])

with tab1:
    st.markdown("### 📤 Upload Patient CSV File")
    uploaded_file = st.file_uploader("Upload your patient claims CSV file", type=["csv"])

    if uploaded_file:
        try:
            # Show uploaded file
            raw_df = pd.read_csv(uploaded_file)
            st.markdown("#### 📄 Uploaded Input Data")
            st.dataframe(raw_df, use_container_width=True)

            with st.spinner("🔄 Processing data..."):
                # Preprocess and feature engineer
                cleaned_df = preprocess(raw_df)
                final_df = featureEngineer(cleaned_df).reset_index(drop=True)

                # Load trained model
                model = joblib.load("output/patient_risk_model.pkl")

                # Define features
                features = ['Age', 'Gender', 'NumClaims', 'UniqueDiseases', 'AvgCriticality',
                            'MaxCriticality', 'ChronicCount', 'DiagnosisCode',
                            'ProcedureCode', 'AmountBilled']

                # Predict
                final_df['PredictedRisk'] = model.predict(final_df[features])

                # Recommendation logic
                def generate_recommendation(score):
                    if score > 4:
                        return "🔴 Immediate Attention"
                    elif score > 2:
                        return "🟠 Monitor Closely"
                    else:
                        return "🟢 Low Risk"

                final_df['Recommendation'] = final_df['PredictedRisk'].apply(generate_recommendation)

                # Group by MemberID (show one row per patient)
                def get_most_frequent(series):
                    return series.mode().iloc[0] if not series.mode().empty else series.iloc[0]

                summary_df = final_df.groupby("MemberID").agg({
                    "Age": "first",
                    "Gender": "first",
                    "DiagnosisCode": get_most_frequent,
                    "ProcedureCode": get_most_frequent,
                    "DiseaseName": get_most_frequent,
                    "NumClaims": "first",
                    "UniqueDiseases": "first",
                    "AvgCriticality": "first",
                    "MaxCriticality": "first",
                    "ChronicCount": "first",
                    "PredictedRisk": "first",
                    "Recommendation": "first"
                }).reset_index()

                
                # Load the encoders
                le_gender = joblib.load("output/le_gender.pkl")
                le_proc = joblib.load("output/le_proc.pkl")
                le_diag = joblib.load("output/le_diag.pkl")

                # Inverse transform the encoded columns
                summary_df['Gender'] = le_gender.inverse_transform(summary_df['Gender'])
                summary_df['DiagnosisCode'] = le_diag.inverse_transform(summary_df['DiagnosisCode'])
                summary_df['ProcedureCode'] = le_proc.inverse_transform(summary_df['ProcedureCode'])
                summary_df['DiseaseName'] = summary_df['DiagnosisCode'].apply(map_diseases)

                # Display final summary
                st.success("✅ Predictions generated successfully!")
                st.markdown("### 📋 Patient Risk Summary")
                st.dataframe(summary_df[['MemberID', 'Age', 'Gender', 'ProcedureCode', 'DiagnosisCode', 'DiseaseName', 'NumClaims', 'UniqueDiseases',
                                         'AvgCriticality', 'MaxCriticality', 'ChronicCount',
                                         'PredictedRisk', 'Recommendation']].reset_index(drop=True),
                             use_container_width=True)

                # Download option
                csv = summary_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results as CSV", data=csv,
                                   file_name="patient_predictions.csv", mime="text/csv")
                
                # Generate reports for each patient
                patient_reports = generate_patient_report(final_df)

                st.markdown("### 📄 Download Individual Patient Reports (DOCX)")
                for member_id, doc_buffer in patient_reports.items():
                    st.download_button(
                        label=f"📥 Download Report for Member {member_id}",
                        data=doc_buffer,
                        file_name=f"Patient_Report_{member_id}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )

        except Exception as e:
            st.error(f"❌ Failed to process: {e}")
    else:
        st.info("Upload your CSV to begin.")

with tab2:
   with tab2:
    st.markdown("""
        <div style='color: #1a1a1a; font-size: 16px; line-height: 1.6;'>

        <h2>🧾 How It Works</h2>
        <ul>
            <li>The model takes patient claim data with fields like age, gender, diagnosis, procedures, and billing.</li>
            <li>It applies advanced feature engineering (chronic disease count, disease diversity, etc.).</li>
            <li>Predictions are made using a machine learning model trained on past claim data.</li>
            <li>Risk levels:
                <ul>
                    <li>🔴 <strong>High Risk:</strong> Likely needing immediate care</li>
                    <li>🟠 <strong>Medium Risk:</strong> Requires monitoring</li>
                    <li>🟢 <strong>Low Risk:</strong> Healthy or stable</li>
                </ul>
            </li>
        </ul>

        <h2>🤖 Model Details</h2>
        <ul>
            <li><strong>Algorithm:</strong> Gradient Boosting (or similar)</li>
            <li><strong>Features used:</strong> Clinical & financial claim features</li>
            <li><strong>Custom logic:</strong> Feature extraction based on medical codes</li>
        </ul>

        <h2>🔐 Data Privacy</h2>
        <p>All data is processed locally and not stored anywhere. Your privacy is safe!</p>

        </div>
    """, unsafe_allow_html=True)
