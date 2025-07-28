import streamlit as st
import pandas as pd
import joblib
from model.preprocess import preprocess
from model.featureEngineer import featureEngineer

st.set_page_config(page_title="🧠 Patient Risk Analyzer", layout="wide")

# Title and subtitle
st.markdown("<h1 style='text-align: center;'>🧠 Patient Risk Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict and monitor health risks using AI-powered analysis of patient claim data.</p>", unsafe_allow_html=True)
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
                summary_df = final_df.sort_values("PredictedRisk", ascending=False).drop_duplicates("MemberID")

                # Display final summary
                st.success("✅ Predictions generated successfully!")
                st.markdown("### 📋 Patient Risk Summary")
                st.dataframe(summary_df[['MemberID', 'Age', 'Gender','NumClaims', 'UniqueDiseases',
                                         'AvgCriticality', 'MaxCriticality', 'ChronicCount',
                                         'PredictedRisk', 'Recommendation']].reset_index(drop=True),
                             use_container_width=True)

                # Download option
                csv = summary_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results as CSV", data=csv,
                                   file_name="patient_predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"❌ Failed to process: {e}")
    else:
        st.info("Upload your CSV to begin.")

with tab2:
    st.markdown("## 🧾 How It Works")
    st.markdown("""
    - The model takes patient claim data with fields like age, gender, diagnosis, procedures, and billing.
    - It applies advanced feature engineering (chronic disease count, disease diversity, etc.).
    - Predictions are made using a machine learning model trained on past claim data.
    - Risk levels:
        - 🔴 High Risk: Likely needing immediate care
        - 🟠 Medium Risk: Requires monitoring
        - 🟢 Low Risk: Healthy or stable
    """)

    st.markdown("## 🤖 Model Details")
    st.markdown("""
    - **Algorithm:** Gradient Boosting (or similar)
    - **Features used:** Clinical & financial claim features
    - **Custom logic:** Feature extraction based on medical codes
    """)

    st.markdown("## 🔐 Data Privacy")
    st.markdown("All data is processed locally and not stored anywhere. Your privacy is safe!")
