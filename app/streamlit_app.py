import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.title("📊 Customer Churn Prediction + Retention Dashboard")

# Load model & scaler
try:
    model = joblib.load("models/rf_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
except:
    st.error("Model files not found. Please place rf_model.pkl and scaler.pkl inside a 'models/' folder.")

uploaded_file = st.file_uploader("📂 Upload Customer CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_cols = ['Age','MonthlyCharges','Tenure']
    if all(col in df.columns for col in required_cols):
        df_scaled = scaler.transform(df[required_cols])
        df['Churn_Probability'] = model.predict_proba(df_scaled)[:,1]
        df['Retention_Score'] = df['Churn_Probability'] * df['MonthlyCharges']

        if 'CustomerID' not in df.columns:
            df['CustomerID'] = range(1, len(df)+1)

        st.subheader("🔎 Prediction Results")
        st.dataframe(df)

        st.subheader("📈 Churn Probability by Customer")
        fig = px.bar(df, x='CustomerID', y='Churn_Probability', color='Retention_Score',
                     labels={"Churn_Probability":"Churn Probability", "Retention_Score":"Retention Score"})
        st.plotly_chart(fig)
    else:
        st.error(f"Missing columns in dataset! Required: {required_cols}")

