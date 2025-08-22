import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load model & scaler once
model = joblib.load("C:/Users/Roshan/customer_churn_project/model/churn_model.pkl")
scaler = joblib.load("C:/Users/Roshan/customer_churn_project/model/scaler.pkl")

st.title("Customer Churn Prediction Dashboard")

uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Preprocess
    df_scaled = scaler.transform(df[['Age','MonthlyCharges','Tenure']])
    df['Churn_Probability'] = model.predict_proba(df_scaled)[:,1]
    df['Retention_Score'] = df['Churn_Probability'] * df['MonthlyCharges']

    # Show table
    st.dataframe(df)

    # Show chart
    fig = px.bar(df, x='CustomerID', y='Churn_Probability', color='Retention_Score')
    st.plotly_chart(fig)
