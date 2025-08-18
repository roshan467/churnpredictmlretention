import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.title("Customer Churn Prediction Dashboard")

uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    model = joblib.load("../models/rf_model.pkl")
    scaler = joblib.load("../models/scaler.pkl")
    df_scaled = scaler.transform(df[['Age','MonthlyCharges','Tenure']])
    df['Churn_Probability'] = model.predict_proba(df_scaled)[:,1]
    df['Retention_Score'] = df['Churn_Probability'] * df['MonthlyCharges']
    st.dataframe(df)
    fig = px.bar(df, x='CustomerID', y='Churn_Probability', color='Retention_Score')
    st.plotly_chart(fig)
