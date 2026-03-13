import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/loan_default_model.pkl")

st.title("Loan Default Risk Predictor")

st.write("Enter applicant details to predict loan default risk.")

duration = st.slider("Loan Duration (months)", 6, 72, 24)
credit_amount = st.number_input("Credit Amount", 100, 20000, 5000)
age = st.slider("Age", 18, 75, 30)

input_data = pd.DataFrame({
    "duration": [duration],
    "credit_amount": [credit_amount],
    "age": [age]
})

if st.button("Predict Risk"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("High Risk of Default")
    else:
        st.success("Low Risk of Default")