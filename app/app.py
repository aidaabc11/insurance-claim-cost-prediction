import streamlit as st
import pandas as pd
import pickle

st.title("Insurance Claim Prediction App")

model = pickle.load(open("models/random_forest_model.pkl","rb"))

st.header("Enter Policyholder Information")

age = st.slider("Age",18,80,30)
months_as_customer = st.slider("Months as Customer",1,500,100)

if st.button("Predict Claim Cost"):

    data = pd.DataFrame({
        "age":[age],
        "months_as_customer":[months_as_customer]
    })

    prediction = model.predict(data)

    st.success(f"Predicted Claim Cost: ${prediction[0]:,.2f}")