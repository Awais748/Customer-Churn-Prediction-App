import streamlit as st
from pathlib import Path
import pandas as pd
from joblib import load

MODEL_PATH = Path(__file__).resolve().with_name("random_forest_model.joblib")

@st.cache_resource
def load_model():
    return load(MODEL_PATH)

# Create a Streamlit app
st.title("Customer Churn Prediction App")

# Input fields for feature values on the main screen
st.header("Enter Customer Information")
tenure = st.number_input("Tenure (in months)", min_value=0, max_value=100, value=1)
internet_service = st.selectbox("Internet Service", ('DSL', 'Fiber optic', 'No'))
contract = st.selectbox("Contract", ('Month-to-month', 'One year', 'Two year'))
monthly_charges = st.number_input("Monthly Charges", min_value=0, max_value=200, value=50)
total_charges = st.number_input("Total Charges", min_value=0, max_value=10000, value=0)

# Map input values to numeric using the label mapping
label_mapping = {
    'DSL': 0,
    'Fiber optic': 1,
    'No': 2,
    'Month-to-month': 0,
    'One year': 1,
    'Two year': 2,
}
internet_service = label_mapping[internet_service]
contract = label_mapping[contract]

# Make a prediction using the model
features = pd.DataFrame([{
    "tenure": tenure,
    "InternetService": internet_service,
    "Contract": contract,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
}])
prediction = load_model().predict(features)

# Display the prediction result on the main screen
st.header("Prediction Result")
if prediction[0] == 0:
    st.success("This customer is likely to stay.")
else:
    st.error("This customer is likely to churn.")
