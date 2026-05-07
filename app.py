# making a app for the user
import streamlit as st
import pandas as pd
import joblib
from src.preprocess import transform_input

# page configuration
st.set_page_config(page_title="Country Development Predictor", layout="wide")

# Loading Model Function
@st.cache_resource
def load_prediction_artifacts(model_path):
    artifacts = joblib.load(model_path)
    return artifacts['model'], artifacts['scaler']

# UI Header
st.title("🌍 Country Development Predictor")
st.markdown("Classifies a country's development status based on socio-economic and health factors.")

# Model Path
MODEL_PATH = r'C:\Users\HP\OneDrive\Desktop\country_development\model\country_model.pkl'

try:
    model, scaler = load_prediction_artifacts(MODEL_PATH)
except Exception as e:
    st.error(f"Error Loading Model : {e}")
    st.stop()

# INPUT FORM
with st.form("Prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Health & Life")
        child_mort = st.number_input("Child Mortality (per 1000 births)", min_value=0.0, max_value=300.0, value=50.0)
        life_expec = st.number_input("Life Expectancy (years)", min_value=0.0, max_value=100.0, value=70.0)
        total_fer  = st.number_input("Total Fertility Rate", min_value=0.0, max_value=10.0, value=2.5)
        health     = st.number_input("Health Spending (% of GDP)", min_value=0.0, max_value=20.0, value=6.0)

    with col2:
        st.subheader("Economy")
        income    = st.number_input("Per Capita Income (USD)", min_value=0.0, max_value=200000.0, value=5000.0)
        gdpp      = st.number_input("GDP Per Capita (USD)", min_value=0.0, max_value=200000.0, value=4000.0)
        inflation = st.number_input("Inflation Rate (%)", min_value=-10.0, max_value=100.0, value=3.0)

    with col3:
        st.subheader("Trade")
        exports = st.number_input("Exports (% of GDP)", min_value=0.0, max_value=200.0, value=30.0)
        imports = st.number_input("Imports (% of GDP)", min_value=0.0, max_value=200.0, value=35.0)

    submit = st.form_submit_button("Predict Development Status")

# PREDICTION LOGIC
if submit:
    input_data = {
        'child_mort': child_mort,
        'exports'   : exports,
        'health'    : health,
        'imports'   : imports,
        'income'    : income,
        'inflation' : inflation,
        'life_expec': life_expec,
        'total_fer' : total_fer,
        'gdpp'      : gdpp
    }

    X_scaled = transform_input(input_data, scaler)
    cluster  = model.predict(X_scaled)[0]

    cluster_labels = {
        0: ('Developed',      '🟢', 'High income, low child mortality, high life expectancy.'),
        1: ('Developing',     '🟡', 'Moderate income and health indicators, growing economy.'),
        2: ('Underdeveloped', '🟠', 'Low income, high child mortality, poor health infrastructure.'),
        3: ('Critical',       '🔴', 'Very low income, very high child mortality, needs urgent aid.')
    }

    label, emoji, description = cluster_labels.get(cluster, (f'Cluster {cluster}', '⚪', ''))

    st.divider()
    st.markdown(f"## {emoji} Development Status : **{label}**")
    st.info(description)
