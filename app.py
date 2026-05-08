import streamlit as st
import pandas as pd
import joblib
import uuid
from sqlalchemy import text
from src.preprocess import transform_input

# 1. PAGE CONFIGURATION (Must be the very first Streamlit command)
st.set_page_config(page_title="Country Development Predictor", layout="wide")

# 2. DATABASE CONNECTION
conn = st.connection("postgresql", type="sql")

# 3. SESSION STATE FOR LOGIN
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None

# --- AUTHENTICATION FUNCTIONS ---
def auth_page():
    st.title("🌍 Country Development Predictor")
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab2:
        st.subheader("Create Account")
        new_un = st.text_input("Username", key="reg_un")
        new_addr = st.text_area("Address", key="reg_addr")
        if st.button("Register"):
            if new_un:
                u_id = str(uuid.uuid4())
                try:
                    with conn.session as s:
                        s.execute(text("INSERT INTO users (user_id, username, address) VALUES (:id, :un, :ad)"),
                                  {"id": u_id, "un": new_un, "ad": new_addr})
                        s.commit()
                    st.success(f"Registered! Your ID: {u_id}. Please switch to Login.")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Please enter a username.")

    with tab1:
        st.subheader("Login")
        un = st.text_input("Enter Username", key="log_un")
        if st.button("Login"):
            res = conn.query(f"SELECT user_id FROM users WHERE username = '{un}'", ttl=0)
            if not res.empty:
                st.session_state.logged_in = True
                st.session_state.user_id = res.iloc[0]['user_id']
                st.session_state.username = un
                st.rerun()
            else:
                st.error("Username not found.")

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_prediction_artifacts(model_path):
    artifacts = joblib.load(model_path)
    return artifacts['model'], artifacts['scaler']

# --- MAIN APP LOGIC ---
if not st.session_state.logged_in:
    auth_page()
else:
    # 1. SIDEBAR LOGOUT
    st.sidebar.title("Navigation")
    st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    # 2. UI HEADER
    st.title("🌍 Country Development Predictor")
    st.markdown("Classifies a country's development status based on socio-economic and health factors.")

    # 3. LOAD MODEL
    MODEL_PATH = 'model/country_model.pkl'
    try:
        model, scaler = load_prediction_artifacts(MODEL_PATH)
    except Exception as e:
        st.error(f"Error Loading Model: {e}")
        st.stop()

    # 4. INPUT FORM
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

    # 5. PREDICTION & LOGGING
    if submit:
        input_data = {
            'child_mort': child_mort, 'exports': exports, 'health': health,
            'imports': imports, 'income': income, 'inflation': inflation,
            'life_expec': life_expec, 'total_fer': total_fer, 'gdpp': gdpp
        }

        # Run Prediction
        X_scaled = transform_input(input_data, scaler)
        cluster  = model.predict(X_scaled)[0]

        cluster_labels = {
            0: ('Developed',      '🟢', 'High income, low child mortality, high life expectancy.'),
            1: ('Developing',     '🟡', 'Moderate income and health indicators, growing economy.'),
            2: ('Underdeveloped', '🟠', 'Low income, high child mortality, poor health infrastructure.'),
            3: ('Critical',       '🔴', 'Very low income, very high child mortality, needs urgent aid.')
        }

        label, emoji, description = cluster_labels.get(cluster, (f'Cluster {cluster}', '⚪', ''))

        # Display Results
        st.divider()
        st.markdown(f"## {emoji} Development Status: **{label}**")
        st.info(description)

        # SAVE TO NEON DATABASE
        try:
            with conn.session as s:
                s.execute(text("""
                    INSERT INTO predictions (user_id, child_mort, income, gdpp, prediction_label)
                    VALUES (:uid, :cm, :inc, :gdp, :lbl)
                """), {
                    "uid": st.session_state.user_id,
                    "cm": child_mort, "inc": income, "gdp": gdpp, "lbl": label
                })
                s.commit()
            st.success("Prediction saved to your history.")
        except Exception as e:
            st.error(f"Database Logging Error: {e}")