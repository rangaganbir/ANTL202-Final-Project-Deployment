import streamlit as st
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
import os

# 1. This will load the models 
@st.cache_resource
def load_models():
    # --- Classification Model ---
    # Try loading your specific file first, then fall back to the standard name
    if os.path.exists('best_loan_classifier_pipeline.joblib'):
        clf_model = joblib.load('best_loan_classifier_pipeline.joblib')
    elif os.path.exists('classification_model.pkl'):
        clf_model = joblib.load('classification_model.pkl')
    else:
        clf_model = None
    
    # --- Regression Model ---
    if os.path.exists('regression_model.pkl'):
        reg_model = joblib.load('regression_model.pkl')
    else:
        reg_model = None

    # --- Deep Learning Model ---
    if os.path.exists('deep_learning_model.h5'):
        dl_model = tf.keras.models.load_model('deep_learning_model.h5')
    else:
        dl_model = None

    return clf_model, reg_model, dl_model

clf_pipeline, reg_pipeline, dl_model = load_models()

# 2. App Title
st.title("Loan Dataset Application")
st.markdown("This platform assesses customer risk.")

# 3. Sidebar Inputs
st.sidebar.header("Customer Information")

def user_input_features():
    # Numeric Inputs
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
    annual_income = st.sidebar.number_input("Annual Income", min_value=0, value=50000)
    monthly_income = annual_income / 12
    debt_to_income = st.sidebar.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
    credit_score = st.sidebar.slider("Credit Score", 300, 850, 700)
    loan_term = st.sidebar.selectbox("Loan Term (Months)", [36, 60])
    installment = st.sidebar.number_input("Monthly Installment", value=300)
    open_acc = st.sidebar.number_input("Open Accounts", value=5)
    total_credit = st.sidebar.number_input("Total Credit Limit", value=20000)
    current_bal = st.sidebar.number_input("Current Balance", value=5000)
    delinq_hist = st.sidebar.number_input("Delinquency History (Years)", value=0)
    num_delinq = st.sidebar.number_input("Number of Delinquencies", value=0)
    pub_rec = st.sidebar.number_input("Public Records", value=0)
    interest_rate = st.sidebar.number_input("Interest Rate (%)", value=10.0)

    # Categorical Inputs
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    marital = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    edu = st.sidebar.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD", "Associate's"])
    emp_status = st.sidebar.selectbox("Employment Status", ["Employed", "Unemployed", "Self-Employed", "Retired"])
    purpose = st.sidebar.selectbox("Loan Purpose", ["Debt consolidation", "Credit card", "Home improvement", "Other"])
    grade = st.sidebar.selectbox("Grade Subgrade", ["A1", "B1", "C1", "D1", "E1", "F1", "G1"]) 

    data = {
        'age': age, 'annual_income': annual_income, 'monthly_income': monthly_income,
        'debt_to_income_ratio': debt_to_income, 'credit_score': credit_score,
        'interest_rate': interest_rate, 'loan_term': loan_term, 'installment': installment,
        'num_of_open_accounts': open_acc, 'total_credit_limit': total_credit,
        'current_balance': current_bal, 'delinquency_history': delinq_hist,
        'num_of_delinquencies': num_delinq, 'public_records': pub_rec,
        'gender': gender, 'marital_status': marital, 'education_level': edu,
        'employment_status': emp_status, 'loan_purpose': purpose, 'grade_subgrade': grade
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()
st.subheader("Customer Profile")
st.write(input_df)

# 5. This makes the predictions on loan repayment
if st.button("Assess Risk"):
    
    # --- Classification Result ---
    if clf_pipeline:
        try:
            risk_prob = clf_pipeline.predict_proba(input_df)[0][1]
            risk_pred = "Default" if risk_prob > 0.5 else "Paid Off"
            if risk_prob > 0.5:
                st.error(f"High Risk Applicant (Prob: {risk_prob:.2%})")
            else:
                st.success(f"Low Risk Applicant (Prob: {risk_prob:.2%})")
        except Exception as e:
            st.warning("Error using Classification Model. Check inputs.")
    else:
        st.warning("Classification Model not found.")

    # --- Regression Result ---
    if reg_pipeline:
        try:
            loan_amount_pred = reg_pipeline.predict(input_df)[0]
            st.metric("Predicted Loan Amount", f"${loan_amount_pred:,.2f}")
        except:
            st.warning("Could not predict loan amount.")
    else:
        st.info("Regression Model not uploaded yet.")

    # --- Deep Learning Result ---
    if dl_model:
        try:
            # DL usually needs scaled data. We try to use the classifier's scaler if available
            if clf_pipeline:
                preprocessor = clf_pipeline.named_steps['preprocessor']
                input_scaled = preprocessor.transform(input_df)
                dl_prob = dl_model.predict(input_scaled)[0][0]
                st.metric("DL Risk Score", f"{dl_prob:.2%}")
        except:
            st.warning("Could not run Deep Learning model.")
    else:
        st.info("Deep Learning Model not uploaded yet.")
