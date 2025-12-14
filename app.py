import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- OUR MAIN PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Loan DataSet Platform",
    layout="wide"
)

# --- LOADS THE  MODELS ---
@st.cache_resource
def load_models():
    """
    Load the trained models. This will fail if model files are missing.
    """
    # Load your actual saved pipelines from Phase 4
    clf_model = joblib.load('clf_pipeline.joblib')
    reg_model = joblib.load('reg_pipeline.joblib')
    return clf_model, reg_model

clf_pipeline, reg_pipeline = load_models()

# --- OUR SIDEBAR INPUTS ---
# Using a standard bank icon image URL instead of emoji
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)
st.sidebar.title("Loan Application")
st.sidebar.write("Enter applicant details below:")

def user_input_features():
    # 1. Numeric Inputs (Matching your dataset columns)
    annual_income = st.sidebar.number_input("Annual Income ($)", min_value=10000.0, max_value=1000000.0, value=65000.0, step=1000.0)
    loan_amount = st.sidebar.number_input("Requested Loan Amount ($)", min_value=500.0, max_value=50000.0, value=15000.0, step=500.0)
    credit_score = st.sidebar.slider("Credit Score", min_value=300, max_value=850, value=700)
    debt_to_income_ratio = st.sidebar.slider("Debt-to-Income Ratio (DTI)", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
    age = st.sidebar.slider("Age", min_value=18, max_value=100, value=30)
    
    # 2. Categorical Inputs
    education_level = st.sidebar.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD", "Associate's"])
    employment_status = st.sidebar.selectbox("Employment Status", ["Employed", "Unemployed", "Self-employed", "Other"])
    marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
    loan_purpose = st.sidebar.selectbox("Loan Purpose", 
        ["Debt consolidation", "Credit card", "Home improvement", "Car", "Major purchase", "Medical", "Business", "Moving", "Other"])
    loan_term = st.sidebar.selectbox("Loan Term (Months)", [36, 60])

    # 3. Create DataFrame with EXACT column names from your CSV
    data = {
        'annual_income': annual_income,
        'loan_amount': loan_amount,
        'credit_score': credit_score,
        'debt_to_income_ratio': debt_to_income_ratio,
        'age': age,
        'education_level': education_level,
        'employment_status': employment_status,
        'marital_status': marital_status,
        'loan_purpose': loan_purpose,
        'loan_term': loan_term,
        # Default values for columns that might be in the model but not in the form
        'gender': 'Male', 
        'installment': loan_amount / loan_term, # Estimate
        'total_credit_limit': annual_income * 0.5, # Estimate
        'num_of_open_accounts': 5,
        'current_balance': 2000,
        'delinquency_history': 0
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- THE MAIN DASHBOARD ---
st.title("Loan DataSet")
st.markdown("""
This platform uses **data driven insights** to assess loan applications in real-time.
""")

# Display the Input Data
with st.expander("View Applicant Data Summary"):
    st.dataframe(input_df)

# --- PREDICTION LOGIC ---
if st.button("Assess Application Risk", type="primary"):
    
    col1, col2 = st.columns(2)

    # 1. CLASSIFICATION TASK (Will they pay back?)
    # Target: 'loan_paid_back' (1 = Yes, 0 = No) based on your CSV
    with col1:
        st.subheader("Risk Assessment")
        
        # Direct model usage - no mock fallbacks
        prediction = clf_pipeline.predict(input_df)[0]
        probability = clf_pipeline.predict_proba(input_df)[0][1] 

        # Logic: In your CSV, loan_paid_back=1 usually means Good, 0 means Default
        if prediction == 0: 
            st.error(f"**Status: HIGH RISK (Likely Default)**")
            st.metric(label="Repayment Probability", value=f"{1-probability:.1%}", delta="-High Risk")
        else:
            st.success(f"**Status: APPROVED (Likely to Repay)**")
            st.metric(label="Repayment Probability", value=f"{probability:.1%}", delta="Safe")

    # 2. REGRESSION TASK (Predicted Loan Amount Limit)
    # Comparing what they asked for vs. what the model thinks they should get
    with col2:
        st.subheader("Credit Limit Intelligence")
        
        # Direct model usage
        pred_value = reg_pipeline.predict(input_df)[0]

        st.info(f"**Recommended Loan Limit**")
        st.metric(label="AI Suggested Limit", value=f"${pred_value:,.2f}")
        
        requested = input_df['loan_amount'][0]
        if requested > pred_value:
            st.warning(f"Requested amount (${requested:,.0f}) exceeds recommended limit.")
        else:
            st.success(f"Requested amount is within safe limits.")

    # 3. THE EXPLAINABLE INSIGHTS
    st.divider()
    st.subheader("Automated Insights")
    
    if input_df['debt_to_income_ratio'][0] > 0.4:
        st.warning("• **High DTI:** Debt-to-Income ratio is above 40%, indicating financial strain.")
    
    if input_df['credit_score'][0] < 620:
        st.error("• **Credit Alert:** Credit score is below prime threshold (620).")
    elif input_df['credit_score'][0] > 720:
        st.success("• **Excellent Credit:** High credit score supports approval.")

    if input_df['employment_status'][0] == "Unemployed":
         st.error("• **Employment Alert:** Lack of active employment significantly increases risk.")


st.markdown("---")

st.markdown("© 2025 SecureBank Team | Powered by Streamlit")
