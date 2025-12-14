import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import tensorflow as tf
import os
import re
import sklearn.compose._column_transformer

# --- This is to get the RemainderColsList working ---
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        def __getstate__(self):
            return list(self)
        def __setstate__(self, state):
            self[:] = state
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
# 


# --- Page Configuration ---
st.set_page_config(
    page_title="ML Loan Dataset Model Dashboard",
    layout="wide"
)

# --- Sidebar: File Debugger ---
st.sidebar.header("File System Debugger")
st.sidebar.write("Files found in current directory:")
try:
    files = os.listdir('.')
    file_info = []
    for f in files:
        try:
            size_bytes = os.path.getsize(f)
            size_str = f"{size_bytes} bytes"
            if size_bytes > 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            if size_bytes > 1024 * 1024:
                size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
            file_info.append(f"{f} ({size_str})")
        except Exception:
            file_info.append(f"{f} (size unknown)")
    
    st.sidebar.code('\n'.join(file_info))
except Exception as e:
    st.sidebar.error(f"Could not list files: {e}")

st.title("ML Loan Dataset Model Deployment Dashboard")

# --- Model Loading ---
@st.cache_resource
def load_models():
    models = {}
    
    # 1. Load Loan Classifier Pipeline
    # Tries .gz first, then falls back to standard .joblib
    try:
        models['loan_pipeline'] = joblib.load('best_loan_classifier_pipeline.joblib.gz')
    except FileNotFoundError:
        try:
            models['loan_pipeline'] = joblib.load('best_loan_classifier_pipeline.joblib')
        except Exception as e:
            models['loan_pipeline'] = None
            st.error(f"Loan Classifier not found: {e}")
    except Exception as e:
        models['loan_pipeline'] = None
        st.error(f"Error loading Loan Classifier: {e}")

    # 2. Load Regression Model
    # UPDATED: Changed from pickle.load to joblib.load to handle compression
    try:
        models['regression'] = joblib.load('regression_model.pkl')
    except Exception as e:
        models['regression'] = None
        st.warning(f"Regression Model not loaded: {e}. (Check if file size > 1KB in sidebar)")

    # 3. Load General Classification Model
    # UPDATED: Changed from pickle.load to joblib.load to handle compression
    try:
        models['classifier'] = joblib.load('classification_model.pkl')
    except Exception as e:
        models['classifier'] = None
        st.warning(f"Classification Model not loaded: {e}. (Check if file size > 1KB in sidebar)")

    # 4. Load Deep Learning Model
    try:
        models['deep_learning'] = tf.keras.models.load_model('deep_learning_model.h5')
    except Exception as e:
        models['deep_learning'] = None
        st.warning(f"Deep Learning Model not loaded: {e}")
        
    return models

models = load_models()

# --- Tabs Interface ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Loan Classifier", 
    "Regression", 
    "Classification", 
    "Deep Learning"
])

# --- TAB 1: The Loan Classifier ---
with tab1:
    st.header("Loan Approval Predictor")
    if models.get('loan_pipeline'):
        col1, col2, col3 = st.columns(3)
        with col1:
            loan_amnt = st.number_input("Loan Amount ($)", min_value=500, value=10000)
            term = st.selectbox("Loan Term", ["36 months", "60 months"])
            int_rate = st.number_input("Interest Rate (%)", value=10.5)
            grade = st.selectbox("Grade", ["A", "B", "C", "D", "E", "F", "G"])
        with col2:
            annual_inc = st.number_input("Annual Income ($)", value=60000)
            fico_score = st.slider("Credit Score (FICO)", 300, 850, 700)
            home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN"])
            emp_length = st.selectbox("Employment Length", ["< 1 year", "1 year", "5 years", "10+ years"])
        with col3:
            age = st.number_input("Age", min_value=18, value=30)
            dti = st.number_input("Debt-to-Income Ratio", value=15.0)
            open_acc = st.number_input("Open Accounts", value=5)
            
        if st.button("Predict Loan Status", type="primary"):
            # FIX: Convert string inputs to numbers for the model
            term_val = int(term.split()[0]) # "36 months" -> 36
            
            # Helper to extract numbers from "10+ years" or "< 1 year"
            emp_val = 0
            if emp_length:
                nums = re.findall(r'\d+', str(emp_length))
                if nums:
                    emp_val = int(nums[0])

            input_data = pd.DataFrame([{
                'loan_amount': loan_amnt,
                'loan_term': term_val, 
                'interest_rate': int_rate,
                'annual_income': annual_inc,
                'credit_score': fico_score,
                'age': age,
                'debt_to_income_ratio': dti,
                'num_of_open_accounts': open_acc,
                'monthly_income': annual_inc / 12, 
                'installment': (loan_amnt * (int_rate/100)) / 12,
                'gender': 'Male',
                'marital_status': 'Single',
                'education_level': 'Bachelor',
                'employment_status': 'Employed',
                'loan_purpose': 'Debt Consolidation',
                'grade_subgrade': grade + "1",
                'delinquency_history': 0,
                'num_of_delinquencies': 0,
                'total_credit_limit': 20000,
                'current_balance': 5000,
                'public_records': 0,
                'emp_length': emp_val,
            }])
            
            try:
                prediction = models['loan_pipeline'].predict(input_data)[0]
                proba = models['loan_pipeline'].predict_proba(input_data)[0]
                
                if prediction == 1:
                    st.success(f"Prediction: Approved (Probability: {proba[1]:.2%})")
                else:
                    st.error(f"Prediction: Rejected (Probability: {proba[0]:.2%})")
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.write("Debug - Input Columns:", input_data.columns.tolist())
    else:
        st.warning("Loan model not loaded.")

# --- TAB 2: The Regression Model ---
with tab2:
    st.header("Numerical Regression")
    if models.get('regression'):
        f1 = st.number_input("Feature 1", value=0.0)
        f2 = st.number_input("Feature 2", value=0.0)
        
        if st.button("Predict Value"):
            input_arr = np.array([[f1, f2]])
            try:
                pred = models['regression'].predict(input_arr)
                st.metric(label="Predicted Output", value=f"{pred[0]:.4f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.warning("Regression model not loaded. Check the file list in the sidebar.")

# --- TAB 3: The General Classification ---
with tab3:
    st.header("General Classification")
    if models.get('classifier'):
        user_input = st.text_input("Input Features (comma separated)", "1.5, 2.3, 4.0")
        
        if st.button("Classify Input"):
            try:
                feats = [float(x.strip()) for x in user_input.split(',')]
                pred = models['classifier'].predict([feats])
                st.info(f"Predicted Class: {pred[0]}")
            except Exception as e:
                st.error(f"Error processing input: {e}")
    else:
        st.warning("Classification model not loaded. Check the file list in the sidebar.")

# --- TAB 4: Deep Learning ---
with tab4:
    st.header("Deep Learning Inference")
    if models.get('deep_learning'):
        st.write("Input Data for Neural Network (Requires 69 features)")
        
        default_vals = ", ".join(["0.0"] * 69)
        dl_input = st.text_area("Enter input vector (comma separated)", default_vals, height=150)
        
        if st.button("Run Neural Net", type="primary"):
            try:
                input_list = [float(x.strip()) for x in dl_input.split(',')]
                
                if len(input_list) != 69:
                    st.error(f"Error: Expected 69 inputs, but got {len(input_list)}.")
                else:
                    input_tensor = np.array([input_list])
                    prediction = models['deep_learning'].predict(input_tensor)
                    st.write("Model Output:")
                    st.dataframe(pd.DataFrame(prediction))
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Deep Learning model not loaded. Check the file list in the sidebar.")
