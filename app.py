import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import tensorflow as tf
import os
import re
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
import sklearn.compose._column_transformer

 #This fixes the _RemainderColsList issue with the newer scikit library
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        def __getstate__(self):
            return list(self)
        def __setstate__(self, state):
            self[:] = state
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
 

# --- Page Configuration ---
st.set_page_config(
    page_title="ML Loan Dataset Model Dashboard",
    layout="wide"
)

 
def patch_model_attributes(model):
    from sklearn.impute import SimpleImputer
    if hasattr(model, 'steps'):
        for name, step in model.steps:
            patch_model_attributes(step)
    elif hasattr(model, 'transformers_'):
        for name, transformer, columns in model.transformers_:
            if isinstance(transformer, (BaseEstimator, TransformerMixin)):
                patch_model_attributes(transformer)
    elif isinstance(model, SimpleImputer):
        if not hasattr(model, '_fill_dtype'):
            if hasattr(model, 'statistics_'):
                model._fill_dtype = model.statistics_.dtype
            else:
                model._fill_dtype = np.float64

# --- Sidebar: File Debugger ---
st.sidebar.header("System Info")
st.sidebar.write(f"Scikit-learn: {sklearn.__version__}")
st.sidebar.write(f"TensorFlow: {tf.__version__}")

st.title("ML Loan Dataset Model Deployment Dashboard")

# --- Model Loading ---
@st.cache_resource
def load_models():
    models = {}
    
    # 1. Load Loan Classifier Pipeline
    try:
        loaded_model = joblib.load('best_loan_classifier_pipeline.joblib')
        patch_model_attributes(loaded_model)
        models['loan_pipeline'] = loaded_model
    except FileNotFoundError:
        try:
            loaded_model = joblib.load('best_loan_classifier_pipeline.joblib.gz')
            patch_model_attributes(loaded_model)
            models['loan_pipeline'] = loaded_model
        except Exception as e:
            models['loan_pipeline'] = None
            st.error(f"Loan Classifier not found: {e}")
    except Exception as e:
        models['loan_pipeline'] = None
        st.error(f"Error loading Loan Classifier: {e}")

    # 2. Load Regression Model
    try:
        with open('regression_model.pkl', 'rb') as f:
            models['regression'] = pickle.load(f)
    except Exception:
        try:
            models['regression'] = joblib.load('regression_model.pkl')
        except Exception as e:
            models['regression'] = None
            st.warning(f"Regression Model not loaded: {e}")

    # 3. Load General Classification Model
    try:
        with open('classification_model.pkl', 'rb') as f:
            models['classifier'] = pickle.load(f)
    except Exception:
        try:
            models['classifier'] = joblib.load('classification_model.pkl')
        except Exception as e:
            models['classifier'] = None
            st.warning(f"Classification Model not loaded: {e}")

    # 4. Load Deep Learning Model
    try:
        models['deep_learning'] = tf.keras.models.load_model('deep_learning_model.h5')
    except Exception as e:
        models['deep_learning'] = None
        st.warning(f"Deep Learning Model not loaded: {e}")
        
    return models

models = load_models()

# --- Helper Function for Robust Parsing ---
def parse_input_string(input_str):
    if not input_str:
        return []
    
    Replace newlines with commas to ensure compatibility
    cleaned_str = input_str.replace('\n', ',')
    
     Replaces the non-breaking spaces or other common invisible characters
    cleaned_str = cleaned_str.replace('\xa0', ' ')
    
    Splits by comma
    tokens = cleaned_str.split(',')
    
    Gets rid of the whitespace and filter empty strings
    tokens = [t.strip() for t in tokens if t.strip()]
    
    Convert to floats with detailed error reporting.
    result = []
    for i, t in enumerate(tokens):
        try:
            result.append(float(t))
        except ValueError:
            raise ValueError(f"Item {i+1} ('{t}') is not a valid number.")
            
    return result

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
            term_val = int(term.split()[0]) 
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
    else:
        st.warning("Loan model not loaded.")

# --- TAB 2: The Regression Model ---
with tab2:
    st.header("Numerical Regression")
    if models.get('regression'):
        st.info("This model expects 20 numeric features.")
        default_20_feats = ", ".join(["0.5"] * 20)
        
        reg_input = st.text_area("Enter 20 numeric features (comma separated)", 
                                value=default_20_feats, 
                                height=100,
                                key="reg_input")
        
        if st.button("Predict Value (Regression)"):
            try:
                # Use robust parser
                feats = parse_input_string(reg_input)
                
                if len(feats) != 20:
                    st.error(f"Input Error: You provided {len(feats)} features, but the model expects exactly 20.")
                else:
                    input_arr = np.array([feats])
                    pred = models['regression'].predict(input_arr)
                    st.metric(label="Predicted Output", value=f"{pred[0]:.4f}")
            except ValueError as ve:
                st.error(f"Invalid Input: {ve}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.warning("Regression model not loaded.")

# --- TAB 3: The General Classification ---
with tab3:
    st.header("General Classification")
    if models.get('classifier'):
        st.info("This model expects 20 numeric features.")
        default_20_feats = ", ".join(["0.5"] * 20)
        
        class_input = st.text_area("Enter 20 numeric features (comma separated)", 
                                  value=default_20_feats, 
                                  height=100,
                                  key="class_input")
        
        if st.button("Classify Input"):
            try:
                # Use robust parser
                feats = parse_input_string(class_input)
                
                if len(feats) != 20:
                    st.error(f"Input Error: You provided {len(feats)} features, but the model expects exactly 20.")
                else:
                    pred = models['classifier'].predict([feats])
                    st.info(f"Predicted Class: {pred[0]}")
            except ValueError as ve:
                st.error(f"Invalid Input: {ve}")
            except Exception as e:
                st.error(f"Error processing input: {e}")
    else:
        st.warning("Classification model not loaded.")

# --- TAB 4: Deep Learning ---
with tab4:
    st.header("Deep Learning Inference")
    if models.get('deep_learning'):
        st.info("This model expects 69 features.")
        default_vals = ", ".join(["0.0"] * 69)
        dl_input = st.text_area("Enter input vector (comma separated)", default_vals, height=150)
        
        if st.button("Run Neural Net", type="primary"):
            try:
                # Use robust parser
                input_list = parse_input_string(dl_input)
                
                if len(input_list) != 69:
                    st.error(f"Error: Expected 69 inputs, but got {len(input_list)}.")
                else:
                    input_tensor = np.array([input_list])
                    prediction = models['deep_learning'].predict(input_tensor)
                    st.write("Model Output:")
                    st.dataframe(pd.DataFrame(prediction))
            except ValueError as ve:
                st.error(f"Invalid Input: {ve}")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Deep Learning model not loaded.")
