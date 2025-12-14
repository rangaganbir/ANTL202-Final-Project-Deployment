import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import tensorflow as tf
import os
import sklearn.compose._column_transformer

#The patch for compatibillity
# This block fixes the "Can't get attribute '_RemainderColsList'" error we ran into.
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

st.title("ML Loan Dataset Model Deployment Dashboard")
st.markdown("""
This application serves four different machine learning models. 
Use this tabs below to switch between models and make predictions.
""")

# --- Model Loading ---
@st.cache_resource
def load_models():
    models = {}
    
    # 1. Load Loan Classifier Pipeline (.joblib)
    try:
        # Tries to load the compressed file first (best practice for GitHub)
        models['loan_pipeline'] = joblib.load('best_loan_classifier_pipeline.joblib.gz')
    except FileNotFoundError:
        try:
            # Fallback for uncompressed if needed
            models['loan_pipeline'] = joblib.load('best_loan_classifier_pipeline.joblib')
        except Exception as e:
            models['loan_pipeline'] = None
            st.error(f"Error loading Loan Classifier: {e}")
    except Exception as e:
        models['loan_pipeline'] = None
        st.error(f"Error loading Loan Classifier: {e}")

    # 2. Load Regression Model (.pkl)
    try:
        with open('regression_model.pkl', 'rb') as f:
            models['regression'] = pickle.load(f)
    except Exception as e:
        models['regression'] = None
        # error logged silently or use st.warning if critical

    # 3. Load General Classification Model (.pkl)
    try:
        with open('classification_model.pkl', 'rb') as f:
            models['classifier'] = pickle.load(f)
    except Exception as e:
        models['classifier'] = None

    # 4. Load Deep Learning Model (.h5)
    try:
        models['deep_learning'] = tf.keras.models.load_model('deep_learning_model.h5')
    except Exception as e:
        models['deep_learning'] = None
        st.warning(f"Deep Learning model could not be loaded: {e}")
        
    return models

# Load models once and cache them
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
        col1, col2 = st.columns(2)
        with col1:
            loan_amnt = st.number_input("Loan Amount ($)", min_value=500, value=10000, step=100)
            term = st.selectbox("Loan Term", [" 36 months", " 60 months"])
            int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=10.5, step=0.1)
        with col2:
            annual_inc = st.number_input("Annual Income ($)", min_value=0, value=60000, step=1000)
            fico_score = st.slider("FICO Score", 300, 850, 700)
            
        if st.button("Predict Loan Status", type="primary"):
            # Construct DataFrame with exact column names expected by the pipeline
            input_data = pd.DataFrame([{
                'loan_amnt': loan_amnt,
                'term': term,
                'int_rate': int_rate,
                'annual_inc': annual_inc,
                'fico_range_low': fico_score
            }])
            
            try:
                prediction = models['loan_pipeline'].predict(input_data)[0]
                proba = models['loan_pipeline'].predict_proba(input_data)[0]
                
                # Assuming 1 = Approved, 0 = Rejected
                if prediction == 1:
                    st.success(f"Prediction: Approved (Probability: {proba[1]:.2%})")
                else:
                    st.error(f"Prediction: Rejected (Probability: {proba[0]:.2%})")
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.info("Tip: Check that the input dictionary keys in `app.py` match your training features.")
    else:
        st.warning("Loan model not loaded. Please check 'best_loan_classifier_pipeline.joblib.gz'.")

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
        st.warning("Regression model not loaded.")

# --- TAB 3: The General Classification ---
with tab3:
    st.header("General Classification")
    if models.get('classifier'):
        user_input = st.text_input("Input Features (comma separated)", "1.5, 2.3, 4.0")
        
        if st.button("Classify Input"):
            try:
                # parsing string input to list/array
                feats = [float(x.strip()) for x in user_input.split(',')]
                pred = models['classifier'].predict([feats])
                st.info(f"Predicted Class: {pred[0]}")
            except Exception as e:
                st.error(f"Error processing input: {e}")
    else:
        st.warning("Classification model not loaded.")

# --- TAB 4: Deep Learning ---
with tab4:
    st.header("Deep Learning Inference")
    if models.get('deep_learning'):
        st.write("Input Data for Neural Network")
        dl_input = st.text_area("Enter input vector (comma separated)", "0.1, 0.5, 0.3, 0.2")
        
        if st.button("Run Neural Net", type="primary"):
            try:
                # Convert csv string to numpy array (1, n_features)
                input_list = [float(x.strip()) for x in dl_input.split(',')]
                input_tensor = np.array([input_list])
                
                prediction = models['deep_learning'].predict(input_tensor)
                st.write("Model Output:")
                st.dataframe(pd.DataFrame(prediction))
            except Exception as e:
                st.error(f"Error: {e}")
                st.caption("Ensure your input size matches the model's input layer shape.")
    else:
        st.warning("Deep Learning model not loaded.")
