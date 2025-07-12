import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import our predictor
from single_patient_predictor import SinglePatientPredictor

# Page configuration
st.set_page_config(
    page_title="Diabetic Patient Readmission Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    .medium-risk {
        background-color: #fff3e0;
        border: 2px solid #ff9800;
    }
    .low-risk {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def load_predictor():
    """Load the prediction model"""
    try:
        predictor = SinglePatientPredictor()
        return predictor
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def create_patient_form():
    """Create a form for patient data input"""
    st.subheader(" Patient Information")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Demographics**")
        race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.selectbox("Age Group", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", 
                                       "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"])
        
        st.write("**Admission Details**")
        admission_type_id = st.selectbox("Admission Type", [1, 2, 3, 4, 5, 6, 7, 8], 
                                       format_func=lambda x: {1: "Emergency", 2: "Urgent", 3: "Elective", 
                                                             4: "Newborn", 5: "Not Available", 6: "NULL", 
                                                             7: "Trauma Center", 8: "Not Mapped"}[x])
        discharge_disposition_id = st.selectbox("Discharge Disposition", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
        admission_source_id = st.selectbox("Admission Source", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27])
        
        st.write("**Hospital Stay**")
        time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 5)
        num_lab_procedures = st.slider("Number of Lab Procedures", 0, 100, 41)
        num_procedures = st.slider("Number of Procedures", 0, 10, 0)
        num_medications = st.slider("Number of Medications", 1, 30, 15)
    
    with col2:
        st.write("**Medical History**")
        number_outpatient = st.slider("Number of Outpatient Visits", 0, 50, 0)
        number_emergency = st.slider("Number of Emergency Visits", 0, 50, 0)
        number_inpatient = st.slider("Number of Inpatient Visits", 0, 50, 0)
        number_diagnoses = st.slider("Number of Diagnoses", 1, 20, 9)
        
        st.write("**Lab Results**")
        max_glu_serum = st.selectbox("Max Glucose Serum", ["None", "Norm", ">200", ">300"])
        A1Cresult = st.selectbox("A1C Result", ["None", "Norm", ">7", ">8"])
        
        st.write("**Diagnoses**")
        diag_1 = st.text_input("Primary Diagnosis (ICD-9)", "250.01", help="Enter ICD-9 diagnosis code")
        diag_2 = st.text_input("Secondary Diagnosis (ICD-9)", "401.9", help="Enter ICD-9 diagnosis code")
        diag_3 = st.text_input("Tertiary Diagnosis (ICD-9)", "272.4", help="Enter ICD-9 diagnosis code")
    
    # Medications section
    st.subheader(" Medications")
    med_col1, med_col2, med_col3 = st.columns(3)
    
    with med_col1:
        metformin = st.selectbox("Metformin", ["No", "Up", "Down", "Steady"])
        repaglinide = st.selectbox("Repaglinide", ["No", "Up", "Down", "Steady"])
        nateglinide = st.selectbox("Nateglinide", ["No", "Up", "Down", "Steady"])
        chlorpropamide = st.selectbox("Chlorpropamide", ["No", "Up", "Down", "Steady"])
        glimepiride = st.selectbox("Glimepiride", ["No", "Up", "Down", "Steady"])
        glipizide = st.selectbox("Glipizide", ["No", "Up", "Down", "Steady"])
        glyburide = st.selectbox("Glyburide", ["No", "Up", "Down", "Steady"])
    
    with med_col2:
        tolbutamide = st.selectbox("Tolbutamide", ["No", "Up", "Down", "Steady"])
        pioglitazone = st.selectbox("Pioglitazone", ["No", "Up", "Down", "Steady"])
        rosiglitazone = st.selectbox("Rosiglitazone", ["No", "Up", "Down", "Steady"])
        acarbose = st.selectbox("Acarbose", ["No", "Up", "Down", "Steady"])
        miglitol = st.selectbox("Miglitol", ["No", "Up", "Down", "Steady"])
        tolazamide = st.selectbox("Tolazamide", ["No", "Up", "Down", "Steady"])
        insulin = st.selectbox("Insulin", ["No", "Up", "Down", "Steady"])
    
    with med_col3:
        glyburide_metformin = st.selectbox("Glyburide-Metformin", ["No", "Up", "Down", "Steady"])
        glipizide_metformin = st.selectbox("Glipizide-Metformin", ["No", "Up", "Down", "Steady"])
        glimepiride_pioglitazone = st.selectbox("Glimepiride-Pioglitazone", ["No", "Up", "Down", "Steady"])
        metformin_rosiglitazone = st.selectbox("Metformin-Rosiglitazone", ["No", "Up", "Down", "Steady"])
        metformin_pioglitazone = st.selectbox("Metformin-Pioglitazone", ["No", "Up", "Down", "Steady"])
        change = st.selectbox("Change in Medications", ["No", "Ch"])
        diabetesMed = st.selectbox("Diabetes Medication", ["Yes", "No"])
    
    # Create patient data dictionary
    patient_data = {
        'race': race,
        'gender': gender,
        'age': age,
        'admission_type_id': admission_type_id,
        'discharge_disposition_id': discharge_disposition_id,
        'admission_source_id': admission_source_id,
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'num_procedures': num_procedures,
        'num_medications': num_medications,
        'number_outpatient': number_outpatient,
        'number_emergency': number_emergency,
        'number_inpatient': number_inpatient,
        'number_diagnoses': number_diagnoses,
        'max_glu_serum': max_glu_serum,
        'A1Cresult': A1Cresult,
        'metformin': metformin,
        'repaglinide': repaglinide,
        'nateglinide': nateglinide,
        'chlorpropamide': chlorpropamide,
        'glimepiride': glimepiride,
        'glipizide': glipizide,
        'glyburide': glyburide,
        'tolbutamide': tolbutamide,
        'pioglitazone': pioglitazone,
        'rosiglitazone': rosiglitazone,
        'acarbose': acarbose,
        'miglitol': miglitol,
        'tolazamide': tolazamide,
        'insulin': insulin,
        'glyburide-metformin': glyburide_metformin,
        'glipizide-metformin': glipizide_metformin,
        'glimepiride-pioglitazone': glimepiride_pioglitazone,
        'metformin-rosiglitazone': metformin_rosiglitazone,
        'metformin-pioglitazone': metformin_pioglitazone,
        'change': change,
        'diabetesMed': diabetesMed,
        'diag_1': diag_1,
        'diag_2': diag_2,
        'diag_3': diag_3
    }
    
    return patient_data

def display_prediction_results(predictions, probabilities, ensemble_pred, ensemble_prob):
    """Display prediction results with visualizations"""
    
    # Main prediction result
    st.subheader(" Prediction Results")
    
    if ensemble_pred is not None:
        # Determine risk level and styling
        if ensemble_prob < 0.3:
            risk_level = "LOW RISK"
            risk_class = "low-risk"
            risk_color = "#4caf50"
        elif ensemble_prob < 0.6:
            risk_level = "MEDIUM RISK"
            risk_class = "medium-risk"
            risk_color = "#ff9800"
        else:
            risk_level = "HIGH RISK"
            risk_class = "high-risk"
            risk_color = "#f44336"
        
        result_text = "NO READMISSION" if ensemble_pred == 0 else "READMISSION LIKELY"
        
        # Display main result
        st.markdown(f"""
        <div class="prediction-box {risk_class}">
            <h2 style="text-align: center; color: {risk_color};">{result_text}</h2>
            <p style="text-align: center; font-size: 1.2rem;">Confidence: {ensemble_prob:.1%}</p>
            <p style="text-align: center; font-size: 1.1rem;">Risk Level: {risk_level}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Individual model predictions
    st.subheader(" Model Predictions")
    
    if predictions:
        # Create metrics for each model
        col1, col2, col3 = st.columns(3)
        
        for i, (model_name, pred) in enumerate(predictions.items()):
            if pred is not None and probabilities[model_name] is not None:
                prob = probabilities[model_name]['readmission']
                status = "READMISSION" if pred == 1 else "NO READMISSION"
                color = "#f44336" if pred == 1 else "#4caf50"
                
                with [col1, col2, col3][i % 3]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{model_name}</h4>
                        <p style="color: {color}; font-weight: bold;">{status}</p>
                        <p>Confidence: {prob:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Create visualization
    if probabilities:
        st.subheader(" Prediction Confidence Comparison")
        
        # Prepare data for plotting
        model_names = []
        confidences = []
        colors = []
        
        for model_name, prob_dict in probabilities.items():
            if prob_dict is not None:
                model_names.append(model_name)
                confidences.append(prob_dict['readmission'])
                colors.append("#f44336" if prob_dict['readmission'] > 0.5 else "#4caf50")
        
        if model_names:
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=model_names,
                    y=confidences,
                    marker_color=colors,
                    text=[f"{conf:.1%}" for conf in confidences],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Model Prediction Confidence",
                xaxis_title="Models",
                yaxis_title="Readmission Probability",
                yaxis=dict(range=[0, 1]),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.subheader(" Clinical Recommendations")
    
    if ensemble_pred == 1:
        st.markdown("""
        **High Risk Patient - Immediate Action Required:**
        - Schedule follow-up appointment within 1 week
        - Monitor blood glucose levels closely (3-4 times daily)
        - Review medication adherence and compliance
        - Consider additional support services (diabetes educator, nutritionist)
        - Implement enhanced discharge planning
        - Schedule home health visits if appropriate
        """)
    else:
        st.markdown("""
        **Low Risk Patient - Standard Care:**
        - Continue current treatment plan
        - Schedule routine follow-up (3-6 months)
        - Maintain healthy lifestyle habits
        - Regular blood glucose monitoring
        - Annual comprehensive diabetes assessment
        """)

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Diabetic Patient Readmission Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Predict Readmission", "About", "Model Information"])
    
    if page == "Predict Readmission":
        st.markdown("""
        This dashboard helps healthcare professionals predict the likelihood of 30-day readmission 
        for diabetic patients. Enter the patient's information below to get a prediction.
        """)
        
        # Load predictor
        with st.spinner("Loading prediction models..."):
            predictor = load_predictor()
        
        if predictor is None:
            st.error("Failed to load prediction models. Please check if the models are available.")
            return
        
        # Create patient form
        patient_data = create_patient_form()
        
        # Prediction button
        if st.button(" Predict Readmission Risk", type="primary", use_container_width=True):
            with st.spinner("Analyzing patient data..."):
                try:
                    # Make prediction
                    predictions, probabilities, ensemble_pred, ensemble_prob = predictor.predict_readmission(patient_data)
                    
                    # Display results
                    display_prediction_results(predictions, probabilities, ensemble_pred, ensemble_prob)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
    
    elif page == "About":
        st.subheader("About This System")
        st.markdown("""
        **Diabetic Patient Readmission Predictor**
        
        This machine learning system predicts the likelihood of 30-day readmission for diabetic patients 
        based on their clinical and demographic information.
        
        **Features:**
        - Predict Readmission risk using Multiple ML models
        - Risk level assessment (Low, Medium, High)
        - Clinical recommendations based on prediction
        
        **Data Source:**
        - Diabetic patient dataset with comprehensive clinical features
        - Includes demographics, medications, diagnoses, and hospital stay information
        """)
    
    elif page == "Model Information":
        st.subheader("Model Details")
        
        # Load model info
        try:
            X_train = joblib.load('models/X_train.pkl')
            feature_importance = joblib.load('models/feature_importance.pkl')
            
            st.write(f"**Training Data:** {X_train.shape[0]} patients, {X_train.shape[1]} features")
            
            # Feature importance visualization
            if 'XGBoost' in feature_importance:
                st.subheader("Top 10 Most Important Features")
                
                importance = feature_importance['XGBoost']
                feature_names = X_train.columns
                
                # Get top 10 features
                top_indices = np.argsort(importance)[-10:]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=[importance[i] for i in top_indices],
                        y=[feature_names[i] for i in top_indices],
                        orientation='h',
                        marker_color='#1f77b4'
                    )
                ])
                
                fig.update_layout(
                    title="Feature Importance",
                    xaxis_title="Importance Score",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading model information: {e}")

if __name__ == "__main__":
    main() 