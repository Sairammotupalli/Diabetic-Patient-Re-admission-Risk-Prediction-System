import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

class SinglePatientPredictor:
    def __init__(self):
        """Initialize the single patient predictor"""
        self.models = {}
        self.label_encoders = {}
        self.scaler = None
        self.reference_columns = None
        self.load_models_and_preprocessors()
    
    def load_models_and_preprocessors(self):
        """Load trained models and preprocessors"""
        print("Loading models and preprocessors...")
        
        try:
            # Load models
            self.models['XGBoost'] = joblib.load('models/xgboost_model.pkl')
            self.models['LogisticRegression'] = joblib.load('models/logisticregression_model.pkl')
            
            # Load preprocessors
            self.label_encoders = joblib.load('models/label_encoders.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            
            # Load reference columns from training data
            X_train = joblib.load('models/X_train.pkl')
            self.reference_columns = X_train.columns.tolist()
            
            print("✓ All models and preprocessors loaded successfully")
            print(f"✓ Reference columns: {len(self.reference_columns)} features")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def create_sample_patient(self):
        """Create a sample patient with all required features"""
        sample_data = {
            # Basic patient info
            'race': 'Caucasian',
            'gender': 'Female', 
            'age': '[50-60)',
            'admission_type_id': 1,
            'discharge_disposition_id': 1,
            'admission_source_id': 7,
            'time_in_hospital': 5,
            'num_lab_procedures': 41,
            'num_procedures': 0,
            'num_medications': 15,
            'number_outpatient': 0,
            'number_emergency': 0,
            'number_inpatient': 0,
            'number_diagnoses': 9,
            'max_glu_serum': 'None',
            'A1Cresult': 'None',
            
            # Medications (excluding constant columns that were dropped during training)
            'metformin': 'No',
            'repaglinide': 'No',
            'nateglinide': 'No',
            'chlorpropamide': 'No',
            'glimepiride': 'No',
            'glipizide': 'No',
            'glyburide': 'No',
            'tolbutamide': 'No',
            'pioglitazone': 'No',
            'rosiglitazone': 'No',
            'acarbose': 'No',
            'miglitol': 'No',
            'tolazamide': 'No',
            'insulin': 'No',
            'glyburide-metformin': 'No',
            'glipizide-metformin': 'No',
            'glimepiride-pioglitazone': 'No',
            'metformin-rosiglitazone': 'No',
            'metformin-pioglitazone': 'No',
            'change': 'No',
            'diabetesMed': 'Yes',
            
            # Diagnoses (these will be engineered into features)
            'diag_1': '250.01',
            'diag_2': '401.9',
            'diag_3': '272.4'
        }
        
        return sample_data
    
    def engineer_features(self, patient_data):
        """Apply the same feature engineering as in training"""
        df = pd.DataFrame([patient_data])
        
        # Create medication features
        medication_cols = [col for col in df.columns if any(med in col.lower() for med in 
                          ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 
                           'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 
                           'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 
                           'miglitol', 'troglitazone', 'tolazamide', 'examide', 
                           'citoglipton', 'insulin'])]
        
        # Create consolidated medication features
        df['insulin_usage'] = (df['insulin'] == 'Up').astype(int)
        df['oral_medications'] = df[medication_cols].apply(lambda x: (x == 'Up').sum(), axis=1)
        df['total_medications'] = df[medication_cols].apply(lambda x: (x != 'No').sum(), axis=1)
        
        # Create diagnosis features
        diagnosis_cols = ['diag_1', 'diag_2', 'diag_3']
        for col in diagnosis_cols:
            if col in df.columns:
                # Create binary features for common diagnosis categories
                df[f'{col}_diabetes'] = df[col].str.contains('250', na=False).astype(int)
                df[f'{col}_circulatory'] = df[col].str.contains('^[4-5]', na=False).astype(int)
                df[f'{col}_respiratory'] = df[col].str.contains('^[4-5]', na=False).astype(int)
        
        return df
    
    def encode_categorical_variables(self, df):
        """Apply label encoding to categorical variables"""
        for col in df.columns:
            if col in self.label_encoders:
                try:
                    # Handle unseen categories
                    unique_values = df[col].unique()
                    for val in unique_values:
                        if val not in self.label_encoders[col].classes_:
                            df[col] = df[col].replace(val, self.label_encoders[col].classes_[0])
                    
                    df[col] = self.label_encoders[col].transform(df[col])
                except Exception as e:
                    print(f"Warning: Encoding failed for {col}: {e}")
                    # Use default value
                    df[col] = 0
        
        return df
    
    def align_columns(self, df):
        """Align columns to match training data, but only keep columns present in both"""
        # Only keep columns that are in both the DataFrame and the reference columns
        valid_cols = [col for col in self.reference_columns if col in df.columns]
        df = df.reindex(columns=valid_cols, fill_value=0)
        return df
    
    def preprocess_patient_data(self, patient_data):
        """Complete preprocessing pipeline for single patient"""
        print("Preprocessing patient data...")
        
        # Engineer features
        df = self.engineer_features(patient_data)
        
        # Encode categorical variables
        df = self.encode_categorical_variables(df)
        
        # Align columns to training data
        df = self.align_columns(df)
        
        # Clean data
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        # Apply scaling if available
        if self.scaler is not None:
            try:
                df = pd.DataFrame(self.scaler.transform(df), columns=df.columns)
            except Exception as e:
                print(f"Warning: Scaling failed: {e}")
        
        print(f"✓ Preprocessing complete. Final shape: {df.shape}")
        return df
    
    def predict_readmission(self, patient_data):
        """Predict readmission for a single patient"""
        print("\n" + "="*50)
        print("DIABETIC PATIENT READMISSION PREDICTION")
        print("="*50)
        
        # Preprocess the patient data
        processed_data = self.preprocess_patient_data(patient_data)
        
        predictions = {}
        probabilities = {}
        
        print("\nMaking predictions with trained models...")
        
        # Make predictions with each model
        for model_name, model in self.models.items():
            try:
                # Get prediction (0 = no readmission, 1 = readmission)
                pred = model.predict(processed_data)[0]
                prob = model.predict_proba(processed_data)[0]
                
                # Debug information for Logistic Regression
                if model_name == 'LogisticRegression':
                    print(f"\nDebug info for {model_name}:")
                    print(f"  Raw prediction: {pred}")
                    print(f"  Raw probabilities: {prob}")
                    print(f"  Probability sum: {sum(prob)}")
                    
                    # Check if probabilities are valid
                    if sum(prob) < 0.99 or sum(prob) > 1.01:
                        print(f"  Warning: Probabilities don't sum to 1: {sum(prob)}")
                    
                    # If probabilities are all zeros or invalid, try alternative approach
                    if all(p == 0 for p in prob) or sum(prob) == 0:
                        print(f"  Attempting alternative prediction method...")
                        # Try direct prediction without probabilities
                        pred = model.predict(processed_data)[0]
                        # Use a simple heuristic for probability
                        prob = [0.1, 0.9] if pred == 1 else [0.9, 0.1]
                        print(f"  Using fallback probabilities: {prob}")
                
                predictions[model_name] = pred
                probabilities[model_name] = {
                    'no_readmission': prob[0],
                    'readmission': prob[1]
                }
                
                status = " NO READMISSION" if pred == 0 else " READMISSION LIKELY"
                print(f"{model_name:15} | {status:20} | Confidence: {prob[1]:.1%}")
                
            except Exception as e:
                print(f"{model_name:15} | ERROR: {e}")
                print(f"  Attempting fallback prediction...")
                
                # Fallback: use a simple prediction based on key features
                try:
                    # Simple heuristic based on key features
                    key_features = ['time_in_hospital', 'num_medications', 'number_diagnoses']
                    risk_score = 0
                    
                    for feature in key_features:
                        if feature in processed_data.columns:
                            value = processed_data[feature].iloc[0]
                            if feature == 'time_in_hospital' and value > 7:
                                risk_score += 1
                            elif feature == 'num_medications' and value > 10:
                                risk_score += 1
                            elif feature == 'number_diagnoses' and value > 5:
                                risk_score += 1
                    
                    # Simple prediction based on risk score
                    pred = 1 if risk_score >= 2 else 0
                    prob = [0.2, 0.8] if pred == 1 else [0.8, 0.2]
                    
                    predictions[model_name] = pred
                    probabilities[model_name] = {
                        'no_readmission': prob[0],
                        'readmission': prob[1]
                    }
                    
                    status = " NO READMISSION" if pred == 0 else " READMISSION LIKELY"
                    print(f"{model_name:15} | {status:20} | Confidence: {prob[1]:.1%} (FALLBACK)")
                    
                except Exception as fallback_error:
                    print(f"  Fallback also failed: {fallback_error}")
                    predictions[model_name] = None
                    probabilities[model_name] = None
        
        # Get ensemble prediction
        ensemble_pred, ensemble_prob = self.get_ensemble_prediction(predictions, probabilities)
        
        # Display results
        self.display_results(predictions, probabilities, ensemble_pred, ensemble_prob)
        
        return predictions, probabilities, ensemble_pred, ensemble_prob
    
    def get_ensemble_prediction(self, predictions, probabilities):
        """Get ensemble prediction using voting"""
        valid_models = [name for name, pred in predictions.items() if pred is not None]
        
        if not valid_models:
            return None, None
        
        # Majority voting for classification
        votes = []
        avg_prob = 0
        
        for model_name in valid_models:
            votes.append(predictions[model_name])
            if probabilities[model_name] is not None:
                avg_prob += probabilities[model_name]['readmission']
        
        ensemble_pred = 1 if sum(votes) > len(votes) / 2 else 0
        ensemble_prob = avg_prob / len(valid_models)
        
        return ensemble_pred, ensemble_prob
    
    def display_results(self, predictions, predictions_probs, ensemble_pred, ensemble_prob):
        """Display prediction results in a formatted way"""
        print("\n" + "="*50)
        print("PREDICTION RESULTS")
        print("="*50)
        
        # Individual model results
        print("\n Individual Model Predictions:")
        for model_name, pred in predictions.items():
            if pred is not None:
                status = " READMISSION" if pred == 1 else "NO READMISSION"
                prob = predictions_probs[model_name]['readmission']
                print(f"  {model_name:15} | {status:20} | {prob:.1%}")
            else:
                print(f"  {model_name:15} |  ERROR")
        
        # Ensemble result
        print(f"\n ENSEMBLE PREDICTION:")
        if ensemble_pred is not None:
            result = "READMISSION LIKELY" if ensemble_pred == 1 else " NO READMISSION"
            print(f"  {result}")
            print(f"  Confidence: {ensemble_prob:.1%} chance of readmission")
            
            # Risk level interpretation
            if ensemble_prob < 0.3:
                risk_level = " LOW RISK"
            elif ensemble_prob < 0.6:
                risk_level = " MEDIUM RISK"
            else:
                risk_level = " HIGH RISK"
            
            print(f"  Risk Level: {risk_level}")
        else:
            print(" Unable to make ensemble prediction")
        
        # Recommendations
        print(f"\n RECOMMENDATIONS:")
        if ensemble_pred == 1:
            print("  • Schedule follow-up appointment within 1 week")
            print("  • Monitor blood glucose levels closely")
            print("  • Review medication adherence")
            print("  • Consider additional support services")
        else:
            print("  • Continue current treatment plan")
            print("  • Schedule routine follow-up")
            print("  • Maintain healthy lifestyle habits")
        
        print("\n" + "="*50)

def main():
    """Main function to demonstrate single patient prediction"""
    try:
        # Initialize predictor
        predictor = SinglePatientPredictor()
        
        # Create sample patient data
        sample_patient = predictor.create_sample_patient()
        
        print("Sample patient data:")
        for key, value in sample_patient.items():
            print(f"  {key}: {value}")
        
        # Make prediction
        predictions, probabilities, ensemble_pred, ensemble_prob = predictor.predict_readmission(sample_patient)
        
        return predictions, probabilities, ensemble_pred, ensemble_prob
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None

if __name__ == "__main__":
    main() 