import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load reference columns from training
X_train = joblib.load('models/X_train.pkl')
reference_columns = X_train.columns

class ReadmissionPredictor:
    def __init__(self):
        """Initialize the predictor with trained models"""
        self.models = {}
        self.preprocessors = {}
        self.load_models()
        
    def load_models(self):
        """Load trained models and preprocessors"""
        print("Loading trained models...")
        
        # Load models
        model_files = {
            'LogisticRegression': 'models/logisticregression_model.pkl',
            'XGBoost': 'models/xgboost_model.pkl'
        }
        
        for name, file_path in model_files.items():
            try:
                self.models[name] = joblib.load(file_path)
                print(f"✓ Loaded {name} model")
            except Exception as e:
                print(f"✗ Failed to load {name} model: {e}")
        
        # Load preprocessors
        try:
            self.preprocessors['imputer'] = joblib.load('models/imputer.pkl')
            self.preprocessors['label_encoders'] = joblib.load('models/label_encoders.pkl')
            self.preprocessors['scaler'] = joblib.load('models/scaler.pkl')
            print("✓ Loaded preprocessors")
        except Exception as e:
            print(f"✗ Failed to load preprocessors: {e}")
    
    def preprocess_new_data(self, data):
        """Preprocess new patient data using the same pipeline as training"""
        print("Preprocessing new patient data...")
        
        # Make a copy to avoid modifying original data
        processed_data = data.copy()
        
        # Apply the same cleaning steps as in training
        processed_data = self._clean_data(processed_data)
        
        # Apply label encoding for categorical variables
        if 'label_encoders' in self.preprocessors:
            processed_data = self._apply_label_encoding(processed_data)
        
        # Apply imputation
        if 'imputer' in self.preprocessors:
            processed_data = self._apply_imputation(processed_data)
        
        # Apply scaling
        if 'scaler' in self.preprocessors:
            processed_data = self._apply_scaling(processed_data)
        
        return processed_data
    
    def _clean_data(self, data):
        """Apply the same data cleaning as in training"""
        # Replace infinite values
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with median for numerical columns
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                median_val = data[col].median()
                if pd.isna(median_val):
                    median_val = 0
                data[col] = data[col].fillna(median_val)
        
        # Ensure all values are finite
        data = data.replace([np.inf, -np.inf], 0)
        
        # Clip extreme values to [-10, 10]
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                data[col] = data[col].clip(-10, 10)
        
        # Final fillna
        data = data.fillna(0)
        
        return data
    
    def _apply_label_encoding(self, data):
        """Apply label encoding to categorical variables"""
        label_encoders = self.preprocessors['label_encoders']
        
        for column, encoder in label_encoders.items():
            if column in data.columns:
                # Handle unseen categories by using the most frequent category
                unique_values = data[column].unique()
                for val in unique_values:
                    if val not in encoder.classes_:
                        data[column] = data[column].replace(val, encoder.classes_[0])
                
                data[column] = encoder.transform(data[column])
        
        return data
    
    def _apply_imputation(self, data):
        """Apply the same imputation as in training"""
        try:
            imputer = self.preprocessors['imputer']
            return imputer.transform(data)
        except Exception as e:
            print(f"Warning: Imputation failed ({e}), using data as-is")
            return data
    
    def _apply_scaling(self, data):
        """Apply the same scaling as in training"""
        try:
            scaler = self.preprocessors['scaler']
            return scaler.transform(data)
        except Exception as e:
            print(f"Warning: Scaling failed ({e}), using data as-is")
            return data
    
    def predict_single_patient(self, patient_data):
        """Predict readmission for a single patient"""
        print(f"\nMaking prediction for patient...")
        
        # Preprocess the data
        processed_data = self.preprocess_new_data(patient_data)
        
        predictions = {}
        probabilities = {}
        
        # Make predictions with each model
        for model_name, model in self.models.items():
            try:
                # Get prediction (0 = no readmission, 1 = readmission)
                pred = model.predict(processed_data)[0]
                prob = model.predict_proba(processed_data)[0]
                
                predictions[model_name] = pred
                probabilities[model_name] = {
                    'no_readmission': prob[0],
                    'readmission': prob[1]
                }
                
                print(f"{model_name}: {'READMISSION LIKELY' if pred == 1 else 'NO READMISSION'}")
                print(f"  Confidence: {prob[1]:.2%} chance of readmission")
                
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                predictions[model_name] = None
                probabilities[model_name] = None
        
        return predictions, probabilities
    
    def predict_batch(self, patient_data_batch):
        """Predict readmission for multiple patients"""
        print(f"\nMaking predictions for {len(patient_data_batch)} patients...")
        
        # Preprocess the data
        processed_data = self.preprocess_new_data(patient_data_batch)
        
        all_predictions = {}
        all_probabilities = {}
        
        # Make predictions with each model
        for model_name, model in self.models.items():
            try:
                # Get predictions
                preds = model.predict(processed_data)
                probs = model.predict_proba(processed_data)
                
                all_predictions[model_name] = preds
                all_probabilities[model_name] = probs[:, 1]  # Probability of readmission
                
                readmission_count = sum(preds)
                print(f"{model_name}: {readmission_count}/{len(preds)} patients predicted for readmission")
                
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                all_predictions[model_name] = None
                all_probabilities[model_name] = None
        
        return all_predictions, all_probabilities
    
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
                if isinstance(probabilities[model_name], dict):
                    avg_prob += probabilities[model_name]['readmission']
                else:
                    avg_prob += probabilities[model_name]
        
        ensemble_pred = 1 if sum(votes) > len(votes) / 2 else 0
        ensemble_prob = avg_prob / len(valid_models)
        
        return ensemble_pred, ensemble_prob

def create_sample_patient():
    """Create a sample patient data for testing"""
    # This is a sample patient with typical diabetic patient features
    sample_data = {
        'race': ['Caucasian'],
        'gender': ['Female'],
        'age': ['[50-60)'],
        'admission_type_id': [1],
        'discharge_disposition_id': [1],
        'admission_source_id': [7],
        'time_in_hospital': [5],
        'num_lab_procedures': [41],
        'num_procedures': [0],
        'num_medications': [15],
        'number_outpatient': [0],
        'number_emergency': [0],
        'number_inpatient': [0],
        'number_diagnoses': [9],
        'max_glu_serum': ['None'],
        'A1Cresult': ['None'],
        'metformin': ['No'],
        'repaglinide': ['No'],
        'nateglinide': ['No'],
        'chlorpropamide': ['No'],
        'glimepiride': ['No'],
        'acetohexamide': ['No'],
        'glipizide': ['No'],
        'glyburide': ['No'],
        'tolbutamide': ['No'],
        'pioglitazone': ['No'],
        'rosiglitazone': ['No'],
        'acarbose': ['No'],
        'miglitol': ['No'],
        'troglitazone': ['No'],
        'tolazamide': ['No'],
        'examide': ['No'],
        'citoglipton': ['No'],
        'insulin': ['No'],
        'glyburide-metformin': ['No'],
        'glipizide-metformin': ['No'],
        'glimepiride-pioglitazone': ['No'],
        'metformin-rosiglitazone': ['No'],
        'metformin-pioglitazone': ['No'],
        'change': ['No'],
        'diabetesMed': ['Yes'],
        'readmitted': ['NO']  # This will be predicted
    }
    
    return pd.DataFrame(sample_data)

def main():
    """Main function to demonstrate prediction"""
    print("=== Diabetic Patient Readmission Predictor ===\n")
    
    # Initialize predictor
    predictor = ReadmissionPredictor()
    
    if not predictor.models:
        print(" No models loaded. Please run model training first.")
        return
    
    # Create sample patient data
    sample_patient = create_sample_patient()
    print("Sample patient data:")
    print(sample_patient.head())
    
    # Make prediction
    predictions, probabilities = predictor.predict_single_patient(sample_patient)
    
    # Get ensemble prediction
    ensemble_pred, ensemble_prob = predictor.get_ensemble_prediction(predictions, probabilities)
    
    print(f"\n=== ENSEMBLE PREDICTION ===")
    if ensemble_pred is not None:
        result = "READMISSION LIKELY" if ensemble_pred == 1 else "NO READMISSION"
        print(f"Prediction: {result}")
        print(f"Confidence: {ensemble_prob:.2%} chance of readmission")
    else:
        print("Unable to make ensemble prediction")
    
    print(f"\n=== PREDICTION SUMMARY ===")
    for model_name, pred in predictions.items():
        if pred is not None:
            status = "READMISSION" if pred == 1 else "NO READMISSION"
            print(f"{model_name}: {status}")
        else:
            print(f"{model_name}: ERROR")

if __name__ == "__main__":
    main() 