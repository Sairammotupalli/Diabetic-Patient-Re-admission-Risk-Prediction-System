from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import our predictor
from predict_readmission import ReadmissionPredictor

app = FastAPI(
    title="Diabetic Patient Readmission Prediction API",
    description="API for predicting 30-day readmission risk for diabetic patients",
    version="1.0.0"
)

# Initialize predictor
predictor = ReadmissionPredictor()

class PatientData(BaseModel):
    """Schema for patient data input"""
    race: str = "Caucasian"
    gender: str = "Female"
    age: str = "[50-60)"
    admission_type_id: int = 1
    discharge_disposition_id: int = 1
    admission_source_id: int = 7
    time_in_hospital: int = 5
    num_lab_procedures: int = 41
    num_procedures: int = 0
    num_medications: int = 15
    number_outpatient: int = 0
    number_emergency: int = 0
    number_inpatient: int = 0
    number_diagnoses: int = 9
    max_glu_serum: str = "None"
    A1Cresult: str = "None"
    metformin: str = "No"
    repaglinide: str = "No"
    nateglinide: str = "No"
    chlorpropamide: str = "No"
    glimepiride: str = "No"
    acetohexamide: str = "No"
    glipizide: str = "No"
    glyburide: str = "No"
    tolbutamide: str = "No"
    pioglitazone: str = "No"
    rosiglitazone: str = "No"
    acarbose: str = "No"
    miglitol: str = "No"
    troglitazone: str = "No"
    tolazamide: str = "No"
    examide: str = "No"
    citoglipton: str = "No"
    insulin: str = "No"
    glyburide_metformin: str = "No"
    glipizide_metformin: str = "No"
    glimepiride_pioglitazone: str = "No"
    metformin_rosiglitazone: str = "No"
    metformin_pioglitazone: str = "No"
    change: str = "No"
    diabetesMed: str = "Yes"

class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    prediction: str  # "READMISSION" or "NO READMISSION"
    confidence: float  # Probability of readmission
    ensemble_prediction: str
    ensemble_confidence: float
    model_predictions: Dict[str, str]
    model_confidences: Dict[str, float]

class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request"""
    patients: List[PatientData]

class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response"""
    predictions: List[Dict]
    summary: Dict[str, int]

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Diabetic Patient Readmission Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "Single patient prediction",
            "/predict_batch": "Batch prediction for multiple patients",
            "/health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(predictor.models) > 0,
        "available_models": list(predictor.models.keys())
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_single_patient(patient: PatientData):
    """Predict readmission for a single patient"""
    try:
        # Convert Pydantic model to DataFrame
        patient_dict = patient.dict()
        
        # Fix column names to match training data
        patient_dict['glyburide-metformin'] = patient_dict.pop('glyburide_metformin')
        patient_dict['glipizide-metformin'] = patient_dict.pop('glipizide_metformin')
        patient_dict['glimepiride-pioglitazone'] = patient_dict.pop('glimepiride_pioglitazone')
        patient_dict['metformin-rosiglitazone'] = patient_dict.pop('metformin_rosiglitazone')
        patient_dict['metformin-pioglitazone'] = patient_dict.pop('metformin_pioglitazone')
        
        patient_df = pd.DataFrame([patient_dict])
        
        # Make prediction
        predictions, probabilities = predictor.predict_single_patient(patient_df)
        
        # Get ensemble prediction
        ensemble_pred, ensemble_prob = predictor.get_ensemble_prediction(predictions, probabilities)
        
        # Prepare response
        model_predictions = {}
        model_confidences = {}
        
        for model_name, pred in predictions.items():
            if pred is not None:
                model_predictions[model_name] = "READMISSION" if pred == 1 else "NO READMISSION"
                if probabilities[model_name] is not None:
                    model_confidences[model_name] = probabilities[model_name]['readmission']
                else:
                    model_confidences[model_name] = 0.0
            else:
                model_predictions[model_name] = "ERROR"
                model_confidences[model_name] = 0.0
        
        # Use best model prediction as primary (XGBoost was best in previous runs)
        best_model = "XGBoost"
        primary_prediction = model_predictions.get(best_model, "NO READMISSION")
        primary_confidence = model_confidences.get(best_model, 0.0)
        
        return PredictionResponse(
            prediction=primary_prediction,
            confidence=primary_confidence,
            ensemble_prediction="READMISSION" if ensemble_pred == 1 else "NO READMISSION",
            ensemble_confidence=ensemble_prob if ensemble_prob is not None else 0.0,
            model_predictions=model_predictions,
            model_confidences=model_confidences
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch_patients(request: BatchPredictionRequest):
    """Predict readmission for multiple patients"""
    try:
        # Convert patients to DataFrame
        patients_data = []
        for patient in request.patients:
            patient_dict = patient.dict()
            
            # Fix column names
            patient_dict['glyburide-metformin'] = patient_dict.pop('glyburide_metformin')
            patient_dict['glipizide-metformin'] = patient_dict.pop('glipizide_metformin')
            patient_dict['glimepiride-pioglitazone'] = patient_dict.pop('glimepiride_pioglitazone')
            patient_dict['metformin-rosiglitazone'] = patient_dict.pop('metformin_rosiglitazone')
            patient_dict['metformin-pioglitazone'] = patient_dict.pop('metformin_pioglitazone')
            
            patients_data.append(patient_dict)
        
        patients_df = pd.DataFrame(patients_data)
        
        # Make batch predictions
        all_predictions, all_probabilities = predictor.predict_batch(patients_df)
        
        # Prepare response
        predictions_list = []
        summary = {"total_patients": len(patients_data), "readmission_count": 0}
        
        for i in range(len(patients_data)):
            patient_prediction = {
                "patient_id": i,
                "predictions": {},
                "ensemble_prediction": "NO READMISSION",
                "ensemble_confidence": 0.0
            }
            
            # Collect predictions from each model
            valid_predictions = []
            valid_probabilities = []
            
            for model_name, preds in all_predictions.items():
                if preds is not None and i < len(preds):
                    pred = preds[i]
                    prob = all_probabilities[model_name][i] if all_probabilities[model_name] is not None else 0.0
                    
                    patient_prediction["predictions"][model_name] = {
                        "prediction": "READMISSION" if pred == 1 else "NO READMISSION",
                        "confidence": prob
                    }
                    
                    valid_predictions.append(pred)
                    valid_probabilities.append(prob)
            
            # Calculate ensemble prediction
            if valid_predictions:
                ensemble_pred = 1 if sum(valid_predictions) > len(valid_predictions) / 2 else 0
                ensemble_prob = sum(valid_probabilities) / len(valid_probabilities)
                
                patient_prediction["ensemble_prediction"] = "READMISSION" if ensemble_pred == 1 else "NO READMISSION"
                patient_prediction["ensemble_confidence"] = ensemble_prob
                
                if ensemble_pred == 1:
                    summary["readmission_count"] += 1
            
            predictions_list.append(patient_prediction)
        
        return BatchPredictionResponse(
            predictions=predictions_list,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """Get information about loaded models"""
    return {
        "models_loaded": len(predictor.models),
        "available_models": list(predictor.models.keys()),
        "preprocessors_loaded": len(predictor.preprocessors),
        "available_preprocessors": list(predictor.preprocessors.keys())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 