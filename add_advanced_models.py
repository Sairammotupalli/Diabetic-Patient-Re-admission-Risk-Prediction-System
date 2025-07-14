#!/usr/bin/env python3
"""
Add Advanced Models to Diabetic Readmission Prediction
=====================================================

This script adds Neural Network model to the existing trained models
with robust configurations to avoid numerical issues.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import warnings
import traceback
warnings.filterwarnings('ignore')

def add_advanced_models():
    """Add Neural Network model with robust configurations"""
    print("Adding Neural Network model...")
    
    try:
        # Load existing data
        print("Loading data...")
        X_train = joblib.load('models/X_train.pkl')
        X_test = joblib.load('models/X_test.pkl')
        y_train = joblib.load('models/y_train.pkl')
        y_test = joblib.load('models/y_test.pkl')
        print(f"✓ Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Clean and validate data
        def clean_data(X):
            print(f"Cleaning data with shape: {X.shape}")
            X_clean = X.copy()
            X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
            X_clean = X_clean.fillna(0)
            X_clean = X_clean.clip(-3, 3)  # More conservative clipping
            X_clean = X_clean.astype(float)
            print(f"✓ Data cleaned - Shape: {X_clean.shape}")
            return X_clean
        
        X_train_clean = clean_data(X_train)
        X_test_clean = clean_data(X_test)
        
        # Load existing models
        existing_models = {}
        try:
            existing_models['XGBoost'] = joblib.load('models/xgboost_model.pkl')
            existing_models['CatBoost'] = joblib.load('models/catboost_model.pkl')
            existing_models['LightGBM'] = joblib.load('models/lightgbm_model.pkl')
            print("✓ Loaded existing models")
        except Exception as e:
            print(f"⚠️ Could not load existing models: {e}")
        
        # Create advanced model pipelines
        advanced_models = {}
        
        # Neural Network with very conservative settings
        print("\nTraining Neural Network...")
        try:
            nn_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', MLPClassifier(
                    random_state=42,
                    max_iter=200,  # Reduced iterations
                    early_stopping=True,
                    hidden_layer_sizes=(20,),  # Smaller network
                    alpha=0.01,  # Stronger regularization
                    learning_rate_init=0.001,
                    solver='adam',
                    activation='relu'
                ))
            ])
            
            # Simple hyperparameter grid
            nn_param_grid = {
                'classifier__hidden_layer_sizes': [(20,), (30,)],
                'classifier__alpha': [0.01, 0.1]
            }
            
            print("Starting GridSearchCV for Neural Network...")
            nn_grid = GridSearchCV(
                nn_pipeline, nn_param_grid, cv=3, scoring='roc_auc',
                n_jobs=1, verbose=1, error_score=0.0
            )
            
            nn_grid.fit(X_train_clean, y_train)
            advanced_models['NeuralNetwork'] = nn_grid.best_estimator_
            
            # Evaluate
            y_pred_nn = nn_grid.predict(X_test_clean)
            y_prob_nn = nn_grid.predict_proba(X_test_clean)[:, 1]
            auc_nn = roc_auc_score(y_test, y_prob_nn)
            
            print(f"✓ Neural Network trained successfully")
            print(f"  Best parameters: {nn_grid.best_params_}")
            print(f"  CV Score: {nn_grid.best_score_:.4f}")
            print(f"  Test AUC: {auc_nn:.4f}")
            
        except Exception as e:
            print(f"❌ Neural Network training failed: {e}")
            print(f"Full error: {traceback.format_exc()}")
        
        # Save advanced models
        if advanced_models:
            print("\nSaving advanced models...")
            for model_name, model in advanced_models.items():
                try:
                    model_path = f'models/{model_name.lower()}_model.pkl'
                    joblib.dump(model, model_path)
                    print(f"✓ Saved {model_name} model to {model_path}")
                except Exception as e:
                    print(f"❌ Failed to save {model_name}: {e}")
        else:
            print("⚠️ No advanced models to save")
        
        # Update feature importance
        feature_importance = {}
        try:
            feature_importance = joblib.load('models/feature_importance.pkl')
            print("✓ Loaded existing feature importance")
        except Exception as e:
            print(f"⚠️ Could not load feature importance: {e}")
        
        # Add feature importance for new models if available
        for model_name, model in advanced_models.items():
            try:
                if hasattr(model.named_steps['classifier'], 'feature_importances_'):
                    feature_importance[model_name] = model.named_steps['classifier'].feature_importances_
                elif hasattr(model.named_steps['classifier'], 'coef_'):
                    feature_importance[model_name] = np.abs(model.named_steps['classifier'].coef_[0])
                print(f"✓ Added feature importance for {model_name}")
            except Exception as e:
                print(f"⚠️ Could not get feature importance for {model_name}: {e}")
        
        try:
            joblib.dump(feature_importance, 'models/feature_importance.pkl')
            print("✓ Updated feature importance")
        except Exception as e:
            print(f"❌ Failed to save feature importance: {e}")
        
        # Print summary
        print("\n" + "="*50)
        print("ADVANCED MODELS SUMMARY")
        print("="*50)
        
        all_models = {**existing_models, **advanced_models}
        
        for model_name, model in all_models.items():
            try:
                y_pred = model.predict(X_test_clean)
                y_prob = model.predict_proba(X_test_clean)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
                accuracy = model.score(X_test_clean, y_test)
                
                print(f"{model_name}:")
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  AUC: {auc:.4f}")
            except Exception as e:
                print(f"{model_name}: Error evaluating - {e}")
        
        print(f"\n✓ Total models available: {len(all_models)}")
        print("Models: " + ", ".join(all_models.keys()))
        
    except Exception as e:
        print(f"❌ Script failed with error: {e}")
        print(f"Full error: {traceback.format_exc()}")

if __name__ == "__main__":
    add_advanced_models() 