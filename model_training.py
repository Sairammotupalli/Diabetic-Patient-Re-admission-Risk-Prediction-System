import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import wandb

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.feature_importance = {}
        self.wandb_enabled = False
        
        # Initialize W&B only if not in Docker or explicitly enabled
        wandb_mode = os.getenv('WANDB_MODE', 'run')
        if wandb_mode != 'disabled':
            try:
                wandb.init(
                    project="diabetic-readmission-prediction",
                    config={
                        "model_types": ["XGBoost", "CatBoost", "LightGBM"],
                        "dataset": "UCI Diabetes 130-US Hospitals",
                        "target": "30-day readmission prediction"
                    }
                )
                self.wandb_enabled = True
                print("✅ W&B initialized successfully")
            except Exception as e:
                print(f"⚠️ W&B initialization failed: {e}")
                print("Continuing without W&B tracking...")
                self.wandb_enabled = False
        else:
            print("ℹ️ W&B disabled in Docker environment")
            self.wandb_enabled = False
        
    def load_data(self):
        """Load preprocessed data"""
        print("Loading preprocessed data...")
        
        X_train = joblib.load('models/X_train.pkl')
        X_test = joblib.load('models/X_test.pkl')
        y_train = joblib.load('models/y_train.pkl')
        y_test = joblib.load('models/y_test.pkl')
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def create_pipelines(self):
        """Create ML pipelines for each model"""
        print("Creating ML pipelines...")
        
        # XGBoost Pipeline
        xgb_pipeline = Pipeline([
            ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
        ])
        
        # CatBoost Pipeline
        cat_pipeline = Pipeline([
            ('classifier', CatBoostClassifier(random_state=42, verbose=False))
        ])
        
        # LightGBM Pipeline
        lgb_pipeline = Pipeline([
            ('classifier', LGBMClassifier(random_state=42, verbose=-1))
        ])
        
        self.models = {
            'XGBoost': xgb_pipeline,
            'CatBoost': cat_pipeline,
            'LightGBM': lgb_pipeline
        }
        
        return self.models
    
    def define_hyperparameter_grids(self):
        """Define hyperparameter grids for GridSearchCV"""
        print("Defining hyperparameter grids...")
        
        param_grids = {
            'XGBoost': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 6],
                'classifier__learning_rate': [0.1, 0.2]
            },
            
            'CatBoost': {
                'classifier__iterations': [100, 200],
                'classifier__depth': [4, 6, 8],
                'classifier__learning_rate': [0.1, 0.2],
                'classifier__l2_leaf_reg': [1, 3, 5]
            },
            
            'LightGBM': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 6, 9],
                'classifier__learning_rate': [0.1, 0.2],
                'classifier__num_leaves': [31, 63, 127]
            }
        }
        
        return param_grids
    
    def _validate_and_clean_data(self, X, reference_columns=None):
        """Validate and clean data before training or evaluation. Optionally align columns to reference_columns."""
        X_clean = X.copy()
        
        # Replace infinite values
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with median
        for col in X_clean.columns:
            if X_clean[col].dtype in ['float64', 'int64']:
                median_val = X_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0
                X_clean[col] = X_clean[col].fillna(median_val)
        
        # Ensure all values are finite
        X_clean = X_clean.replace([np.inf, -np.inf], 0)
        
        # Clip extreme values to [-5, 5] for better numerical stability
        for col in X_clean.columns:
            if X_clean[col].dtype in ['float64', 'int64']:
                X_clean[col] = X_clean[col].clip(-5, 5)
        
        # Remove constant columns (zero variance)
        if reference_columns is None:
            stds = X_clean.std(axis=0)
            constant_cols = stds[stds == 0].index.tolist()
            if constant_cols:
                print(f"Dropping constant columns: {constant_cols}")
                X_clean = X_clean.drop(columns=constant_cols)
        
        # Final fillna (if any left)
        X_clean = X_clean.fillna(0)
        
        # If reference_columns is provided, align columns
        if reference_columns is not None:
            X_clean = X_clean.reindex(columns=reference_columns, fill_value=0)
        
        # Additional numerical stability checks
        X_clean = X_clean.astype(float)
        
        # Check for any remaining infinite or NaN values
        if X_clean.isin([np.inf, -np.inf, np.nan]).any().any():
            print("Warning: Found infinite/NaN values, replacing with 0")
            X_clean = X_clean.replace([np.inf, -np.inf, np.nan], 0)
        
        return X_clean

    def train_models(self, X_train, y_train):
        """Train models using GridSearchCV"""
        print("Training models with GridSearchCV...")
        
        # Validate and clean data before training
        X_train_clean = self._validate_and_clean_data(X_train)
        self._train_columns = X_train_clean.columns  # Save for test set alignment
        
        param_grids = self.define_hyperparameter_grids()
        
        for model_name, pipeline in self.models.items():
            print(f"\nTraining {model_name}...")
            
            try:
                # GridSearchCV with simplified settings
                grid_search = GridSearchCV(
                    pipeline,
                    param_grids[model_name],
                    cv=3,
                    scoring='roc_auc',
                    n_jobs=1,  # Use single job to avoid memory issues
                    verbose=1,
                    error_score=0.0  # Return 0 score if model fails
                )
                
                # Train the model
                grid_search.fit(X_train_clean, y_train)
                
                # Store the best model
                self.best_models[model_name] = grid_search.best_estimator_
                
                print(f"Best parameters for {model_name}: {grid_search.best_params_}")
                print(f"Best CV score for {model_name}: {grid_search.best_score_:.4f}")
                
                # Log to W&B
                if self.wandb_enabled:
                    try:
                        wandb.log({
                            f"{model_name}_best_params": grid_search.best_params_,
                            f"{model_name}_cv_score": grid_search.best_score_,
                            f"{model_name}_training_time": grid_search.refit_time_
                        })
                    except Exception as e:
                        print(f"⚠️ W&B logging failed for {model_name}: {e}")
                
                # Extract feature importance if available
                if hasattr(grid_search.best_estimator_.named_steps['classifier'], 'feature_importances_'):
                    importance_array = grid_search.best_estimator_.named_steps['classifier'].feature_importances_
                    print(f"✓ Extracted feature importance for {model_name}: {len(importance_array)} features")
                    self.feature_importance[model_name] = importance_array
                elif hasattr(grid_search.best_estimator_.named_steps['classifier'], 'coef_'):
                    coef_array = grid_search.best_estimator_.named_steps['classifier'].coef_[0]
                    print(f"✓ Extracted coefficients for {model_name}: {len(coef_array)} features")
                    self.feature_importance[model_name] = np.abs(coef_array)
                else:
                    print(f"⚠️ No feature importance available for {model_name}")
                    
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                print(f"Skipping {model_name} and continuing with other models...")
                continue

    def evaluate_models(self, X_test, y_test):
        """Evaluate all models on test set"""
        print("\nEvaluating models on test set...")
        
        # Clean and align test set columns to training columns
        X_test_clean = self._validate_and_clean_data(X_test, reference_columns=getattr(self, '_train_columns', None))
        
        results = {}
        
        for model_name, model in self.best_models.items():
            print(f"\nEvaluating {model_name}...")
            
            # Predictions
            y_pred = model.predict(X_test_clean)
            y_pred_proba = model.predict_proba(X_test_clean)[:, 1]
            
            # Metrics
            accuracy = model.score(X_test_clean, y_test)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Classification report
            report = classification_report(y_test, y_pred)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            results[model_name] = {
                'accuracy': accuracy,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': report,
                'confusion_matrix': cm
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"AUC: {auc:.4f}")
            print("Classification Report:")
            print(report)
            
            # Log to W&B
            if self.wandb_enabled:
                try:
                    wandb.log({
                        f"{model_name}_test_accuracy": accuracy,
                        f"{model_name}_test_auc": auc,
                        f"{model_name}_classification_report": report
                    })
                except Exception as e:
                    print(f"⚠️ W&B logging failed for {model_name}: {e}")
        
        return results
    
    def save_models(self):
        """Save trained models"""
        print("\nSaving models...")
        
        os.makedirs('models', exist_ok=True)
        
        for model_name, model in self.best_models.items():
            model_path = f'models/{model_name.lower()}_model.pkl'
            joblib.dump(model, model_path)
            print(f"Saved {model_name} model")
            
            # Log model to W&B
            if self.wandb_enabled:
                try:
                    wandb.save(model_path)
                    wandb.log_artifact(model_path, name=f"{model_name.lower()}_model", type="model")
                except Exception as e:
                    print(f"⚠️ W&B model logging failed for {model_name}: {e}")
        
        # Save feature importance
        feature_path = 'models/feature_importance.pkl'
        joblib.dump(self.feature_importance, feature_path)
        print("Saved feature importance")
        
        # Log feature importance to W&B
        if self.wandb_enabled:
            try:
                wandb.save(feature_path)
                wandb.log_artifact(feature_path, name="feature_importance", type="data")
            except Exception as e:
                print(f"⚠️ W&B feature importance logging failed: {e}")
    
    def plot_results(self, results, X_test, y_test):
        """Plot model comparison results"""
        print("\nCreating visualization plots...")
        
        os.makedirs('plots', exist_ok=True)
        
        # ROC Curves
        plt.figure(figsize=(10, 6))
        for model_name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {result["auc"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Confusion Matrices
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, (model_name, result) in enumerate(results.items()):
            sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', 
                       cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{model_name} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('plots/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature Importance (for tree-based models)
        tree_models = ['XGBoost', 'CatBoost', 'LightGBM']
        print(f"\nFeature importance status: {list(self.feature_importance.keys())}")
        
        if any(model in self.feature_importance for model in tree_models):
            fig, axes = plt.subplots(1, 1, figsize=(12, 8))
            
            # Use training columns (which the models were trained on)
            feature_names = getattr(self, '_train_columns', X_test.columns).tolist()
            print(f"Feature names count: {len(feature_names)}")
            
            # Combine feature importance from all tree models
            all_importance = np.zeros(len(feature_names))
            
            for model_name in tree_models:
                if model_name in self.feature_importance:
                    importance_array = self.feature_importance[model_name]
                    print(f"Processing {model_name}: {len(importance_array)} importance values")
                    # Add importance values (ensure same length)
                    if len(importance_array) == len(feature_names):
                        all_importance += importance_array
                        print(f"✓ Added {model_name} importance")
                    else:
                        print(f"⚠️ Length mismatch for {model_name}: {len(importance_array)} vs {len(feature_names)}")
                        # Try to align arrays if possible
                        min_len = min(len(importance_array), len(feature_names))
                        all_importance[:min_len] += importance_array[:min_len]
                        print(f"✓ Added {model_name} importance (aligned to {min_len} features)")
            
            # Check if we have any importance values
            if np.sum(all_importance) > 0:
                # Get top 20 features
                top_indices = np.argsort(all_importance)[-20:]
                top_features = [feature_names[i] for i in top_indices]
                top_importances = all_importance[top_indices]
                
                # Create bar plot
                axes.barh(range(len(top_features)), top_importances)
                axes.set_yticks(range(len(top_features)))
                axes.set_yticklabels(top_features)
                axes.set_xlabel('Feature Importance')
                axes.set_title('Top 20 Feature Importance (Combined Tree Models)')
                axes.invert_yaxis()
                
                plt.tight_layout()
                plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("✓ Feature importance plot saved")
                
                # Log to W&B
                if self.wandb_enabled:
                    try:
                        wandb.log({"feature_importance": wandb.Image('plots/feature_importance.png')})
                    except Exception as e:
                        print(f"⚠️ W&B feature importance logging failed: {e}")
            else:
                print("⚠️ No feature importance values available for plotting")
        else:
            print("⚠️ No tree-based models with feature importance found")
        
        print("Plots saved to plots/ directory")
        
        # Log plots to W&B
        if self.wandb_enabled:
            try:
                wandb.save('plots/roc_curves.png')
                wandb.save('plots/confusion_matrices.png')
                wandb.save('plots/feature_importance.png')
                wandb.log({"roc_curves": wandb.Image('plots/roc_curves.png')})
                wandb.log({"confusion_matrices": wandb.Image('plots/confusion_matrices.png')})
                wandb.log({"feature_importance": wandb.Image('plots/feature_importance.png')})
            except Exception as e:
                print(f"⚠️ W&B plot logging failed: {e}")
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        print("Starting model training pipeline...")
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Create pipelines
        self.create_pipelines()
        
        # Train models
        self.train_models(X_train, y_train)
        
        # Evaluate models
        results = self.evaluate_models(X_test, y_test)
        
        # Save models
        self.save_models()
        
        # Create plots
        self.plot_results(results, X_test, y_test)
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['auc'])
        print(f"\nBest model: {best_model[0]} with AUC: {best_model[1]['auc']:.4f}")
        
        # Log best model to W&B
        if self.wandb_enabled:
            try:
                wandb.log({
                    "best_model": best_model[0],
                    "best_auc": best_model[1]['auc'],
                    "best_accuracy": best_model[1]['accuracy']
                })
                wandb.finish()
                print("✅ W&B run completed successfully")
            except Exception as e:
                print(f"⚠️ W&B finalization failed: {e}")
        
        return results

if __name__ == "__main__":
    trainer = ModelTrainer()
    results = trainer.run_training_pipeline() 