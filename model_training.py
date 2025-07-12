import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.feature_importance = {}
        
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
        
        # Logistic Regression Pipeline with robust settings
        lr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000, solver='liblinear', C=1.0))
        ])
        
        # XGBoost Pipeline
        xgb_pipeline = Pipeline([
            ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
        ])
        
        self.models = {
            'LogisticRegression': lr_pipeline,
            'XGBoost': xgb_pipeline
        }
        
        return self.models
    
    def define_hyperparameter_grids(self):
        """Define hyperparameter grids for GridSearchCV"""
        print("Defining hyperparameter grids...")
        
        param_grids = {
            'LogisticRegression': {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear']
            },

            'XGBoost': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 6],
                'classifier__learning_rate': [0.1, 0.2]
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
        
        # Clip extreme values to [-10, 10]
        for col in X_clean.columns:
            if X_clean[col].dtype in ['float64', 'int64']:
                X_clean[col] = X_clean[col].clip(-10, 10)
        
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
                # GridSearchCV
                grid_search = GridSearchCV(
                    pipeline,
                    param_grids[model_name],
                    cv=3,  # Reduced from 5 to speed up and reduce numerical issues
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1
                )
                
                # Train the model
                grid_search.fit(X_train_clean, y_train)
                
                # Store the best model
                self.best_models[model_name] = grid_search.best_estimator_
                
                print(f"Best parameters for {model_name}: {grid_search.best_params_}")
                print(f"Best CV score for {model_name}: {grid_search.best_score_:.4f}")
                
                # Extract feature importance if available
                if hasattr(grid_search.best_estimator_.named_steps['classifier'], 'feature_importances_'):
                    self.feature_importance[model_name] = grid_search.best_estimator_.named_steps['classifier'].feature_importances_
                elif hasattr(grid_search.best_estimator_.named_steps['classifier'], 'coef_'):
                    self.feature_importance[model_name] = np.abs(grid_search.best_estimator_.named_steps['classifier'].coef_[0])
                    
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
        
        return results
    
    def save_models(self):
        """Save trained models"""
        print("\nSaving models...")
        
        os.makedirs('models', exist_ok=True)
        
        for model_name, model in self.best_models.items():
            joblib.dump(model, f'models/{model_name.lower()}_model.pkl')
            print(f"Saved {model_name} model")
        
        # Save feature importance
        joblib.dump(self.feature_importance, 'models/feature_importance.pkl')
        print("Saved feature importance")
    
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
        tree_models = ['XGBoost']
        if any(model in self.feature_importance for model in tree_models):
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            for i, model_name in enumerate(tree_models):
                if model_name in self.feature_importance:
                    importance = self.feature_importance[model_name]
                    feature_names = X_test.columns
                    
                    # Get top 10 features
                    top_indices = np.argsort(importance)[-10:]
                    
                    axes[i].barh(range(len(top_indices)), importance[top_indices])
                    axes[i].set_yticks(range(len(top_indices)))
                    axes[i].set_yticklabels([feature_names[j] for j in top_indices])
                    axes[i].set_xlabel('Feature Importance')
                    axes[i].set_title(f'{model_name} Feature Importance')
            
            plt.tight_layout()
            plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("Plots saved to plots/ directory")
    
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
        
        return results

if __name__ == "__main__":
    trainer = ModelTrainer()
    results = trainer.run_training_pipeline() 