"""
Weights & Biases (W&B) Configuration for Diabetic Readmission Prediction
"""

import wandb
import os

def init_wandb():
    """Initialize W&B with project configuration"""
    try:
        wandb.init(
            project="diabetic-readmission-prediction",
            config={
                "model_types": ["XGBoost", "CatBoost", "LightGBM"],
                "dataset": "UCI Diabetes 130-US Hospitals",
                "target": "30-day readmission prediction",
                "data_size": "101,766 records",
                "features": "50+ clinical variables",
                "test_size": 0.2,
                "random_state": 42
            }
        )
        print(" W&B initialized successfully")
        return True
    except Exception as e:
        print(f" W&B initialization failed: {e}")
        print("Continuing without W&B tracking...")
        return False

def get_sweep_config():
    """Get W&B sweep configuration for hyperparameter optimization"""
    sweep_config = {
        "method": "grid",
        "name": "diabetic-readmission-hyperparameter-sweep",
        "metric": {
            "name": "test_auc",
            "goal": "maximize"
        },
        "parameters": {
            "xgboost_n_estimators": {
                "values": [100, 200, 300]
            },
            "xgboost_max_depth": {
                "values": [3, 6, 9]
            },
            "xgboost_learning_rate": {
                "values": [0.1, 0.2, 0.3]
            },
            "catboost_iterations": {
                "values": [100, 200, 300]
            },
            "catboost_depth": {
                "values": [4, 6, 8]
            },
            "catboost_learning_rate": {
                "values": [0.1, 0.2, 0.3]
            },
            "lightgbm_n_estimators": {
                "values": [100, 200, 300]
            },
            "lightgbm_max_depth": {
                "values": [3, 6, 9]
            },
            "lightgbm_learning_rate": {
                "values": [0.1, 0.2, 0.3]
            }
        }
    }
    return sweep_config

def log_model_performance(model_name, metrics, params=None):
    """Log model performance metrics to W&B"""
    try:
        log_dict = {
            f"{model_name}_accuracy": metrics.get('accuracy', 0),
            f"{model_name}_auc": metrics.get('auc', 0),
            f"{model_name}_precision": metrics.get('precision', 0),
            f"{model_name}_recall": metrics.get('recall', 0),
            f"{model_name}_f1_score": metrics.get('f1_score', 0)
        }
        
        if params:
            log_dict[f"{model_name}_params"] = params
            
        wandb.log(log_dict)
        print(f" Logged {model_name} performance to W&B")
    except Exception as e:
        print(f" W&B logging failed for {model_name}: {e}")

def log_feature_importance(model_name, feature_names, importance_scores):
    """Log feature importance to W&B"""
    try:
        # Create feature importance table
        importance_data = [[name, score] for name, score in zip(feature_names, importance_scores)]
        table = wandb.Table(data=importance_data, columns=["Feature", "Importance"])
        wandb.log({f"{model_name}_feature_importance": table})
        print(f" Logged {model_name} feature importance to W&B")
    except Exception as e:
        print(f" W&B feature importance logging failed: {e}")

def log_plots(plot_paths):
    """Log plot images to W&B"""
    try:
        for plot_name, plot_path in plot_paths.items():
            if os.path.exists(plot_path):
                wandb.log({plot_name: wandb.Image(plot_path)})
                print(f" Logged {plot_name} to W&B")
    except Exception as e:
        print(f" W&B plot logging failed: {e}")

def finish_wandb():
    """Finish W&B run"""
    try:
        wandb.finish()
        print(" W&B run completed successfully")
    except Exception as e:
        print(f" W&B finalization failed: {e}")

# Environment variable for W&B API key
def setup_wandb():
    """Setup W&B environment"""
    if not os.getenv("WANDB_API_KEY"):
        print(" WANDB_API_KEY not found. Set it to enable W&B tracking.")
        print("Get your API key from: https://wandb.ai/settings")
        return False
    return True 