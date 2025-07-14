#!/usr/bin/env python3
"""
Healthcare Readmission Prediction Pipeline
========================================

This script orchestrates the complete pipeline for predicting diabetic patient readmissions:
1. Data preprocessing
2. Model training (Logistic Regression, Random Forest, XGBoost)
3. Model evaluation
4. Dashboard generation

Usage:
    python main.py [--preprocess] [--train] [--test] [--dashboard]
"""

import argparse
import os
import sys
from pathlib import Path

# Import W&B configuration
try:
    from wandb_config import init_wandb, setup_wandb, finish_wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ W&B not available. Install with: pip install wandb")

def main():
    parser = argparse.ArgumentParser(description='Healthcare Readmission Prediction Pipeline')
    parser.add_argument('--preprocess', action='store_true', help='Run data preprocessing')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--test', action='store_true', help='Run hypothesis testing')
    parser.add_argument('--dashboard', action='store_true', help='Launch dashboard')
    parser.add_argument('--all', action='store_true', help='Run complete pipeline')
    
    args = parser.parse_args()
    
    # If no arguments provided, run complete pipeline
    if not any([args.preprocess, args.train, args.test, args.dashboard, args.all]):
        args.all = True
    
    print(" Healthcare Readmission Prediction Pipeline")
    print("=" * 50)
    
    try:
        if args.all or args.preprocess:
            print("\n Step 1: Data Preprocessing")
            print("-" * 30)
            run_preprocessing()
        
        if args.all or args.train:
            print("\n Step 2: Model Training")
            print("-" * 30)
            run_training()
        
        if args.all or args.test:
            print("\n Step 3: Hypothesis Testing")
            print("-" * 30)
            run_hypothesis_testing()
        
        if args.all or args.dashboard:
            print("\n Step 4: Launching Dashboard")
            print("-" * 30)
            launch_dashboard()
        
        print("\n Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n Error in pipeline: {e}")
        sys.exit(1)

def run_preprocessing():
    """Run data preprocessing pipeline"""
    try:
        from data_preprocessing import DataPreprocessor
        
        print("Running data preprocessing...")
        preprocessor = DataPreprocessor()
        df, X_train, X_test, y_train, y_test = preprocessor.run_preprocessing_pipeline(
            'data/diabetic_data.csv',
            'data/processed_data.csv'
        )
        
        print(f" Preprocessing completed!")
        print(f"   - Processed data shape: {df.shape}")
        print(f"   - Training set shape: {X_train.shape}")
        print(f"   - Test set shape: {X_test.shape}")
        
    except ImportError as e:
        print(f" Import error: {e}")
        print("Please install required packages: pip install -r requirements.txt")
        raise
    except Exception as e:
        print(f" Preprocessing error: {e}")
        raise

def run_training():
    """Run model training pipeline"""
    try:
        from model_training import ModelTrainer
        
        # Initialize W&B if available
        if WANDB_AVAILABLE and setup_wandb():
            init_wandb()
            print("✅ W&B tracking enabled")
        else:
            print("⚠️ W&B tracking disabled")
        
        print("Training models...")
        trainer = ModelTrainer()
        results = trainer.run_training_pipeline()
        
        print(" Model training completed!")
        
        # Print best model
        best_model = max(results.items(), key=lambda x: x[1]['auc'])
        print(f"   - Best model: {best_model[0]} (AUC: {best_model[1]['auc']:.4f})")
        
        # Finish W&B run
        if WANDB_AVAILABLE:
            finish_wandb()
        
    except ImportError as e:
        print(f" Import error: {e}")
        print("Please install required packages: pip install -r requirements.txt")
        raise
    except Exception as e:
        print(f" Training error: {e}")
        raise

def run_hypothesis_testing():
    """Run hypothesis testing"""
    try:
        from hypothesis_testing import HypothesisTester
        
        print("Running hypothesis tests...")
        tester = HypothesisTester()
        results = tester.run_hypothesis_tests()
        
        print(" Hypothesis testing completed!")
        
        # Print key findings
        if 'insulin_test' in results:
            insulin_test = results['insulin_test']
            significant = insulin_test['p_value'] < 0.05
            print(f"   - Insulin usage correlation: {'Significant' if significant else 'Not significant'}")
            print(f"   - p-value: {insulin_test['p_value']:.4f}")
        
    except ImportError as e:
        print(f" Import error: {e}")
        print("Please install required packages: pip install -r requirements.txt")
        raise
    except Exception as e:
        print(f" Hypothesis testing error: {e}")
        raise

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    try:
        import subprocess
        import sys
        
        print("Launching Streamlit dashboard...")
        print("Dashboard will open in your browser at http://localhost:8501")
        print("Press Ctrl+C to stop the dashboard")
        
        # Run streamlit dashboard
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
    except ImportError:
        print(" Streamlit not installed. Installing...")
        os.system("pip install streamlit plotly")
        print("Please run the dashboard again.")
    except KeyboardInterrupt:
        print("\n Dashboard stopped by user")
    except Exception as e:
        print(f" Dashboard error: {e}")
        raise

def check_dependencies():
    """Check if all required dependencies are installed"""
    # Map of package names to their import names
    package_imports = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'scikit-learn': 'sklearn',
        'xgboost': 'xgboost',
        'imbalanced-learn': 'imblearn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'plotly': 'plotly',
        'streamlit': 'streamlit',
        'scipy': 'scipy'
    }
    
    missing_packages = []
    
    for package_name, import_name in package_imports.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f" Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        'data/diabetic_data.csv'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f" Missing data files: {', '.join(missing_files)}")
        print("Please ensure the data files are in the correct location.")
        return False
    
    return True

if __name__ == "__main__":
    # Check dependencies and data files
    if not check_dependencies():
        sys.exit(1)
    
    if not check_data_files():
        sys.exit(1)
    
    # Run main pipeline
    main() 