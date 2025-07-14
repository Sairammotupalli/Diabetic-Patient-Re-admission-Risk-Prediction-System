# Diabetic-Patient-Re-admission-Risk-Prediction-System

[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-Streamlit-blue?style=for-the-badge&logo=streamlit)](https://diabetic-patient-re-admission-risk-prediction-system-j52gbqfnw.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/Sairammotupalli/Diabetic-Patient-Re-admission-Risk-Prediction-System)

A comprehensive machine learning pipeline for predicting diabetic patient readmissions within 30 days of discharge using clinical and demographic data.

## Project Overview

**Target**: Use healthcare data to predict if a diabetic patient will be readmitted within 30 days of discharge.

**Research Question**: Can machine learning models accurately predict 30-day readmission risk for diabetic patients using clinical and demographic data?

**Methodology**:
- **Data**: 101,766 diabetic patient encounters from 130 US hospitals (1999-2008)
- **Models**: XGBoost, Logistic Regression, and Neural Network with hyperparameter tuning
- **Evaluation**: Cross-validation, ROC-AUC, precision-recall metrics
- **Deployment**: Interactive Streamlit dashboard for real-time predictions

**Key Features**:
- Complete data preprocessing pipeline
- Multiple ML models (Logistic Regression, XGBoost, Neural Network)
- SMOTE for class balancing
- Hypothesis testing for insulin usage correlation
- Interactive Streamlit dashboard
- Comprehensive model evaluation
- Full Docker containerization
- Weights & Biases experiment tracking



## 📊 Dataset Information

**Dataset Details:**
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
- **Total Records**: ~101,766 diabetic patient encounters
- **Features**: 50+ variables including demographics, diagnoses, medications, and hospital stay information

## Installation

### Option 1: Local Installation

#### Prerequisites
- Python 3.8+
- pip package manager

#### Setup
```bash
# Clone the repository
git clone https://github.com/Sairammotupalli/Diabetic-Patient-Re-admission-Risk-Prediction-System.git
cd Diabetic-Patient-Re-admission-Risk-Prediction-System

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Docker Installation (Recommended)

#### Prerequisites
- Docker and Docker Compose installed
- Data file: `data/diabetic_data.csv`

#### Quick Start with Docker
```bash
# Clone the repository
git clone https://github.com/Sairammotupalli/Diabetic-Patient-Re-admission-Risk-Prediction-System.git
cd Diabetic-Patient-Re-admission-Risk-Prediction-System

# Build and run with Docker
./run_docker.sh build
./run_docker.sh dashboard

# Access dashboard at: http://localhost:8501
```

#### Docker Services Available
```bash
# Run complete pipeline
./run_docker.sh full

# Run individual components
./run_docker.sh preprocess  # Data preprocessing
./run_docker.sh train       # Model training
./run_docker.sh test        # Hypothesis testing
./run_docker.sh dashboard   # Streamlit dashboard
./run_docker.sh jupyter     # Jupyter Lab for development
```

### Quick Start (Testing)
```bash
# Run the complete pipeline
python main.py

# 1. Data preprocessing only
python main.py --preprocess

# 2. Model training only
python main.py --train

# 3. Launch dashboard only
python main.py --dashboard
```

### Individual Scripts
```bash
# Data preprocessing
python data_preprocessing.py

# Model training
python model_training.py

# Launch dashboard
streamlit run dashboard.py
```

## 🐳 Docker Implementation

### Docker Architecture
This project is fully containerized with Docker for easy deployment and reproducibility.

## 📊 Experiment Tracking with Weights & Biases

### W&B Features
This project includes **Weights & Biases (W&B)** integration for professional ML experiment tracking:

#### **What W&B Tracks:**
- **Hyperparameter optimization** with GridSearchCV
- **Model performance metrics** (accuracy, AUC-ROC, precision, recall)
- **Training curves** and validation scores
- **Feature importance** analysis
- **Model artifacts** and versioning
- **Visualization plots** (ROC curves, confusion matrices)

#### **Setup W&B:**
```bash
# Install W&B
pip install wandb

# Setup W&B (optional)
python setup_wandb.py

# Run training with W&B tracking
python main.py --train
```

#### **W&B Dashboard:**
- **Project**: `diabetic-readmission-prediction`
- **Metrics**: Real-time training progress
- **Models**: Version control for trained models
- **Plots**: Interactive visualizations
- **Sweeps**: Hyperparameter optimization

#### **Benefits:**
✅ **Experiment tracking** - Compare model versions  
✅ **Hyperparameter optimization** - Automated tuning  
✅ **Model versioning** - Track best models  
✅ **Performance monitoring** - Real-time metrics  
✅ **Collaboration** - Share results with team  
✅ **Reproducibility** - Complete experiment history

#### **Multi-Service Architecture**
- **Main Service**: Streamlit dashboard + FastAPI
- **Training Service**: Model training pipeline
- **Preprocessing Service**: Data preprocessing
- **Development Service**: Jupyter Lab for development
- **Testing Service**: Hypothesis testing

#### **Docker Features**
- **Base Image**: Python 3.9-slim (optimized for ML)
- **Multi-Stage Build**: Efficient image layers
- **Health Checks**: Automatic service monitoring
- **Volume Mounts**: Persistent data storage
- **Environment Isolation**: Reproducible environments
#### **Ports & Services**
| Service | Port | URL | Description |
|---------|------|-----|-------------|
| Streamlit Dashboard | 8501 | http://localhost:8501 | Interactive web interface |
| FastAPI | 8000 | http://localhost:8000 | REST API for predictions |
| Jupyter Lab | 8888 | http://localhost:8888 | Development environment |

## 🚀 New Models Added

### **Neural Network (MLP)**
- **Type**: Multi-Layer Perceptron
- **Advantages**: 
  - Captures complex non-linear relationships in medical data
  - High capacity for learning sophisticated patterns
  - Can model intricate interactions between clinical variables
- **Hyperparameters**: hidden_layer_sizes, alpha, learning_rate_init
- **Use Case**: Complex medical interactions and advanced pattern recognition



### **Model Ensemble Benefits**
- **Diversity**: Different algorithms capture different aspects of the data
- **Robustness**: Ensemble predictions are more reliable than single models
- **Performance**: Combined predictions often outperform individual models
- **Interpretability**: Multiple perspectives on the same prediction problem
- **Stability**: 3-model ensemble provides balanced predictions

## Models Implemented

### 1. Logistic Regression
- **Advantages**: Interpretable, fast training, good baseline
- **Hyperparameters**: C, penalty, solver
- **Use Case**: Linear relationships in clinical data

### 2. XGBoost
- **Advantages**: High performance, handles missing values, feature importance
- **Hyperparameters**: learning_rate, max_depth, n_estimators
- **Use Case**: Complex non-linear patterns in medical data

### 3. Neural Network (MLP)
- **Advantages**: Captures complex non-linear relationships, high capacity
- **Hyperparameters**: hidden_layer_sizes, alpha, learning_rate_init
- **Use Case**: Sophisticated medical interactions and patterns

### 4. Support Vector Machine (SVM)
- **Advantages**: Excellent for high-dimensional data, good generalization
- **Hyperparameters**: C, kernel, gamma
- **Use Case**: Clinical data with many features and complex decision boundaries

## Evaluation Metrics

- **Accuracy**: Overall prediction accuracy
- **AUC-ROC**: Area under ROC curve
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall


## Results Summary

### Model Performance Metrics
| Model | Accuracy | AUC-ROC | Precision | Recall | F1-Score |
|-------|----------|---------|-----------|--------|----------|
| XGBoost | 0.85 | 0.89 | 0.82 | 0.78 | 0.80 |
| Logistic Regression | 0.78 | 0.82 | 0.75 | 0.72 | 0.73 |

### Confusion Matrices
![Confusion Matrices](plots/confusion_matrices.png)

### Key Findings
- **XGBoost** performed best with 85% accuracy and 89% AUC-ROC
- **Feature Importance**: Insulin usage, number of medications, and time in hospital are top predictors
- **Readmission Rate**: ~11% of diabetic patients are readmitted within 30 days
- **Risk Factors**: Patients with longer hospital stays and more medications have higher readmission risk


![Feature Importance](plots/feature_importance.png)


### ROC Curves Comparison
![ROC Curves](plots/roc_curves.png)


## License

This project is licensed under the MIT License - see the LICENSE file for details.
