# Diabetic-Patient-Re-admission-Risk-Prediction-System

[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-Streamlit-blue?style=for-the-badge&logo=streamlit)](https://diabetic-patient-re-admission-risk-prediction-system-j52gbqfnw.streamlit.app/)



## Project Overview

**Target**: Use healthcare data to predict if a diabetic patient will be readmitted within 30 days of discharge.


**Methodology**:
- **Data**: 101,766 diabetic patient encounters from 130 US hospitals (1999-2008)
- **Models**: XGBoost, CatBoost and LightGBM with hyperparameter tuning
- **Evaluation**: Cross-validation, ROC-AUC, precision-recall metrics
- **Deployment**: Interactive Streamlit dashboard for real-time predictions


## Dataset Information

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


## Experiment Tracking with Weights & Biases

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


#### **Ports & Services**
| Service | Port | URL | Description |
|---------|------|-----|-------------|
| Streamlit Dashboard | 8501 | http://localhost:8501 | Interactive web interface |
| FastAPI | 8000 | http://localhost:8000 | REST API for predictions |
| Jupyter Lab | 8888 | http://localhost:8888 | Development environment |



## Models Implemented

### 1. XGBoost
### 2. CatBoost
### 3. LightGBM


## Evaluation Metrics

- **Accuracy**: Overall prediction accuracy
- **AUC-ROC**: Area under ROC curve
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

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

### Hypothesis testing
![ROC Curves](plots/hypothesis testing.png)


## License

This project is licensed under the MIT License - see the LICENSE file for details.
