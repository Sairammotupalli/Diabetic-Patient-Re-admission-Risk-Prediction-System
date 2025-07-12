#!/bin/bash

# Docker Entrypoint Script for Healthcare Readmission Prediction Pipeline

set -e

# Function to print status
print_status() {
    echo "[INFO] $1"
}

# Function to check if data exists
check_data() {
    if [ ! -f "/app/data/diabetic_data.csv" ]; then
        print_status "Warning: diabetic_data.csv not found in /app/data/"
        print_status "Please ensure your data file is in the data directory"
        return 1
    fi
    return 0
}

# Function to run preprocessing
run_preprocessing() {
    print_status "Running data preprocessing..."
    cd /app
    python data_preprocessing.py
    print_status "Preprocessing completed!"
}

# Function to run training
run_training() {
    print_status "Running model training..."
    cd /app
    python model_training.py
    print_status "Training completed!"
}

# Function to run hypothesis testing
run_hypothesis_testing() {
    print_status "Running hypothesis testing..."
    cd /app
    python hypothesis_testing.py
    print_status "Hypothesis testing completed!"
}

# Function to start dashboard
start_dashboard() {
    print_status "Starting Streamlit dashboard..."
    cd /app
    streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0
}

# Function to start Jupyter
start_jupyter() {
    print_status "Starting Jupyter Lab..."
    cd /app
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
}

# Function to run full pipeline
run_full_pipeline() {
    print_status "Running complete pipeline..."
    cd /app
    python main.py
    print_status "Full pipeline completed!"
}

# Function to run specific step
run_step() {
    case $1 in
        "preprocess")
            run_preprocessing
            ;;
        "train")
            run_training
            ;;
        "test")
            run_hypothesis_testing
            ;;
        "dashboard")
            start_dashboard
            ;;
        "jupyter")
            start_jupyter
            ;;
        "full")
            run_full_pipeline
            ;;
        *)
            print_status "Unknown command: $1"
            print_status "Available commands: preprocess, train, test, dashboard, jupyter, full"
            exit 1
            ;;
    esac
}

# Main execution
main() {
    print_status "Healthcare Readmission Prediction Pipeline - Docker Container"
    print_status "Working directory: $(pwd)"
    
    # Check if data exists
    if ! check_data; then
        print_status "Data check failed, but continuing..."
    fi
    
    # Create necessary directories
    mkdir -p /app/data /app/models /app/plots
    
    # Parse command line arguments
    if [ $# -eq 0 ]; then
        print_status "No command specified, running full pipeline..."
        run_full_pipeline
    else
        run_step "$1"
    fi
}

# Execute main function with all arguments
main "$@" 