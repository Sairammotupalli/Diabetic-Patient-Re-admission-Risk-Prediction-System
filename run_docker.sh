#!/bin/bash

# Healthcare Readmission Prediction Pipeline - Docker Runner
# This script provides easy commands to run the pipeline in Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to check if Docker Compose is available
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose and try again."
        exit 1
    fi
}

# Function to build the Docker image
build_image() {
    print_status "Building Docker image..."
    docker build -t healthcare-readmission-pipeline .
    print_success "Docker image built successfully!"
}

# Function to run the full pipeline
run_full_pipeline() {
    print_status "Running complete healthcare pipeline..."
    docker-compose --profile full-pipeline up --build
}

# Function to run data preprocessing
run_preprocessing() {
    print_status "Running data preprocessing..."
    docker-compose --profile preprocessing up --build
}

# Function to run model training
run_training() {
    print_status "Running model training..."
    docker-compose --profile training up --build
}

# Function to run hypothesis testing
run_hypothesis_testing() {
    print_status "Running hypothesis testing..."
    docker-compose --profile testing up --build
}

# Function to start the dashboard
start_dashboard() {
    print_status "Starting Streamlit dashboard..."
    print_status "Dashboard will be available at http://localhost:8501"
    docker-compose up --build
}

# Function to start Jupyter Lab
start_jupyter() {
    print_status "Starting Jupyter Lab..."
    print_status "Jupyter Lab will be available at http://localhost:8888"
    docker-compose --profile development up --build
}

# Function to run a specific step
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
            print_error "Unknown step: $1"
            print_usage
            exit 1
            ;;
    esac
}

# Function to clean up containers
cleanup() {
    print_status "Cleaning up Docker containers..."
    docker-compose down
    print_success "Containers cleaned up!"
}

# Function to show logs
show_logs() {
    docker-compose logs -f
}

# Function to enter the container
enter_container() {
    print_status "Entering the container..."
    docker-compose exec healthcare-pipeline bash
}

# Function to print usage
print_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build              Build the Docker image"
    echo "  preprocess         Run data preprocessing"
    echo "  train              Run model training"
    echo "  test               Run hypothesis testing"
    echo "  dashboard          Start Streamlit dashboard"
    echo "  jupyter            Start Jupyter Lab"
    echo "  full               Run complete pipeline"
    echo "  cleanup            Clean up containers"
    echo "  logs               Show container logs"
    echo "  shell              Enter the container shell"
    echo "  help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build           # Build the Docker image"
    echo "  $0 dashboard       # Start the dashboard"
    echo "  $0 full            # Run complete pipeline"
}

# Main script logic
main() {
    # Check prerequisites
    check_docker
    check_docker_compose

    # Parse command line arguments
    case ${1:-help} in
        "build")
            build_image
            ;;
        "preprocess"|"train"|"test"|"dashboard"|"jupyter"|"full")
            run_step $1
            ;;
        "cleanup")
            cleanup
            ;;
        "logs")
            show_logs
            ;;
        "shell")
            enter_container
            ;;
        "help"|"-h"|"--help")
            print_usage
            ;;
        *)
            print_error "Unknown command: $1"
            print_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@" 