#!/bin/bash

# Readmission Prediction System - Docker Deployment Script
# Usage: ./deploy.sh [command]

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
    print_success "Docker is running"
}

# Function to build the Docker image
build_image() {
    print_status "Building Docker image..."
    docker build -t readmission-prediction .
    print_success "Docker image built successfully"
}

# Function to run the complete system
run_complete() {
    print_status "Starting complete readmission prediction system..."
    docker-compose up -d
    print_success "System started!"
    print_status "Services available at:"
    echo "  - Dashboard: http://localhost:8501"
    echo "  - API: http://localhost:8000"
    echo "  - API Docs: http://localhost:8000/docs"
}

# Function to run dashboard only
run_dashboard() {
    print_status "Starting dashboard only..."
    docker-compose --profile dashboard up -d dashboard-only
    print_success "Dashboard started at http://localhost:8501"
}

# Function to run API only
run_api() {
    print_status "Starting API only..."
    docker-compose --profile api up -d api-only
    print_success "API started at http://localhost:8000"
    print_status "API documentation available at http://localhost:8000/docs"
}

# Function to run development environment
run_dev() {
    print_status "Starting development environment..."
    docker-compose --profile development up -d development
    print_success "Development environment started!"
    print_status "Services available at:"
    echo "  - Jupyter Lab: http://localhost:8888"
    echo "  - Dashboard: http://localhost:8501"
    echo "  - API: http://localhost:8000"
}

# Function to run training
run_training() {
    print_status "Running model training..."
    docker-compose --profile training up training
    print_success "Training completed"
}

# Function to run preprocessing
run_preprocessing() {
    print_status "Running data preprocessing..."
    docker-compose --profile preprocessing up preprocessing
    print_success "Preprocessing completed"
}

# Function to stop all services
stop_all() {
    print_status "Stopping all services..."
    docker-compose down
    print_success "All services stopped"
}

# Function to show logs
show_logs() {
    print_status "Showing logs..."
    docker-compose logs -f
}

# Function to clean up
cleanup() {
    print_status "Cleaning up Docker resources..."
    docker-compose down --volumes --remove-orphans
    docker system prune -f
    print_success "Cleanup completed"
}

# Function to show status
show_status() {
    print_status "Checking service status..."
    docker-compose ps
}

# Function to show help
show_help() {
    echo "Readmission Prediction System - Docker Deployment Script"
    echo ""
    echo "Usage: ./deploy.sh [command]"
    echo ""
    echo "Commands:"
    echo "  build       - Build Docker image"
    echo "  complete    - Run complete system (dashboard + API)"
    echo "  dashboard   - Run dashboard only"
    echo "  api         - Run API only"
    echo "  dev         - Run development environment (Jupyter + Dashboard + API)"
    echo "  train       - Run model training"
    echo "  preprocess  - Run data preprocessing"
    echo "  stop        - Stop all services"
    echo "  logs        - Show logs"
    echo "  status      - Show service status"
    echo "  cleanup     - Clean up Docker resources"
    echo "  help        - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./deploy.sh build"
    echo "  ./deploy.sh complete"
    echo "  ./deploy.sh dashboard"
    echo "  ./deploy.sh dev"
}

# Main script logic
main() {
    # Check if Docker is running
    check_docker
    
    case "${1:-help}" in
        "build")
            build_image
            ;;
        "complete")
            build_image
            run_complete
            ;;
        "dashboard")
            build_image
            run_dashboard
            ;;
        "api")
            build_image
            run_api
            ;;
        "dev")
            build_image
            run_dev
            ;;
        "train")
            build_image
            run_training
            ;;
        "preprocess")
            build_image
            run_preprocessing
            ;;
        "stop")
            stop_all
            ;;
        "logs")
            show_logs
            ;;
        "status")
            show_status
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Run main function with all arguments
main "$@" 