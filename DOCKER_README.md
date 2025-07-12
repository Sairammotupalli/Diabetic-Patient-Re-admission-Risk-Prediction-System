# 🐳 Docker Deployment Guide

## Readmission Prediction System - Docker Setup

This guide explains how to deploy the complete readmission prediction system using Docker.

## 📋 Prerequisites

- Docker Desktop installed and running
- Docker Compose installed
- At least 4GB RAM available for Docker

## 🚀 Quick Start

### 1. Build and Run Complete System

```bash
# Build and start all services
./deploy.sh complete
```

This will start:
- **Dashboard**: http://localhost:8501
- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### 2. Run Individual Components

```bash
# Dashboard only
./deploy.sh dashboard

# API only
./deploy.sh api

# Development environment (Jupyter + Dashboard + API)
./deploy.sh dev
```

## 📁 Project Structure

```
Readmission_Prediction/
├── Dockerfile                 # Docker image definition
├── docker-compose.yml         # Multi-service orchestration
├── deploy.sh                  # Deployment script
├── requirements.txt           # Python dependencies
├── dashboard.py              # Streamlit dashboard
├── api_predictor.py          # FastAPI service
├── single_patient_predictor.py # Command-line predictor
├── model_training.py         # Model training script
├── data_preprocessing.py     # Data preprocessing script
├── data/                     # Data files
├── models/                   # Trained models
└── plots/                    # Generated plots
```

## 🔧 Available Commands

### Deployment Script Commands

```bash
./deploy.sh build       # Build Docker image
./deploy.sh complete    # Run complete system
./deploy.sh dashboard   # Run dashboard only
./deploy.sh api         # Run API only
./deploy.sh dev         # Run development environment
./deploy.sh train       # Run model training
./deploy.sh preprocess  # Run data preprocessing
./deploy.sh stop        # Stop all services
./deploy.sh logs        # Show logs
./deploy.sh status      # Show service status
./deploy.sh cleanup     # Clean up Docker resources
./deploy.sh help        # Show help
```

### Manual Docker Commands

```bash
# Build image
docker build -t readmission-prediction .

# Run complete system
docker-compose up -d

# Run specific service
docker-compose --profile dashboard up -d dashboard-only

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🌐 Services Overview

### 1. Streamlit Dashboard (Port 8501)
- **URL**: http://localhost:8501
- **Purpose**: User-friendly web interface for predictions
- **Features**: 
  - Patient data input forms
  - Real-time predictions
  - Visual results and recommendations
  - Risk level assessment

### 2. FastAPI Service (Port 8000)
- **URL**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Purpose**: REST API for predictions
- **Features**:
  - Single patient prediction
  - Batch prediction
  - Health check endpoints
  - Model information

### 3. Development Environment (Port 8888)
- **URL**: http://localhost:8888
- **Purpose**: Jupyter Lab for development
- **Features**:
  - Interactive notebooks
  - Data exploration
  - Model development

## 📊 Usage Examples

### Using the Dashboard

1. Open http://localhost:8501
2. Fill in patient information
3. Click "Predict Readmission Risk"
4. View results and recommendations

### Using the API

```bash
# Single patient prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "race": "Caucasian",
    "gender": "Female",
    "age": "[50-60)",
    "time_in_hospital": 5,
    "num_medications": 15
  }'

# Health check
curl http://localhost:8000/health
```

### Using Command Line

```bash
# Run single patient prediction
docker exec readmission-prediction-app python3 single_patient_predictor.py

# Run model training
docker exec readmission-prediction-app python3 model_training.py
```

## 🔍 Monitoring and Debugging

### View Logs
```bash
# All services
./deploy.sh logs

# Specific service
docker-compose logs -f dashboard-only
```

### Check Status
```bash
# Service status
./deploy.sh status

# Container status
docker ps
```

### Health Checks
```bash
# Dashboard health
curl http://localhost:8501

# API health
curl http://localhost:8000/health
```

## 🛠️ Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using the port
   lsof -i :8501
   
   # Stop conflicting services
   docker-compose down
   ```

2. **Out of Memory**
   ```bash
   # Increase Docker memory limit in Docker Desktop
   # Recommended: 4GB minimum
   ```

3. **Model Files Missing**
   ```bash
   # Run preprocessing and training
   ./deploy.sh preprocess
   ./deploy.sh train
   ```

4. **Permission Issues**
   ```bash
   # Make script executable
   chmod +x deploy.sh
   ```

### Reset Everything
```bash
# Stop and clean up
./deploy.sh cleanup

# Rebuild and start
./deploy.sh complete
```

## 🔒 Security Considerations

- Services run on localhost by default
- No authentication implemented (add for production)
- Consider using Docker secrets for sensitive data
- Use HTTPS in production environments

## 📈 Production Deployment

For production deployment:

1. **Use a reverse proxy** (nginx, traefik)
2. **Add authentication** to the dashboard and API
3. **Use environment variables** for configuration
4. **Set up monitoring** (Prometheus, Grafana)
5. **Use Docker volumes** for persistent data
6. **Implement health checks** and auto-restart

## 🆘 Support

If you encounter issues:

1. Check the logs: `./deploy.sh logs`
2. Verify Docker is running: `docker info`
3. Check service status: `./deploy.sh status`
4. Try rebuilding: `./deploy.sh cleanup && ./deploy.sh complete`

## 📝 Notes

- The system requires trained models in the `models/` directory
- First run may take longer due to Docker image building
- Services are configured to restart automatically
- Data is persisted in Docker volumes 

## 🐳 **Why Use Docker for Your Readmission Prediction System?**

Here are the key benefits and use cases for using Docker with your healthcare prediction system:

### 🎯 **Primary Use Cases:**

#### **1. Easy Deployment & Distribution**
- **Share your system** with other hospitals/clinics
- **One command setup**: `./deploy.sh complete`
- **No "works on my machine" issues**
- **Consistent environment** across different computers

#### **2. Healthcare Environment Integration**
- **Deploy in hospital networks** with strict security
- **Isolated environment** - doesn't interfere with existing systems
- **Easy to update** without affecting other applications
- **Compliance-friendly** - can be audited and controlled

#### **3. Multi-Service Architecture**
- **Dashboard** (Streamlit) for healthcare staff
- **API** (FastAPI) for integration with existing hospital systems
- **Development environment** (Jupyter) for research teams
- **All services run independently** but work together

### 🏥 **Healthcare-Specific Benefits:**

#### **Clinical Deployment**
```bash
# Deploy in hospital environment
./deploy.sh dashboard    # Just the prediction interface
./deploy.sh api         # Just the API for integration
```

#### **Research & Development**
```bash
# Development environment for research teams
./deploy.sh dev         # Jupyter + Dashboard + API
```

#### **Training & Updates**
```bash
<code_block_to_apply_changes_from>
```

### 🔧 **Technical Benefits:**

#### **Environment Consistency**
- **Same setup** on developer laptop, test server, and production
- **No dependency conflicts** with other software
- **Reproducible builds** - exact same environment every time

#### **Scalability**
- **Easy to scale** - run multiple instances
- **Load balancing** - distribute predictions across containers
- **Resource isolation** - each service uses only what it needs

#### **Security & Compliance**
- **Isolated environment** - can't access host system
- **Controlled dependencies** - only approved packages
- **Audit trail** - can track exactly what's running

### 📊 **Real-World Scenarios:**

#### **Scenario 1: Hospital Integration**
```bash
# Deploy API for hospital's existing system
./deploy.sh api
# Hospital's EHR system calls your API for predictions
```

#### **Scenario 2: Clinical Research**
```bash
# Deploy development environment for research team
./deploy.sh dev
# Researchers can explore data and develop new models
```

#### **Scenario 3: Multi-Hospital Deployment**
```bash
# Deploy dashboard at multiple hospitals
./deploy.sh dashboard
# Each hospital gets the same interface and functionality
```

### 🚀 **Deployment Flexibility:**

#### **Local Development**
```bash
./deploy.sh dev          # Full development environment
```

#### **Production Deployment**
```bash
./deploy.sh complete     # Dashboard + API for production
```

#### **Testing Environment**
```bash
./deploy.sh dashboard    # Just test the interface
```

### 💡 **Key Advantages for Healthcare:**

1. **Rapid Deployment** - Set up in minutes, not days
2. **Version Control** - Easy to rollback to previous versions
3. **Resource Efficiency** - Uses only needed resources
4. **Security** - Isolated from other systems
5. **Compliance** - Can be audited and controlled
6. **Scalability** - Easy to add more capacity
7. **Maintenance** - Easy to update and patch

### 📈 **Bottom Line:**

Docker makes your readmission prediction system:
- **Easy to deploy** anywhere
- **Consistent** across environments
- **Secure** and isolated
- **Scalable** for growth
- **Maintainable** for long-term use

**Perfect for healthcare environments** where reliability, security, and ease of deployment are critical! 