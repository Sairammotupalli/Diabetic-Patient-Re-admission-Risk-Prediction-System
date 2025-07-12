# Docker Setup for Healthcare Readmission Prediction Pipeline

This document explains how to use Docker to run the healthcare readmission prediction pipeline.

## 🐳 Quick Start

### Prerequisites
- Docker installed and running
- Docker Compose installed
- Data file: `data/diabetic_data.csv`

### 1. Build the Docker Image
```bash
./run_docker.sh build
```

### 2. Run the Dashboard
```bash
./run_docker.sh dashboard
```
The dashboard will be available at: http://localhost:8501

### 3. Run Complete Pipeline
```bash
./run_docker.sh full
```

## 📋 Available Commands

### Using the Helper Script
```bash
./run_docker.sh [COMMAND]
```

**Commands:**
- `build` - Build the Docker image
- `preprocess` - Run data preprocessing only
- `train` - Run model training only
- `test` - Run hypothesis testing only
- `dashboard` - Start Streamlit dashboard
- `jupyter` - Start Jupyter Lab
- `full` - Run complete pipeline
- `cleanup` - Clean up containers
- `logs` - Show container logs
- `shell` - Enter container shell
- `help` - Show help message

### Using Docker Compose Directly
```bash
# Start dashboard
docker-compose up --build

# Run preprocessing
docker-compose --profile preprocessing up --build

# Run training
docker-compose --profile training up --build

# Run hypothesis testing
docker-compose --profile testing up --build

# Run full pipeline
docker-compose --profile full-pipeline up --build

# Start Jupyter Lab
docker-compose --profile development up --build
```

## 🏗️ Docker Architecture

### Services
1. **healthcare-pipeline** - Main service with dashboard
2. **preprocessing** - Data preprocessing service
3. **training** - Model training service
4. **hypothesis-testing** - Statistical testing service
5. **full-pipeline** - Complete pipeline service
6. **jupyter** - Development environment with Jupyter Lab

### Ports
- `8501` - Streamlit dashboard
- `8000` - FastAPI (if needed)
- `8888` - Jupyter Lab

### Volumes
- `./data` → `/app/data` - Data files
- `./models` → `/app/models` - Trained models
- `./plots` → `/app/plots` - Generated plots

## 🔧 Docker Configuration

### Dockerfile Features
- **Base Image**: Python 3.9-slim
- **Security**: Non-root user (appuser)
- **Optimization**: Multi-stage build, .dockerignore
- **Health Check**: Automatic health monitoring
- **Entrypoint**: Custom entrypoint script

### Environment Variables
- `PYTHONUNBUFFERED=1` - Real-time Python output
- `PYTHONDONTWRITEBYTECODE=1` - No .pyc files
- `PYTHONPATH=/app` - Python path configuration

## 📊 Data Management

### Data Files
Place your data files in the `data/` directory:
```
data/
├── diabetic_data.csv    # Required: Raw diabetic data
├── cleaned_data.csv     # Generated: Preprocessed data
└── processed_data.csv   # Generated: Final processed data
```

### Persistent Storage
The following directories are mounted as volumes:
- `./data` - Input and output data files
- `./models` - Trained models and artifacts
- `./plots` - Generated visualizations

## 🚀 Usage Examples

### Example 1: Quick Dashboard
```bash
# Build and start dashboard
./run_docker.sh build
./run_docker.sh dashboard

# Access dashboard at http://localhost:8501
```

### Example 2: Step-by-Step Pipeline
```bash
# 1. Preprocess data
./run_docker.sh preprocess

# 2. Train models
./run_docker.sh train

# 3. Run hypothesis tests
./run_docker.sh test

# 4. Start dashboard
./run_docker.sh dashboard
```

### Example 3: Development with Jupyter
```bash
# Start Jupyter Lab
./run_docker.sh jupyter

# Access Jupyter at http://localhost:8888
```

### Example 4: Complete Pipeline
```bash
# Run everything at once
./run_docker.sh full
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Docker Not Running
```bash
# Check Docker status
docker info

# Start Docker Desktop (if applicable)
```

#### 2. Port Already in Use
```bash
# Check what's using the port
lsof -i :8501

# Kill the process or use different port
docker-compose up -p 8502
```

#### 3. Permission Issues
```bash
# Fix file permissions
chmod +x run_docker.sh
chmod +x docker_entrypoint.sh
```

#### 4. Data File Missing
```bash
# Ensure data file exists
ls -la data/diabetic_data.csv

# Copy data file if needed
cp /path/to/your/data.csv data/diabetic_data.csv
```

#### 5. Memory Issues
```bash
# Increase Docker memory limit
# In Docker Desktop: Settings → Resources → Memory
```

### Debugging Commands

#### View Container Logs
```bash
# All services
./run_docker.sh logs

# Specific service
docker-compose logs healthcare-pipeline
```

#### Enter Container Shell
```bash
# Interactive shell
./run_docker.sh shell

# Or directly
docker-compose exec healthcare-pipeline bash
```

#### Check Container Status
```bash
# List containers
docker-compose ps

# Container details
docker-compose exec healthcare-pipeline ps aux
```

## 🛠️ Development

### Building for Development
```bash
# Build with development dependencies
docker build -t healthcare-dev .

# Run with volume mounts for development
docker run -it -v $(pwd):/app healthcare-dev bash
```

### Custom Docker Compose
Create `docker-compose.override.yml` for custom configurations:
```yaml
version: '3.8'
services:
  healthcare-pipeline:
    environment:
      - DEBUG=1
    volumes:
      - ./custom_data:/app/data
```

## 📈 Performance Optimization

### Build Optimization
- Use `.dockerignore` to exclude unnecessary files
- Multi-stage builds for smaller images
- Layer caching with requirements.txt first

### Runtime Optimization
- Volume mounts for data persistence
- Resource limits in docker-compose.yml
- Health checks for monitoring

### Memory Usage
- Default: 2GB RAM recommended
- For large datasets: 4GB+ RAM
- Monitor with `docker stats`

## 🔒 Security Considerations

### Container Security
- Non-root user (appuser)
- Minimal base image (python:3.9-slim)
- No sensitive data in image
- Health checks for monitoring

### Data Security
- Data files mounted as volumes
- No data stored in container
- Temporary files cleaned up

## 📝 Best Practices

### 1. Data Management
- Always use volume mounts for data
- Keep data files outside container
- Use .gitignore for large files

### 2. Development Workflow
- Use Jupyter service for development
- Mount source code as volume
- Use docker-compose.override.yml for custom configs

### 3. Production Deployment
- Use specific image tags
- Set resource limits
- Configure logging
- Use secrets management

### 4. Monitoring
- Use health checks
- Monitor container logs
- Set up alerts for failures

## 🎯 Next Steps

1. **Customize**: Modify docker-compose.yml for your needs
2. **Scale**: Add more services as needed
3. **Monitor**: Set up monitoring and alerting
4. **CI/CD**: Integrate with your CI/CD pipeline

## 📞 Support

For issues with Docker setup:
1. Check the troubleshooting section
2. Review container logs
3. Verify data file existence
4. Check Docker and Docker Compose versions 