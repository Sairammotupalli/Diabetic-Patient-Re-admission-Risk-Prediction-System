# Use Python 3.9 slim image for better compatibility
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-docker.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy all application files
COPY . .

# Create necessary directories
RUN mkdir -p models data plots

# Expose ports for Streamlit and API
EXPOSE 8501 8000

# Create a startup script
RUN echo '#!/bin/bash\n\
echo "Starting Readmission Prediction System..."\n\
echo "Running full pipeline (preprocessing, training, hypothesis testing)..."\n\
python3 main.py --preprocess --train --test\n\
echo "Pipeline completed!"\n\
echo ""\n\
echo "Starting services..."\n\
echo "1. Streamlit Dashboard: http://localhost:8501"\n\
echo "2. FastAPI: http://localhost:8000"\n\
echo ""\n\
echo "Starting Streamlit Dashboard..."\n\
streamlit run dashboard.py --server.headless true --server.port 8501 --server.address 0.0.0.0 &\n\
echo "Dashboard started at http://localhost:8501"\n\
echo ""\n\
echo "Starting FastAPI..."\n\
python3 api_predictor.py &\n\
echo "API started at http://localhost:8000"\n\
echo ""\n\
echo "System is ready!"\n\
echo "Press Ctrl+C to stop all services"\n\
wait' > /app/start.sh && chmod +x /app/start.sh

# Set the default command
CMD ["/app/start.sh"]
