#!/bin/bash

# Script to enable W&B in Docker
echo "Enabling W&B in Docker..."

# Check if wandb is installed
if ! python3 -c "import wandb" 2>/dev/null; then
    echo "Installing wandb..."
    pip install wandb
fi

# Set environment variable to enable wandb
export WANDB_MODE=run

echo "W&B enabled. You can now run:"
echo "docker-compose up --build"
echo ""
echo "Or to enable W&B in an existing container:"
echo "docker-compose run -e WANDB_MODE=run readmission-prediction" 