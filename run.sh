#!/bin/bash

echo "========================================="
echo "   Satellite Imagery Super-Resolution"
echo "========================================="
echo ""
echo "How would you like to run the application?"
echo "[1] Docker (Recommended, requires Docker installed)"
echo "[2] Python Virtual Environment (Requires Python installed)"
echo ""
read -p "Enter your choice (1 or 2): " choice

if [ "$choice" = "1" ]; then
    echo ""
    echo "Building Docker image..."
    docker build -t srm-app . || { echo "Docker build failed!"; exit 1; }
    echo ""
    echo "Starting Docker container on http://localhost:8000..."
    echo "Press Ctrl+C to stop."
    docker run --rm -p 8000:8000 srm-app
elif [ "$choice" = "2" ]; then
    echo ""
    echo "Setting up Python virtual environment..."
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv || python -m venv .venv
    fi
    source .venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt || { echo "Failed to install dependencies!"; exit 1; }
    echo ""
    echo "Starting FastAPI server..."
    export PYTHONUNBUFFERED=1
    export MODEL_PATH=model/rcan_improved.pth
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
else
    echo "Invalid choice. Exiting."
    exit 1
fi
