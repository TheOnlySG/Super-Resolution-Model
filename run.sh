#!/bin/bash

echo "========================================="
echo "   Satellite Imagery Super-Resolution"
echo "========================================="
echo ""
echo "How would you like to run the application?"
echo "[1] Docker Compose (Recommended, requires Docker installed)"
echo "[2] Python Virtual Environment (Requires Python installed)"
echo ""
read -p "Enter your choice (1 or 2): " choice

if [ "$choice" = "1" ]; then
    echo ""
    if docker compose version &> /dev/null; then
        echo "Building and starting Docker container with 'docker compose'..."
        docker compose up --build || { echo "Docker Compose failed!"; exit 1; }
    elif command -v docker-compose &> /dev/null; then
        echo "Building and starting Docker container with 'docker-compose'..."
        docker-compose up --build || { echo "Docker Compose failed!"; exit 1; }
    elif command -v docker &> /dev/null; then
        echo "Docker Compose not found. Falling back to direct 'docker build' & 'docker run'..."
        docker build -t sih_sr_app . && docker run --rm -p 8000:8000 sih_sr_app || { echo "Docker run failed!"; exit 1; }
    else
        echo "Error: Docker is not installed or not available in PATH."
        exit 1
    fi
elif [ "$choice" = "2" ]; then
    echo ""
    echo "Setting up Python virtual environment..."
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv || python -m venv .venv
    fi
    source .venv/bin/activate
    echo "Installing dependencies..."
    echo "Note: GeoTIFF support uses 'rasterio'. If installation fails, you may need system GDAL libraries:"
    echo "      sudo apt-get install libgdal-dev gdal-bin"
    pip install -r requirements.txt || { echo "Failed to install dependencies!"; exit 1; }
    echo ""
    echo "Starting FastAPI server..."
    export PYTHONUNBUFFERED=1
    export MODEL_PATH=model/RCAN_v2.pth
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
else
    echo "Invalid choice. Exiting."
    exit 1
fi
