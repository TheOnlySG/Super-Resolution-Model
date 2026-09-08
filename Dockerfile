FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if required by Pillow/OpenCV etc (usually libgl1, libglib2.0 for opencv, but we only use Pillow which is fine on slim)
# Actually, slim doesn't have some build tools, but the wheels for torch and Pillow are prebuilt for slim.

# Install GDAL system dependencies for rasterio (GeoTIFF/JP2 support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Ensure temp_outputs exists
RUN mkdir -p temp_outputs

# Expose port
EXPOSE 8000

# Set environment variables
ENV MODEL_PATH=/app/model/rcan_improved.pth
ENV PYTHONUNBUFFERED=1

# Start the FastAPI app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
