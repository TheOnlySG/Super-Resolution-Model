@echo off
setlocal enabledelayedexpansion

echo =========================================
echo    Satellite Imagery Super-Resolution
echo =========================================
echo.
echo How would you like to run the application?
echo [1] Docker (Only for Deployment , requires Docker installed and takes time)
echo [2] Python Virtual Environment (Recommended for local running,Requires Python installed)
echo.
set /p choice="Enter your choice (1 or 2): "

if "%choice%"=="1" (
    echo.
    docker compose version >nul 2>&1
    if !errorlevel! equ 0 (
        echo Building and starting Docker container with 'docker compose'...
        docker compose up --build
    ) else (
        where docker-compose >nul 2>&1
        if !errorlevel! equ 0 (
            echo Building and starting Docker container with 'docker-compose'...
            docker-compose up --build
        ) else (
            echo Docker Compose not found. Falling back to direct 'docker build' ^& 'docker run'...
            echo Building Docker image...
            docker build -t sih_sr_app .
            if !errorlevel! neq 0 (
                echo Docker build failed!
                pause
                exit /b !errorlevel!
            )
            echo.
            echo Starting Docker container on http://localhost:8000...
            echo Close this window or press Ctrl+C to stop.
            docker run --rm -p 8000:8000 sih_sr_app
        )
    )
) else if "%choice%"=="2" (
    echo.
    echo Setting up Python virtual environment...
    if not exist .venv (
        python -m venv .venv
    )
    call .venv\Scripts\activate.bat
    echo Installing dependencies...
    echo (Note: GeoTIFF support requires 'rasterio'. If pip fails to find a pre-compiled wheel, you may need a 64-bit Python version.^)
    pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo Failed to install dependencies!
        echo If rasterio failed, ensure you are using a standard 64-bit Python version from python.org.
        pause
        exit /b !errorlevel!
    )
    echo.
    echo Starting FastAPI server...
    set PYTHONUNBUFFERED=1
    set MODEL_PATH=model/RCAN_v2.pth
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
) else (
    echo Invalid choice. Exiting.
    pause
    exit /b 1
)
