# AI Image Super-Resolution Workstation

A premium, deep learning-based image super-resolution application. This tool takes standard images (PNG, JPG, JPEG) and upscales them by a factor of 4× using an embedded RCAN (Residual Channel Attention Network) architecture originally trained for Sentinel-2 satellite imagery.

The application features a sleek, multi-modal web interface equipped with an interactive Side-by-Side viewer, Split View slider, and Overlay mode to seamlessly compare original and super-resolved results.

## Prerequisites
- Python 3.9+
- Pip (Python package manager)

## Getting Started Locally

We provide automated launcher scripts that make it incredibly easy to start the application using either **Docker** or a **Python Virtual Environment**.

### Option 1: Automated Launch (Easiest)

We've provided automated launcher scripts for Windows, Linux, and macOS. These scripts will prompt you to choose whether you want to run the application via **Docker** (recommended) or via a **Virtual Environment**.

**On Windows:**
Simply double-click the `run.bat` file in your File Explorer, or run it from the command prompt:
```cmd
run.bat
```

**On Linux / macOS:**
Open your terminal, navigate to the directory, and run the shell script:
```bash
./run.sh
```
*Note: The script will automatically build the Docker image or set up the Python `.venv` and install dependencies based on your choice.*

---

### Option 2: Manual Docker Setup

If you prefer to run Docker manually, the application is fully containerized in a single lightweight container (powered by CPU PyTorch for maximum compatibility).

1. **Build the Image**
   ```bash
   docker build -t srm-app .
   ```
2. **Run the Container**
   ```bash
   docker run --rm -p 8000:8000 srm-app
   ```

---

### Option 3: Manual Python Setup (Virtual Environment)

If you wish to set up the Python environment yourself:

1. **Create and Activate a Virtual Environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Start the API Server**
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```

## Usage

Once the server is running (via any of the methods above), open your web browser and navigate to:
**[http://localhost:8000](http://localhost:8000)**

1. Drop a PNG or JPG image into the upload zone.
2. Click "Start Super-Resolution" to process the image.
3. Use the interactive viewer (Side-by-Side, Split View, or Overlay) to compare the original image with the 4× super-resolved output.

## Architecture
- **Frontend**: Pure HTML, CSS, and JavaScript. Built for a premium, hardware-accelerated image viewing experience.
- **Backend**: FastAPI (Python). Automatically manages file uploads, pads 3-channel RGB imagery for the 4-channel model tensor, runs the RCAN inference, and returns the super-resolved RGB image.
- **Model**: `rcan_improved.pth` (located in `/model`).
