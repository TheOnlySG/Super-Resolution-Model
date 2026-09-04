import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
from backend.inference import load_model, run_inference

app = FastAPI(title="RCAN Satellite Super-Resolution API")

# Add CORS middleware to allow the frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold model state
MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "rcan_improved.pth")

@app.on_event("startup")
async def startup_event():
    """Load the model when the API starts."""
    global MODEL
    print(f"Starting up... Loading model from {MODEL_PATH} onto {DEVICE}")
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"WARNING: Model file not found at {MODEL_PATH}")
            return
            
        MODEL, _, _ = load_model(MODEL_PATH, DEVICE)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return JSONResponse({
        "status": "ok",
        "model_loaded": MODEL is not None,
        "device": DEVICE
    })

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an uploaded image, runs it through the RCAN model,
    and returns the super-resolved high-resolution image.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read uploaded file
        image_bytes = await file.read()
        
        # Run inference
        hr_image_io = run_inference(MODEL, image_bytes, DEVICE)
        
        # Return HR image as streaming response
        return StreamingResponse(hr_image_io, media_type="image/png")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

# Mount the frontend directory to serve static files
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
