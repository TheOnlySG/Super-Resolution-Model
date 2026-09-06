import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import torch

from backend.inference import load_model, run_inference_pipeline, TEMP_DIR

app = FastAPI(title="RCAN Satellite Super-Resolution Workstation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
SEG_MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "..", "model", "rcan_improved.pth"))

@app.on_event("startup")
async def startup_event():
    global MODEL, SEG_MODEL
    
    # --- Load SR model ---
    print(f"Starting up... Loading SR model from {MODEL_PATH} onto {DEVICE}")
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"WARNING: SR model file not found at {MODEL_PATH}")
        else:
            MODEL, _, _ = load_model(MODEL_PATH, DEVICE)
            print("SR model loaded successfully.")
    except Exception as e:
        print(f"Error loading SR model: {e}")

    # --- Load Segmentation model (independent — failure does not block SR) ---
    try:
        from model.segmentation.seg_model import load_segmentation_model
        SEG_MODEL, _ = load_segmentation_model(DEVICE)
        print("Segmentation model loaded successfully.")
    except Exception as e:
        print(f"WARNING: Segmentation model failed to load: {e}")
        SEG_MODEL = None

@app.get("/health")
async def health_check():
    return JSONResponse({
        "status": "ok",
        "sr_model_loaded": MODEL is not None,
        "seg_model_loaded": SEG_MODEL is not None,
        "device": DEVICE
    })

@app.post("/predict")
async def predict(lr_file: UploadFile = File(...)):
    """
    Accepts LR image (PNG/JPG).
    Returns JSON with stats, base64 previews, and segmentation results.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    lr_path = os.path.join(TEMP_DIR, f"upload_{lr_file.filename}")
    
    try:
        with open(lr_path, "wb") as buffer:
            shutil.copyfileobj(lr_file.file, buffer)
                
        result, sr_pil = run_inference_pipeline(MODEL, DEVICE, lr_path)
        
        # --- Run segmentation on SR output ---
        if SEG_MODEL is not None:
            try:
                from model.segmentation.inference import run_segmentation
                seg_result = run_segmentation(SEG_MODEL, DEVICE, sr_pil)
                result["segmentation"] = seg_result
            except Exception as seg_e:
                import traceback
                traceback.print_exc()
                result["segmentation"] = {
                    "available": False,
                    "error": f"Segmentation failed: {str(seg_e)}"
                }
        else:
            result["segmentation"] = {
                "available": False,
                "error": "Segmentation model not loaded."
            }
        
        return JSONResponse(content=result)
        
    except ValueError as ve:
        print(f"ValueError in /predict: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        # Cleanup uploaded files
        if os.path.exists(lr_path):
            os.remove(lr_path)

@app.get("/download_image")
async def download_image(job_id: str):
    """Download the generated SR PNG."""
    file_path = os.path.join(TEMP_DIR, f"{job_id}.png")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="SR output not found or expired.")
    
    return FileResponse(
        path=file_path, 
        filename=f"SR_output_{job_id[:8]}.png", 
        media_type="image/png"
    )

frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
