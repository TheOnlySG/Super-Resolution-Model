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
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "..", "model", "RCAN_v2.pth"))

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

@app.post("/preview")
async def preview(file: UploadFile = File(...)):
    """
    Convert any uploaded image to a displayable PNG base64 preview.
    Used by the frontend for formats browsers can't render (TIFF, GeoTIFF, JP2).
    """
    temp_path = os.path.join(TEMP_DIR, f"preview_{file.filename}")
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        from backend.image_utils import load_image_universal
        from backend.inference import array_to_base64_png
        import numpy as np
        
        pil_img, geo_meta = load_image_universal(temp_path)
        img_array = np.array(pil_img)
        preview_b64 = array_to_base64_png(img_array)
        
        return JSONResponse({
            "preview": preview_b64,
            "width": pil_img.width,
            "height": pil_img.height,
            "format": os.path.splitext(file.filename)[1].lower(),
            "bands": geo_meta["band_count"] if geo_meta else 3
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Preview error: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/download_image")
async def download_image(job_id: str):
    """Download the generated SR image."""
    import glob
    search_pattern = os.path.join(TEMP_DIR, f"{job_id}.*")
    matches = glob.glob(search_pattern)
    
    if not matches:
        raise HTTPException(status_code=404, detail="SR output not found or expired.")
        
    file_path = matches[0]
    ext = os.path.splitext(file_path)[1].lower()
    
    media_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.tif': 'image/tiff',
        '.tiff': 'image/tiff',
        '.jp2': 'image/jp2'
    }
    mtype = media_types.get(ext, 'application/octet-stream')
    
    return FileResponse(
        path=file_path, 
        filename=f"SR_output_{job_id[:8]}{ext}", 
        media_type=mtype
    )

@app.post("/validate")
async def validate(hr_file: UploadFile = File(...), job_id: str = Form(...)):
    """Run validation pipeline comparing SR tensor and HR reference."""
    sr_tensor_path = os.path.join(TEMP_DIR, f"{job_id}_sr_tensor.npy")
    if not os.path.exists(sr_tensor_path):
        raise HTTPException(status_code=404, detail="SR tensor not found or expired.")
        
    sr_meta_path = os.path.join(TEMP_DIR, f"{job_id}_meta.json")
    sr_meta = {}
    if os.path.exists(sr_meta_path):
        import json
        with open(sr_meta_path, "r") as f:
            sr_meta = json.load(f)
            
    hr_path = os.path.join(TEMP_DIR, f"hr_upload_{job_id}.tif")
    try:
        with open(hr_path, "wb") as buffer:
            import shutil
            shutil.copyfileobj(hr_file.file, buffer)
            
        from backend.validation import run_validation
        result = run_validation(sr_tensor_path, hr_path, sr_meta)
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=" ".join(result["errors"]))
            
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")
    finally:
        if os.path.exists(hr_path):
            os.remove(hr_path)

frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
