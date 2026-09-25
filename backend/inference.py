import io
import os
import uuid
import torch
import numpy as np
from PIL import Image
import base64

from backend.model import RCAN, RCAN48
from backend.image_utils import load_image_universal

DEFAULT_NORMALIZATION = 3000.0
TEMP_DIR = os.path.join(os.path.dirname(__file__), "..", "temp_outputs")
os.makedirs(TEMP_DIR, exist_ok=True)

def load_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    normalization = checkpoint.get("normalization", DEFAULT_NORMALIZATION)
    model_name = checkpoint.get("model_name", "Unknown")
    
    if model_name == "RCAN v2":
        model = RCAN48(groups=6, blocks_per_group=8, channels=96)
    else:
        # rcan_improved.pth has incorrect metadata (20/128), must hardcode to 12/96 to match state_dict
        num_blocks = 12
        channels = 96
        model = RCAN(num_blocks=num_blocks, channels=channels)
        
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device, normalization

def array_to_base64_png(img_array):
    """Convert (H, W, 3) uint8 array to base64 PNG string."""
    pil_img = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def run_inference_pipeline(model, device, lr_path, normalization=DEFAULT_NORMALIZATION):
    """
    Executes the SR pipeline for standard RGB images using a 4-channel model adapter.
    """
    try:
        ext = os.path.splitext(lr_path)[1].lower()
        if ext in ['.tif', '.tiff']:
            from backend.image_utils import load_geotiff_raw, normalize_to_uint8
            # Load raw 4-band float32 data
            data, geo_meta = load_geotiff_raw(lr_path)
            width, height = geo_meta["width"], geo_meta["height"]
            orig_format = "TIFF"
            
            # Create a simple preview from RGB bands using natural order [0, 1, 2] and global stretch
            lr_rgb_raw = np.transpose(data[:3], (1, 2, 0))
            lr_preview_rgb = normalize_to_uint8(lr_rgb_raw)
            lr_preview = array_to_base64_png(lr_preview_rgb)
            
            lr_norm = data / normalization
            input_tensor = torch.tensor(lr_norm, dtype=torch.float32).unsqueeze(0).to(device)
            used_normalization = normalization
        else:
            pil_img, geo_meta = load_image_universal(lr_path)
            width, height = pil_img.size
            orig_format = pil_img.format or "PNG"
            
            lr_rgb = np.array(pil_img, dtype=np.float32)
            lr_preview = array_to_base64_png(lr_rgb.astype(np.uint8))
            
            lr_padded = np.zeros((height, width, 4), dtype=np.float32)
            lr_padded[:, :, :3] = lr_rgb
            lr_tensor_data = np.transpose(lr_padded, (2, 0, 1))
            
            lr_norm = lr_tensor_data / 255.0
            input_tensor = torch.tensor(lr_norm, dtype=torch.float32).unsqueeze(0).to(device)
            used_normalization = 255.0
    except Exception as e:
        raise ValueError(f"Image Error: {str(e)}")
        
    file_size = os.path.getsize(lr_path)

    # 3. Inference
    with torch.no_grad():
        output_tensor = model(input_tensor)
        
    # Raw tensor for validation (NO CLAMPING)
    sr_raw = output_tensor.squeeze(0).cpu().numpy()
    
    job_id = str(uuid.uuid4())
    
    # Save the full 4-band normalized tensor for downstream validation
    np.save(os.path.join(TEMP_DIR, f"{job_id}_sr_tensor.npy"), sr_raw)
    
    # 4. 4 -> 3 Channel Extraction for RGB visualization
    if ext in ['.tif', '.tiff']:
        # GeoTIFF path: model output is in /3000 space.
        # Use natural order [0, 1, 2] and global stretch to match /preview
        sr_rgb_raw = np.transpose(sr_raw[:3], (1, 2, 0))  # (H, W, 3)
        from backend.image_utils import normalize_to_uint8
        sr_rgb = normalize_to_uint8(sr_rgb_raw)
    else:
        # RGB path: model output is in [0, 1]. Simple × 255.
        sr_clamped = np.clip(sr_raw, 0, 1)
        sr_rgb_norm = sr_clamped[:3, :, :]
        sr_rgb = (np.transpose(sr_rgb_norm, (1, 2, 0)) * 255.0).astype(np.uint8)
        
    sr_height, sr_width, _ = sr_rgb.shape
    sr_preview = array_to_base64_png(sr_rgb)
    
    # 5. Save Output
    ext = os.path.splitext(lr_path)[1].lower()
    if not ext: ext = ".png"
    sr_out_path = os.path.join(TEMP_DIR, f"{job_id}{ext}")
    
    sr_pil = Image.fromarray(sr_rgb)
    
    if ext in ['.tif', '.tiff', '.jp2']:
        import rasterio
        from affine import Affine
        
        driver = 'GTiff' if ext in ['.tif', '.tiff'] else 'JP2OpenJPEG'
        
        transform = None
        crs = None
        if geo_meta and geo_meta.get("transform"):
            t = geo_meta["transform"]
            orig_transform = Affine(*t)
            # 4x scale means pixel size is 1/4th. 
            transform = orig_transform * Affine.scale(0.25, 0.25)
            crs = geo_meta.get("crs")
            
        with rasterio.open(
            sr_out_path, 'w',
            driver=driver,
            height=sr_height,
            width=sr_width,
            count=3,
            dtype=str(sr_rgb.dtype),
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(np.transpose(sr_rgb, (2, 0, 1)))
    elif ext in ['.jpg', '.jpeg']:
        sr_pil.save(sr_out_path, format="JPEG")
    else:
        sr_pil.save(sr_out_path, format="PNG")
    
    input_channels = int(data.shape[0]) if ext in ['.tif', '.tiff'] else 3
    output_channels = int(sr_raw.shape[0]) if ext in ['.tif', '.tiff'] else 3

    result = {
        "status": "success",
        "job_id": job_id,
        "metadata": {
            "input": {
                "filename": os.path.basename(lr_path),
                "width": width,
                "height": height,
                "channels": input_channels,
                "size_bytes": file_size,
                "format": orig_format
            },
            "output": {
                "width": sr_width,
                "height": sr_height,
                "channels": output_channels,
                "scale_factor": 4,
                "format": orig_format
            },
            "tensor": {
                "normalization": used_normalization,
                "is_multispectral": ext in ['.tif', '.tiff']
            }
        },
        "visualizations": {
            "lr_rgb": lr_preview,
            "sr_rgb": sr_preview
        },
        "geo_metadata": geo_meta,
        "normalization": used_normalization
    }
    
    import json
    with open(os.path.join(TEMP_DIR, f"{job_id}_meta.json"), "w") as f:
        json.dump(result["metadata"], f)
        
    # Return the result dict AND the SR PIL image for downstream use (e.g. segmentation)
    return result, sr_pil
