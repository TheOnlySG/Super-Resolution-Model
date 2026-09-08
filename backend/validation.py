import os
import io
import base64
import numpy as np

def validate_alignment(sr_meta, hr_src):
    """
    Checks that the HR GeoTIFF is compatible with the SR output.
    Returns (is_valid: bool, errors: list[str], warnings: list[str])
    """
    errors = []
    warnings = []
    
    # 1. Band count
    if hr_src.count != 4:
        errors.append(f"HR must have exactly 4 bands. Found {hr_src.count}.")
        
    # 2. Spatial dimensions
    if sr_meta and "output" in sr_meta:
        w_sr = sr_meta["output"].get("width")
        h_sr = sr_meta["output"].get("height")
        if hr_src.width != w_sr or hr_src.height != h_sr:
            errors.append(f"Spatial dimension mismatch. SR is {w_sr}x{h_sr}, HR is {hr_src.width}x{hr_src.height}.")
            
    return len(errors) == 0, errors, warnings


def compute_reconstruction_error(sr: np.ndarray, hr_norm: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Per-pixel mean absolute error across B2/B3/B4/B8.
    Returns: (error_map (H,W), mean_error scalar)
    """
    abs_error = np.abs(sr - hr_norm)
    error_map = np.mean(abs_error, axis=0)
    return error_map, float(np.mean(error_map))


def compute_confidence(error_map: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Reference-based confidence: 1 - robust_normalize(error).
    """
    lo, hi = np.percentile(error_map, [2, 98])
    normalized = np.clip((error_map - lo) / (hi - lo + 1e-8), 0, 1)
    confidence = 1.0 - normalized
    return confidence, float(np.mean(confidence))


def compute_sam(sr: np.ndarray, hr_norm: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Spectral Angle Mapper per pixel (radians).
    """
    sr_vec = np.moveaxis(sr, 0, -1)    # (H,W,4)
    hr_vec = np.moveaxis(hr_norm, 0, -1)
    dot = np.sum(sr_vec * hr_vec, axis=-1)
    sr_mag = np.linalg.norm(sr_vec, axis=-1)
    hr_mag = np.linalg.norm(hr_vec, axis=-1)
    sam_map = np.arccos(np.clip(dot / (sr_mag * hr_mag + 1e-8), -1, 1))
    return sam_map, float(np.mean(sam_map))


def compute_ndvi_error(sr: np.ndarray, hr_norm: np.ndarray) -> tuple[np.ndarray, float]:
    """
    NDVI = (B8 - B4) / (B8 + B4 + eps). Band indices: B4=2, B8=3.
    """
    eps = 1e-8
    ndvi_sr = (sr[3] - sr[2]) / (sr[3] + sr[2] + eps)
    ndvi_hr = (hr_norm[3] - hr_norm[2]) / (hr_norm[3] + hr_norm[2] + eps)
    ndvi_error = np.abs(ndvi_sr - ndvi_hr)
    return ndvi_error, float(np.mean(ndvi_error))


def compute_psnr_ssim(sr: np.ndarray, hr_norm: np.ndarray) -> tuple[float, float]:
    """
    PSNR: 20*log10(1.0 / RMSE) in normalized space (peak=1.0).
    SSIM: computed per-band, averaged. Uses skimage.
    """
    mse = np.mean((sr - hr_norm) ** 2)
    rmse = np.sqrt(mse)
    psnr = 20 * np.log10(1.0 / (rmse + 1e-8))
    
    from skimage.metrics import structural_similarity
    ssim_vals = []
    for b in range(sr.shape[0]):
        # Data range is 1.0 because values are typically around 0-1.5, normalized relative to 1.0 peak
        # In fact, data_range depends on min/max but 1.0 is a reasonable default for reflectance.
        # errormapping.ipynb uses default data_range or infers it. We use data_range=1.0 for consistency.
        s = structural_similarity(sr[b], hr_norm[b], data_range=1.0)
        ssim_vals.append(s)
    ssim = float(np.mean(ssim_vals))
    return float(psnr), ssim


def colorize_map(map_data, cmap_name, vmin=None, vmax=None):
    """Convert a float32 (H,W) map to (H,W,3) uint8 using a matplotlib colormap."""
    import matplotlib as mpl
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=vmin if vmin is not None else map_data.min(), vmax=vmax if vmax is not None else map_data.max())
    cmap = mpl.colormaps[cmap_name]
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    rgba = mapper.to_rgba(map_data, bytes=True)  # (H,W,4) uint8
    return rgba[:, :, :3]  # Drop alpha


def array_to_base64_png(img_array):
    """Convert (H, W, 3) uint8 array to base64 PNG string."""
    from PIL import Image
    pil_img = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_validation(sr_tensor_path: str, hr_path: str, sr_meta: dict, normalization: float = 3000.0):
    """
    Full validation pipeline. Produces maps + metrics.
    sr_tensor_path: path to the .npy file with (4, H, W) float32 SR output
    hr_path: path to the 4-band HR GeoTIFF
    """
    import rasterio
    
    # 1. Load HR
    with rasterio.open(hr_path) as hr_src:
        is_valid, errors, warnings = validate_alignment(sr_meta, hr_src)
        if not is_valid:
            return {
                "status": "error",
                "errors": errors,
                "warnings": warnings
            }
        
        # Read bands 1,2,3,4
        hr_raw = hr_src.read([1, 2, 3, 4]).astype(np.float32)
        
    # 2. Normalize HR
    hr_norm = hr_raw / normalization
    
    # 3. Load SR
    sr = np.load(sr_tensor_path)
    
    # 4. Compute metrics
    error_map, recon_mean = compute_reconstruction_error(sr, hr_norm)
    confidence_map, conf_mean = compute_confidence(error_map)
    sam_map, sam_mean = compute_sam(sr, hr_norm)
    ndvi_map, ndvi_mean = compute_ndvi_error(sr, hr_norm)
    psnr, ssim = compute_psnr_ssim(sr, hr_norm)
    
    mse = np.mean((sr - hr_norm) ** 2)
    rmse = np.sqrt(mse)
    
    # 5. Colorize maps
    recon_img = colorize_map(error_map, "inferno")
    conf_img = colorize_map(confidence_map, "Blues", vmin=0, vmax=1)
    sam_img = colorize_map(sam_map, "magma")
    ndvi_img = colorize_map(ndvi_map, "magma")
    
    # 6. Encode maps
    maps = {
        "reconstruction_error": array_to_base64_png(recon_img),
        "confidence": array_to_base64_png(conf_img),
        "sam_error": array_to_base64_png(sam_img),
        "ndvi_error": array_to_base64_png(ndvi_img)
    }
    
    metrics = {
        "psnr": float(psnr),
        "ssim": float(ssim),
        "sam_mean": float(sam_mean),
        "sam_mean_degrees": float(np.degrees(sam_mean)),
        "ndvi_mean_error": float(ndvi_mean),
        "reconstruction_mean_error": float(recon_mean),
        "confidence_mean": float(conf_mean),
        "rmse": float(rmse)
    }
    
    return {
        "status": "success",
        "validation": {
            "alignment": {
                "valid": is_valid,
                "warnings": warnings
            }
        },
        "maps": maps,
        "metrics": metrics,
        "band_info": {
            "order": ["B2 (Blue)", "B3 (Green)", "B4 (Red)", "B8 (NIR)"],
            "count": 4,
            "normalization": normalization
        },
        "dimensions": {
            "width": hr_norm.shape[2],
            "height": hr_norm.shape[1]
        }
    }
