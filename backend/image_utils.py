import os
import numpy as np
from PIL import Image

def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """
    Normalize 16-bit/float32 raster data to uint8 using 2nd-98th percentile stretch.
    Avoids washed-out images from simple min-max scaling.
    """
    if array.dtype == np.uint8:
        return array
    
    # Calculate percentiles across all bands to maintain relative color balance
    p2, p98 = np.percentile(array, (2, 98))
    
    # Prevent division by zero
    if p98 == p2:
        return np.zeros_like(array, dtype=np.uint8)
        
    normalized = np.clip((array - p2) / (p98 - p2), 0, 1)
    return (normalized * 255.0).astype(np.uint8)

def load_image_universal(file_path: str) -> tuple[Image.Image, dict | None]:
    """
    Load any supported image format (PNG, JPG, TIFF, GeoTIFF, JP2) -> (PIL RGB uint8 image, geo_metadata or None).
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in ['.png', '.jpg', '.jpeg']:
        # Existing PIL flow
        pil_img = Image.open(file_path).convert("RGB")
        return pil_img, None
        
    elif ext in ['.tif', '.tiff', '.jp2']:
        # rasterio flow for GeoTIFF, standard TIFF, and JP2
        try:
            import rasterio
        except ImportError:
            raise ImportError("rasterio is required to load TIFF/GeoTIFF/JP2 files. Please install it.")
            
        with rasterio.open(file_path) as src:
            bands = src.count
            geo_meta = {
                "crs": src.crs.to_string() if src.crs else None,
                "transform": tuple(src.transform) if src.transform else None,
                "original_dtype": str(src.dtypes[0]),
                "band_count": bands,
                "width": src.width,
                "height": src.height,
            }
            
            # Read bands based on multi-band strategy
            if bands == 1:
                data = src.read(1)
                data = normalize_to_uint8(data)
                # Replicate to 3 channels for RGB
                img_array = np.stack([data, data, data], axis=-1)
            elif bands == 3:
                data = src.read([1, 2, 3])
                # rasterio reads as (bands, height, width), we need (height, width, bands)
                img_array = np.transpose(data, (1, 2, 0))
                img_array = normalize_to_uint8(img_array)
            else: # bands >= 4
                # Take first 3 bands as RGB
                data = src.read([1, 2, 3])
                img_array = np.transpose(data, (1, 2, 0))
                img_array = normalize_to_uint8(img_array)
                
            pil_img = Image.fromarray(img_array, mode="RGB")
            return pil_img, geo_meta
            
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
