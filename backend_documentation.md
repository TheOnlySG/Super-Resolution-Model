# Backend System Architecture & Data Flow Documentation

This document provides a comprehensive technical overview of the backend architecture, module responsibilities, API endpoint specifications, and end-to-end data flow lifecycles for the **RCAN Satellite Imagery Super-Resolution Workstation**.

---

## 1. System Architecture Overview

The backend is built with **FastAPI** (Python 3.10+) and **PyTorch**, serving as a high-performance RESTful API for satellite imagery super-resolution (4x scaling), semantic land-cover segmentation, and reference-based multispectral validation & quality assessment.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                     FRONTEND UI                                         │
│                      (Vanilla JS / HTML5 Canvas / CSS Design System)                    │
└────────────────────────────┬───────────────────────────────┬────────────────────────────┘
                             │                               │
                      HTTP REST Requests               HTTP REST Requests
                             │                               │
┌────────────────────────────▼───────────────────────────────▼────────────────────────────┐
│                                   FASTAPI ROUTER                                        │
│                                  (backend/main.py)                                      │
├───────────────────┬──────────────────┬───────────────────┬──────────────────────────────┤
│  POST /predict    │  POST /validate  │  POST /preview    │  GET /download_image         │
└─────────┬─────────┴────────┬─────────┴─────────┬─────────┴──────────────┬───────────────┘
          │                  │                   │                        │
┌─────────▼─────────┐┌───────▼─────────┐┌────────▼─────────┐     ┌────────▼────────┐
│ Super-Resolution  ││ Multispectral   ││ Format Convert   │     │ Output Delivery │
│    Inference      ││   Validation    ││    Preview       │     │   File Stream   │
│(backend/inference)││(backend/valid.) ││(backend/img_util)│     │  (TEMP_DIR)     │
└─────────┬─────────┘└───────┬─────────┘└──────────────────┘     └─────────────────┘
          │                  │
┌─────────▼─────────┐┌───────▼─────────┐
│ RCAN Model (4x)   ││ 4-Band Metrics  │
│ (backend/model)   ││  & Colormaps    │
└─────────┬─────────┘└─────────────────┘
          │
┌─────────▼─────────┐
│ Land Cover Seg.   │
│ (SegFormer Model) │
└───────────────────┘
```

### Core Design Principles & Contracts
1. **Multispectral-First Engineering**: The super-resolution network processes full 4-band Sentinel-2 data ($\text{B2}=\text{Blue}, \text{B3}=\text{Green}, \text{B4}=\text{Red}, \text{B8}=\text{NIR}$). Standard 3-band RGB inputs are automatically padded to 4 channels for inference.
2. **Strict Normalization Segregation**:
   - **GeoTIFF / Multispectral inputs**: Scaled by constant factor `3000.0` ($\text{Reflectance} = \text{DN} / 3000.0$). Unclamped float32 values are preserved for accurate scientific metrics.
   - **RGB (PNG/JPG) inputs**: Scaled by `255.0` to $[0, 1]$.
3. **Decoupled Architecture**: Super-resolution (SR) and land cover segmentation run in isolated, sequential stages. A failure in land-cover segmentation does not block or fail the SR generation.
4. **Geospatial Metadata Preservation**: GeoTIFF spatial references ($\text{CRS}$ and $\text{Affine}$ transform) are parsed via `rasterio`. Output GeoTIFFs maintain original spatial bounds while updating pixel transform scales ($\text{Affine} \times 0.25$).

---

## 2. Directory & Module Reference

```
backend/
├── main.py          # FastAPI web server, routes, CORS middleware, startup lifecycle
├── inference.py     # Super-resolution pipeline, tensor normalization, metadata caching
├── model.py         # RCAN architecture definition (4-channel input/output, RCAB blocks)
├── validation.py    # Multispectral validation engine (PSNR, SSIM, SAM, NDVI Error, Maps)
└── image_utils.py   # Universal raster loader, GeoTIFF raw loader, percentile stretch
model/
└── segmentation/    # SegFormer Land Cover classification model & inference module
```

### Module Breakdown

#### `backend/main.py`
- **Role**: Application entrypoint and HTTP route controller.
- **Key Responsibilities**:
  - Initializes FastAPI app with permissive CORS policy for local frontend integration.
  - Loads models on startup (`startup_event`):
    - Loads SR model onto GPU (`cuda`) if available, falling back to `cpu`.
    - Loads SegFormer model safely (catches errors if missing or uninitialized).
  - Implements route handlers for `/health`, `/predict`, `/preview`, `/download_image`, and `/validate`.
  - Mounts `frontend/` directory to serve static UI assets at `/`.

#### `backend/model.py`
- **Role**: Deep Learning Model Architecture for Super-Resolution.
- **Key Components**:
  - **`RCAB` (Residual Channel Attention Block)**: Combines standard $3\times3$ convolutions, ReLU activation, global average pooling, and channel attention mechanism (reduction ratio 16) with a residual skip connection.
  - **`RCAN v2` (Residual Channel Attention Network 48-block)**:
    - `head`: Conv2d mapping $4 \to 96$ feature channels.
    - `body`: Stack of 6 `ResidualGroup` modules (each containing 8 `RCAB` blocks, totaling 48 blocks) followed by a $3\times3$ body convolution and global residual connection. Each group has its own local residual skip.
    - `up1` & `up2`: Two sequential $2\times$ upsampling modules using `PixelShuffle(2)` for a total $4\times$ spatial expansion.
    - `tail`: Conv2d mapping $96 \to 4$ output channels ($\text{B2}, \text{B3}, \text{B4}, \text{B8}$).

#### `backend/inference.py`
- **Role**: Super-Resolution Execution & Asset Lifecycle Engine.
- **Key Responsibilities**:
  - `load_model()`: Deserializes PyTorch checkpoint `RCAN_v2.pth` ($48$ blocks, $96$ channels, default normalization $3000.0$).
  - `run_inference_pipeline()`:
    1. Determines file type (`.tif`/`.tiff` vs RGB).
    2. Loads raster data and metadata.
    3. Normalizes input data ($/3000.0$ for GeoTIFF, $/255.0$ for RGB).
    4. Runs PyTorch model inference under `torch.no_grad()`.
    5. Saves full raw 4-band float32 tensor (`{job_id}_sr_tensor.npy`) in `temp_outputs/` without clamping for validation.
    6. Extracts RGB representation for UI visualization (applies 2nd-98th percentile stretch for GeoTIFFs to prevent washed-out display).
    7. Generates scaled GeoTIFF output using `rasterio` if input was geospatial, or PNG/JPG image otherwise.
    8. Persists job metadata to `{job_id}_meta.json`.

#### `backend/image_utils.py`
- **Role**: Universal Raster I/O & Preprocessing Utilities.
- **Key Responsibilities**:
  - `normalize_to_uint8()`: Computes global 2nd-98th percentile stretch across raster bands to convert 16-bit or float32 satellite images to standard 8-bit RGB without clipping or saturation.
  - `load_image_universal()`: Handles PNG, JPG, TIF, GeoTIFF, and JP2 formats. Returns PIL RGB Image and geospatial metadata dict (`crs`, `transform`, `width`, `height`, `band_count`).
  - `load_geotiff_raw()`: Reads raw 4-band float32 data from GeoTIFF files using `rasterio`.

#### `backend/validation.py`
- **Role**: Multispectral Scientific Validation Engine.
- **Key Responsibilities**:
  - `validate_alignment()`: Verifies HR reference image matches SR output in spatial dimensions ($W \times H$) and channel count ($\ge 4$ bands). Rejects non-multispectral SR jobs.
  - Metrics & Error Maps:
    - **Reconstruction Error Map**: Per-pixel Mean Absolute Error across $\text{B2}, \text{B3}, \text{B4}, \text{B8}$.
    - **Reference-Based Confidence Map**: Robust min-max normalized error mapped via $1.0 - \text{Error}$, rendered with `RdYlGn` colormap (Red = low confidence, Green = high confidence).
    - **Spectral Angle Mapper (SAM)**: Angular spectral distance in radians:
      $$\theta = \arccos\left(\frac{\mathbf{r} \cdot \mathbf{t}}{\|\mathbf{r}\|_2 \|\mathbf{t}\|_2 + \epsilon}\right)$$
    - **NDVI Error Map**: Difference in Normalized Difference Vegetation Index between SR and HR:
      $$\text{NDVI} = \frac{\text{B8} - \text{B4}}{\text{B8} + \text{B4} + \epsilon}$$
    - **PSNR & SSIM**: Calculated in raw normalized reflectance space across all 4 bands.
  - `colorize_map()`: Applies Matplotlib colormaps (`inferno`, `RdYlGn`, `magma`) to single-channel error maps to produce 8-bit RGB PNGs.

#### `model/segmentation/`
- **Role**: Land Cover Classification (SegFormer HuggingFace Model).
- **Key Responsibilities**:
  - SegFormer pretrained model configured for 6 land cover classes (Water, Artificial Bare Ground, Building, High Vegetation, Low Vegetation, Bare Ground).
  - Resizes input image to $512\times512$, normalizes with ImageNet stats, interpolates output logits back to native SR dimensions.
  - Calculates class distribution percentages (excluding ignore class ID 0) and generates colored mask and translucent overlay images ($55\%$ SR + $45\%$ Mask).

---

## 3. API Endpoint Specifications

| Endpoint | Method | Input Parameters | Output Response | Description |
| :--- | :--- | :--- | :--- | :--- |
| `/health` | `GET` | None | `JSON` | Checks server health and model load state. |
| `/preview` | `POST` | `file`: UploadFile | `JSON` (base64 preview, width, height, format, bands) | Converts GeoTIFF/TIFF/JP2 to displayable PNG preview for browser rendering before SR generation. |
| `/predict` | `POST` | `lr_file`: UploadFile | `JSON` (job_id, metadata, base64 SR & LR previews, land cover segmentation) | Runs 4x RCAN Super-Resolution + SegFormer Land Cover Segmentation. |
| `/validate` | `POST` | `hr_file`: UploadFile<br>`job_id`: Form string | `JSON` (alignment status, metrics, base64 encoded error maps) | Compares cached SR tensor with uploaded HR reference GeoTIFF. |
| `/download_image` | `GET` | `job_id`: Query string | `FileResponse` (binary file stream) | Downloads output SR image in original file format (`.tif`, `.png`, `.jpg`). |

---

## 4. End-to-End Site Work Stage & Data Flows

### Stage 1: Image Upload & Super-Resolution Generation Flow

This flow triggers when the user selects or drops a Low-Resolution (LR) satellite image on the main workstation view.

```
[ User Upload (LR Image) ]
          │
          ▼
   Frontend app.js
          │
          │ HTTP POST /predict (Multipart FormData: lr_file)
          ▼
   backend/main.py -> predict()
          │
          ├─► Write file to temp_outputs/upload_{filename}
          │
          ▼
   backend/inference.py -> run_inference_pipeline()
          │
          ├─► File Extension Check:
          │     ├── GeoTIFF (.tif/.tiff):
          │     │     ├─► load_geotiff_raw() -> Float32 array (4, H, W)
          │     │     ├─► Normalize: data / 3000.0
          │     │     └─► Percentile stretch (2nd-98th) -> LR preview Base64
          │     │
          │     └── RGB (.png/.jpg):
          │           ├─► load_image_universal() -> PIL Image
          │           ├─► Pad 3 -> 4 channels (zeros in 4th channel)
          │           ├─► Normalize: array / 255.0
          │           └─► Direct uint8 -> LR preview Base64
          │
          ├─► PyTorch Neural Forward Pass:
          │     └─► output_tensor = RCAN_MODEL(input_tensor)  [Shape: (1, 4, H*4, W*4)]
          │
          ├─► Raw Tensor Persistence:
          │     └─► Save full float32 tensor -> temp_outputs/{job_id}_sr_tensor.npy (NO CLAMPING)
          │
          ├─► Output File Construction:
          │     ├── GeoTIFF: Compute new Affine transform (scale 0.25x), write 3-band RGB TIFF
          │     └── PNG/JPG: Save PIL Image to temp_outputs/{job_id}.ext
          │
          ├─► Save Metadata:
          │     └─► JSON dump -> temp_outputs/{job_id}_meta.json
          │
          ▼
   Land Cover Segmentation Check (main.py):
          │
          ├── If SEG_MODEL loaded:
          │     ├─► run_segmentation(SEG_MODEL, sr_pil)
          │     ├─► SegFormer inference -> logits argmax mask
          │     ├─► Generate colored mask & translucent overlay (55% SR + 45% Mask)
          │     └─► Calculate land-cover class percentages
          │
          └── If SEG_MODEL missing:
                └─► Attach available: false in response JSON
          │
          ▼
   HTTP 200 JSON Response to Frontend:
   {
     "status": "success",
     "job_id": "...",
     "metadata": { ... },
     "visualizations": { "lr_rgb": "data:image/png;base64...", "sr_rgb": "data:image/png;base64..." },
     "segmentation": { "available": true, "mask": "...", "overlay": "...", "classes": [...] }
   }
          │
          ▼
   Frontend app.js Render:
          ├─► Display LR and SR images on interactive pan/zoom split viewers
          ├─► Render Metadata Cards & Resolution stats
          ├─► Render Land Cover Distribution pie chart & Legend
          └─► Store job_id in localStorage for download & validation pages
```

---

### Stage 2: Validation & Quality Assessment Flow

This flow occurs on the **Validation & Quality Assessment** page when the user provides a High-Resolution (HR) reference GeoTIFF for an existing `job_id`.

```
[ User Navigates to Validation Page ]
          │
          ├─► Retrieve job_id from localStorage
          │
[ User Uploads HR Reference GeoTIFF ]
          │
          ▼
   Frontend validation.js
          │
          │ HTTP POST /validate (FormData: hr_file, job_id)
          ▼
   backend/main.py -> validate()
          │
          ├─► Verify SR files exist:
          │     ├─► temp_outputs/{job_id}_sr_tensor.npy
          │     └─► temp_outputs/{job_id}_meta.json
          │
          ▼
   backend/validation.py -> run_validation()
          │
          ├─► Load raw SR tensor (4, H_sr, W_sr) float32
          ├─► Read HR GeoTIFF (rasterio.read([1, 2, 3, 4]))
          │
          ├─► Alignment & Compatibility Check (validate_alignment):
          │     ├── Check 1: Ensure SR was generated from 4-band GeoTIFF (is_multispectral == true)
          │     ├── Check 2: HR band count >= 4
          │     └── Check 3: HR spatial dimensions == SR spatial dimensions (W_hr == W_sr, H_hr == H_sr)
          │     (If failed: Raise HTTP 400 with detailed error message)
          │
          ├─► Normalize HR: hr_raw / 3000.0
          │
          ├─► Scientific Metric & Error Map Computation:
          │     ├── Reconstruction Error Map: Mean absolute error across B2/B3/B4/B8
          │     ├── Confidence Map: 1.0 - robust_normalize(Reconstruction_Error)
          │     ├── SAM Map: Spectral Angle Mapper per-pixel angle theta (radians)
          │     ├── NDVI Error Map: |NDVI_sr - NDVI_hr| where NDVI = (B8 - B4) / (B8 + B4)
          │     └── PSNR & SSIM: Computed in normalized reflectance space across 4 bands
          │
          ├─► Colormap Visualization Rendering (Matplotlib):
          │     ├── Reconstruction Error -> inferno colormap
          │     ├── Confidence Map -> RdYlGn colormap (Red=Low, Green=High)
          │     ├── SAM Error Map -> magma colormap
          │     └── NDVI Error Map -> magma colormap
          │
          ├─► Base64 PNG Encoding of maps
          │
          ▼
   HTTP 200 JSON Response to Frontend:
   {
     "status": "success",
     "maps": { "reconstruction_error": "...", "confidence": "...", "sam_error": "...", "ndvi_error": "..." },
     "metrics": { "psnr": 34.2, "ssim": 0.91, "sam_mean": 0.08, "ndvi_mean_error": 0.03, ... },
     "band_info": { "order": ["B2", "B3", "B4", "B8"], "normalization": 3000.0 }
   }
          │
          ▼
   Frontend validation.js Render:
          ├─► Display active error map on high-resolution canvas with pan/zoom
          ├─► Tab bar switching between Reconstruction Error, Confidence, SAM, and NDVI maps
          ├─► Dynamic metric values rendering (PSNR, SSIM, SAM deg, NDVI error, Confidence %)
          └─► Contextual explanation cards updating based on active map selection
```

---

### Stage 3: Image Format Preview Conversion Flow

Used when a user selects a file format that web browsers cannot natively display (such as 16-bit GeoTIFF, JP2, or multi-band TIF) prior to triggering super-resolution.

```
[ User selects GeoTIFF / JP2 file ]
          │
          ▼
   Frontend app.js -> POST /preview
          │
          ▼
   backend/main.py -> preview()
          │
          ├─► Save to temp_outputs/preview_{filename}
          ├─► backend/image_utils.py -> load_image_universal()
          │     ├─► Reads raster bands via rasterio
          │     └─► Applies 2nd-98th percentile contrast stretch (normalize_to_uint8)
          ├─► Encode 8-bit RGB array to Base64 PNG
          │
          ▼
   HTTP 200 JSON Response: { "preview": "data:image/png;base64...", "width": 512, "height": 512, "bands": 4 }
          │
          ▼
   Frontend renders preview image in upload dropzone container
```

---

### Stage 4: High-Resolution SR Asset Download Flow

Triggered when the user clicks the "Download Super-Resolved Image" button.

```
[ User clicks Download Button ]
          │
          ▼
   Frontend triggers direct navigation / window download link:
   GET /download_image?job_id={job_id}
          │
          ▼
   backend/main.py -> download_image()
          │
          ├─► Search temp_outputs/ for matching file pattern: {job_id}.*
          │     ├── Found: temp_outputs/{job_id}.tif
          │     └── Not Found: Raise HTTP 404 ("SR output expired or not found")
          │
          ├─► Detect MIME Type (.tif -> image/tiff, .png -> image/png, .jp2 -> image/jp2)
          │
          ▼
   HTTP 200 FileResponse Stream:
   Content-Disposition: attachment; filename="SR_output_{job_id[:8]}.tif"
```

---

## 5. Summary Matrix of File Lifecycles & Normalization

| Asset Type | Storage Path | Format / Data Type | Normalization | Lifespan / Cleanup |
| :--- | :--- | :--- | :--- | :--- |
| **LR Upload** | `temp_outputs/upload_{filename}` | Raw File | N/A | Removed in `finally` block of `/predict` |
| **SR Raw Tensor** | `temp_outputs/{job_id}_sr_tensor.npy` | NumPy `float32` array `(4, H, W)` | Unclamped `/3000.0` or `/255.0` | Retained for downstream `/validate` requests |
| **SR Output Image** | `temp_outputs/{job_id}.{ext}` | GeoTIFF / PNG / JPG | 8-bit RGB | Retained for direct download via `/download_image` |
| **Job Metadata** | `temp_outputs/{job_id}_meta.json` | JSON text | N/A | Retained for validation compatibility checks |
| **HR Validation Upload** | `temp_outputs/hr_upload_{job_id}.tif` | Raw GeoTIFF | `/3000.0` | Removed in `finally` block of `/validate` |
