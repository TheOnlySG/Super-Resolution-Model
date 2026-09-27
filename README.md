# Beyond Resolutions — Every Pixel Matters

**SIH 2026 · Problem Statement 26142**  
End-to-end 4× Multispectral Satellite Imagery Super-Resolution, Land Cover Segmentation, and Scientific Quality Validation.

---

## What We Built

A production-ready **AI workstation** that takes a low-resolution (10m) Sentinel-2 satellite image and generates a **2.5m effective-resolution** super-resolved output — exceeding the 4m target resolution stated in the problem statement.

Beyond the model, we built a full **web-based demonstration platform** with three capabilities:

| Capability | What it does |
|---|---|
| **Super-Resolution** | 4× upscaling using a custom 4-band RCAN model preserving geospatial metadata |
| **Land Cover Analysis** | 7-class SegFormer-B2 segmentation run on the SR output to prove spatial/spectral integrity |
| **Quality Validation** | Reference-based scientific engine generating 4 error heatmaps + PSNR / SSIM metrics |

---

## Our USPs

**1. Output resolution of 2.5m — problem required 4m.**  
We deliver sharper output than the stated requirement.

**2. V2 trained on only 1% of available data — and it still performs well.**  
This demonstrates the data efficiency and generalization capability of our architecture. Full dataset training in V3 is the clear path to further gains.

**3. We have a live, deployable platform to demonstrate — not just a model.**  
Any evaluator can run the full end-to-end pipeline locally in seconds with a single script.

---

## Model Evolution

Our solution evolved through deliberate, incremental iterations:

### V1 — Prototype (RCAN Baseline)

| Attribute | Detail |
|---|---|
| Architecture | RCAN — 12 RCAB blocks, 96 channels |
| Training data | ~100 image patches (manual curation) |
| Input channels | 4-band (B2, B3, B4, B8) |
| Purpose | Proof of concept; validate pipeline feasibility |
| Key output | Confirmed the architecture handles multispectral 4-band float32 data end-to-end |

### V2 — Scale-Up (RCAN v2, Current Model)

| Attribute | Detail |
|---|---|
| Architecture | RCAN v2 — **48 RCAB blocks**, 96 channels, 6 Residual Groups |
| Training data | Sentinel-2 L2A tiles — **~1% of the available corpus** |
| Normalization | Float32 division by `3000.0` (unclamped, preserves reflectance) |
| Upsampling | Dual PixelShuffle(2) → 4× spatial expansion |
| **PSNR** | **32.62 dB** |
| **SSIM** | **0.8731** |
| Output resolution | **2.5m effective** (input: 10m Sentinel-2) |

> V2 achieves strong quantitative metrics while using only 1% of the full dataset — demonstrating significant headroom for V3.

### V3 — Roadmap

| Planned Change | Expected Impact |
|---|---|
| Train on 100% of available Sentinel-2 corpus | Substantial PSNR / SSIM gains |
| GPU-accelerated inference endpoint | Real-time (~1.2s) for 512×512 tiles |
| Multi-scene temporal fusion | Improved vegetation and water boundary fidelity |

---

## Model Architecture

### RCAN v2 — 4-Band Super-Resolution Network

```
Input: [B, 4, H, W]  (4-band Sentinel-2: B2/B3/B4/B8 at 10m)
           |
           v
+-----------------------------------------------------------+
|             SHALLOW FEATURE EXTRACTION                    |
|         Conv2d(4 -> 96, 3x3, padding=1)                  |
+------------------------+----------------------------------+
                         |
                         v
+-----------------------------------------------------------+
|              RESIDUAL GROUPS (x6)                         |
|  +-----------------------------------------------------+  |
|  |  RESIDUAL CHANNEL ATTENTION BLOCKS (RCAB) x8 each  |  |
|  |                                                     |  |
|  |  Conv(96,96) -> ReLU -> Conv(96,96)                 |  |
|  |         |                                           |  |
|  |   Channel Attention:                                |  |
|  |   GAP -> FC(96->6) -> ReLU -> FC(6->96) -> Sigmoid |  |
|  |   (multiply: adaptively rescales spectral features) |  |
|  |         |                                           |  |
|  |  + Residual Skip                                    |  |
|  +-----------------------------------------------------+  |
|  Conv2d(96,96) — Group-level residual connection          |
|                                                           |
|                  x6 Groups = 48 RCAB Total                |
+------------------------+----------------------------------+
                         |
              Conv2d(96, 96, 3x3) — Deep Feature
                         |
               Long Skip Add <-------- Shallow Feature
                         |
                         v
+-----------------------------------------------------------+
|                UPSAMPLING MODULE                          |
|  Conv(96 -> 384) -> PixelShuffle(2) -> 2x spatial        |
|  Conv(96 -> 384) -> PixelShuffle(2) -> 2x spatial        |
|              Combined: 4x total upscaling                 |
+------------------------+----------------------------------+
                         |
              Conv2d(96 -> 4, 3x3) — Output Projection

Output: [B, 4, 4H, 4W]  (2.5m effective resolution)
```

**Key Design Choices:**

- **Channel Attention** — Squeeze-and-Excitation with reduction ratio 16. Adaptively weights spectral bands before feature aggregation. Critical for multispectral data where bands carry different physical information content (e.g., NIR vs Blue reflectance).
- **Residual-in-Residual structure** — Enables training very deep networks (48 blocks) without gradient vanishing.
- **No clamping** — Output tensor stored as unclamped float32 (`.npy`) to preserve full scientific dynamic range for downstream NDVI and SAM calculations.

---

## Performance Metrics (V2)

Evaluated against a held-out Sentinel-2 HR reference (4-band, spatially registered):

| Metric | Value | Interpretation |
|---|---|---|
| **PSNR** | **32.62 dB** | High-quality reconstruction; >30 dB is the standard threshold |
| **SSIM** | **0.8731** | Strong structural preservation; >0.85 is considered excellent |
| **Output Resolution** | **2.5 m** | Problem required 4m — we exceeded it |
| **Training Data Used** | **~1% of corpus** | Strong data efficiency; major headroom for V3 |

---

## Full System Pipeline

```
+------------------------------------------------------------+
|                       USER UPLOAD                          |
|          GeoTIFF (4-band, 16-bit) or PNG/JPG               |
+-----------------------------+------------------------------+
                              |
               +--------------+---------------+
          GeoTIFF?                       PNG/JPG?
               |                             |
    4-band B2/B3/B4/B8           RGB + zero-pad to 4ch
    Divide by 3000.0              Divide by 255.0
               |                             |
               +---------------+-------------+
                               |
                               v
               +-------------------------------+
               |  RCAN v2 (48 RCAB, 6 groups)  |
               |  Dual PixelShuffle -> 4x SR    |
               +---------------+---------------+
                               |
               +---------------+-----------------------------+
               |                                             |
               v                                             v
    Float32 SR tensor (.npy)               2-98% percentile stretch
    (scientific storage, unclamped)         -> 8-bit RGB PNG preview
               |                                         Browser viewer
    +----------+-----------+
    |                      |
    v                      v
SegFormer-B2         Validation Engine
7-class mask         (HR reference upload)
+ overlay            PSNR, SSIM
+ area stats         4 Scientific Heatmaps
                     (Recon, Confidence, SAM, NDVI Error)
```

---

## Validation Heatmaps

The Quality Validation page computes 4 scientific error maps against a user-provided HR reference:

| Map | Colormap | What it reveals |
|---|---|---|
| **Reconstruction Error** (per-pixel MAE across 4 bands) | `inferno` | Exact spatial locations where high-frequency detail was lost |
| **Reference Confidence** (`1 − NormError`) | `RdYlGn` | Which regions are scientifically trustworthy for downstream use |
| **SAM Error** (spectral angle in radians) | `magma` | Whether spectral signatures of land covers (water, vegetation) shifted |
| **NDVI Error** (`\|NDVI_SR − NDVI_HR\|`) | `magma` | Whether vegetation health metrics were preserved |

---

## Platform Demo Capabilities

The web platform is **ready to demonstrate** and covers the complete workflow:

1. **Upload** — accepts GeoTIFF (4-band, 16-bit) or standard RGB images
2. **Super-Resolve** — 4× upscaling with live progress tracking and image statistics
3. **View** — side-by-side comparison viewer with synchronized zoom and pan
4. **Analyze** — SegFormer land-cover segmentation with 7-class overlay and area statistics
5. **Validate** — upload an HR reference and receive PSNR, SSIM, and 4 scientific heatmaps
6. **Download** — export the SR output in original GeoTIFF format with corrected affine transform

---

## Getting Started

### One-Command Launch

**Linux / macOS:**
```bash
./run.sh
```

**Windows:**
```cmd
run.bat
```

The script prompts you to choose Docker or Python virtual environment, then handles everything automatically.

### Manual Docker
```bash
docker build -t srm-app .
docker run --rm -p 8000:8000 srm-app
```

### Manual Python
```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open **[http://localhost:8000](http://localhost:8000)** in any browser.

---

## Technology Stack

| Layer | Technology |
|---|---|
| **SR Model** | Custom PyTorch RCAN v2 (4-band, 48 RCAB, 96ch) |
| **Segmentation** | SegFormer-B2 fine-tuned on LoveDA (7 classes) |
| **Backend** | FastAPI, Rasterio, NumPy, SciKit-Image, Matplotlib |
| **Frontend** | Vanilla HTML / CSS / JavaScript — zero framework dependencies |
| **Geospatial** | Rasterio + Affine (CRS and transform preservation) |
| **Deployment** | Docker (single container, CPU PyTorch) or Python venv |

---

## Repository Structure

```
.
├── backend/          # FastAPI server, inference, validation engine
├── frontend/         # Web UI (HTML, CSS, JS)
├── model/            # Trained model weights (.pth)
├── notebooks/        # Training and validation research notebooks
├── demo/             # Visual comparison outputs
├── Dockerfile        # Single-stage lightweight container
├── run.sh / run.bat  # One-command launchers
└── requirements.txt  # Python dependencies
```
