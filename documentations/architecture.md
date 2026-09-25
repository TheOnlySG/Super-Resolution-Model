# 🏗️ Architecture Documentation

> **SIH-26142 — Satellite Imagery Super-Resolution & Land Cover Analysis**
>
> This document provides a complete architectural reference for the project — covering system design, model architectures, API structure, data flow, frontend state management, and deployment topology. All explanations are accompanied by diagrams, flowcharts, and visual references.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [System Architecture](#3-system-architecture)
4. [Backend Architecture](#4-backend-architecture)
   - [API Endpoints](#41-api-endpoints)
   - [Request / Response Contracts](#42-request--response-contracts)
   - [Startup Lifecycle](#43-startup-lifecycle)
5. [Model Architectures](#5-model-architectures)
   - [RCAN (Super-Resolution)](#51-rcan--residual-channel-attention-network)
   - [SegFormer-B2 (Segmentation)](#52-segformer-b2--semantic-segmentation)
6. [Inference Pipelines](#6-inference-pipelines)
   - [SR Inference Pipeline](#61-sr-inference-pipeline)
   - [Segmentation Inference Pipeline](#62-segmentation-inference-pipeline)
   - [End-to-End Data Flow](#63-end-to-end-data-flow)
7. [Frontend Architecture](#7-frontend-architecture)
   - [Component Hierarchy](#71-component-hierarchy)
   - [UI State Machine](#72-ui-state-machine)
   - [Viewer System](#73-viewer-system)
   - [Segmentation UI](#74-segmentation-ui)
8. [Design System](#8-design-system)
9. [Deployment Architecture](#9-deployment-architecture)
10. [Technology Stack](#10-technology-stack)

---

## 1. Project Overview

This application provides an end-to-end pipeline for **satellite image super-resolution** followed by **semantic land cover segmentation**. A user uploads a low-resolution satellite image through a web interface. The backend upscales it to 4× resolution using a custom RCAN neural network, then runs land cover classification using a pretrained SegFormer-B2 model. Results are displayed in a dual-viewer comparison interface with segmentation analytics.

```mermaid
mindmap
  root((SIH-26142<br/>Satellite SR))
    Super-Resolution
      RCAN Model
        12 RCAB Blocks
        96 Feature Channels
        4× PixelShuffle Upsampling
        4-Channel Input/Output
      Channel Adapter
        RGB → 4ch Zero-Padding
        4ch → RGB Extraction
    Segmentation
      SegFormer-B2
        LoveDA Fine-tuned
        8 Land Cover Classes
        512×512 Input
      Visualization
        Colored Mask
        Overlay Blend
        Area Statistics
    Frontend
      Dual Image Viewer
        Side-by-Side
        Split View
        Overlay Mode
      Controls
        Cursor-Anchored Zoom
        Smooth Panning
        Sync Toggle
      Land Cover Section
        Mask Display
        Legend
        Percentage Bars
    Backend
      FastAPI Server
        POST /predict
        GET /download_image
        GET /health
      Static File Serving
        HTML/CSS/JS
    Deployment
      Docker Container
      Python venv
      run.sh / run.bat
```

---

## 2. Directory Structure

```text
SIH--26142-Satellite-imagery-Super-Scaling/
│
├── 📄 Dockerfile                    # Single-container Docker build
├── 📄 requirements.txt             # Python dependencies (PyTorch, FastAPI, transformers)
├── 📄 run.sh                       # Linux launcher (Docker or venv)
├── 📄 run.bat                      # Windows launcher (Docker or venv)
├── 📄 README.md                    # Project documentation
├── 📄 architecture.md              # ← This file
├── 📄 .dockerignore                # Docker build exclusions
├── 📄 .gitignore                   # Git exclusions
│
├── 📁 backend/                      # FastAPI application
│   ├── 📄 main.py                  # App entry point, routes, startup
│   ├── 📄 inference.py             # SR inference pipeline + channel adapter
│   ├── 📄 validation.py            # Quality assessment metrics logic
│   └── 📄 model.py                 # RCAN architecture definition (PyTorch)
│
├── 📁 frontend/                     # Vanilla JS/HTML/CSS SPA
│   ├── 📄 index.html               # Main HTML structure
│   ├── 📄 app.js                   # Application logic, viewer, API calls
│   ├── 📄 validation.html          # Quality Assessment UI
│   ├── 📄 validation.js            # Validation pipeline viewer logic
│   └── 📄 style.css                # Design system, layout, animations
│
├── 📁 model/                        # Trained model weights
│   ├── 📄 rcan_improved.pth        # RCAN checkpoint (~10.6 MB, ~2.78M params)
│   └── 📁 segmentation/            # SegFormer-B2 (LoveDA)
│       ├── 📄 config.json          # HuggingFace model config
│       ├── 📄 preprocessor_config.json  # Input preprocessing spec
│       ├── 📄 pytorch_model.bin    # SegFormer weights (~104 MB)
│       ├── 📄 config.py            # Class labels, color palette, constants
│       ├── 📄 seg_model.py         # Model loading utility
│       ├── 📄 inference.py         # Segmentation pipeline (predict → visualize → stats)
│       └── 📄 __init__.py          # Module marker
│
├── 📁 notebooks/                    # Development notebooks (not used in production)
│   ├── 📄 sih-rcan.ipynb           # RCAN training/evaluation notebook
│   └── 📄 sihValAss.ipynb          # Validation assessment notebook
│
├── 📄 segment.ipynb                 # Verified segmentation inference notebook
│
└── 📁 temp_outputs/                 # Runtime SR output storage (auto-created)
```

---

## 3. System Architecture

### High-Level System Diagram

```mermaid
graph TB
    subgraph CLIENT["🌐 Browser Client"]
        UI["Frontend SPA<br/>(HTML + CSS + JS)"]
    end

    subgraph SERVER["⚙️ FastAPI Server (:8000)"]
        STATIC["Static File Mount<br/>(frontend/)"]
        API["REST API Layer"]
        
        subgraph PIPELINE["Inference Pipeline"]
            ADAPTER["Channel Adapter<br/>RGB→4ch / 4ch→RGB"]
            SR_INF["SR Inference<br/>(RCAN Forward Pass)"]
            SEG_INF["Segmentation Inference<br/>(SegFormer Forward Pass)"]
            VIZ["Visualization Engine<br/>(Mask + Overlay + Stats)"]
        end
        
        subgraph MODELS["Model Registry"]
            SR_MODEL["RCAN Model<br/>~2.78M params<br/>rcan_improved.pth"]
            SEG_MODEL["SegFormer-B2<br/>~27M params<br/>pytorch_model.bin"]
        end
    end

    subgraph STORAGE["📂 File System"]
        TEMP["temp_outputs/<br/>SR PNG files"]
    end

    UI -->|"POST /predict<br/>(multipart/form-data)"| API
    UI -->|"GET /download_image<br/>?job_id=..."| API
    UI -->|"GET /health"| API
    UI -->|"GET /*.html,js,css"| STATIC

    API --> ADAPTER
    ADAPTER --> SR_INF
    SR_INF --> SEG_INF
    SEG_INF --> VIZ
    
    SR_INF --> SR_MODEL
    SEG_INF --> SEG_MODEL
    
    SR_INF -->|"Save PNG"| TEMP
    API -->|"Serve PNG"| TEMP

    VIZ -->|"JSON Response<br/>(base64 images + stats)"| UI

    style CLIENT fill:#1e293b,stroke:#38bdf8,color:#f3f4f6
    style SERVER fill:#111827,stroke:#0ea5e9,color:#f3f4f6
    style PIPELINE fill:#0f172a,stroke:#334155,color:#9ca3af
    style MODELS fill:#0f172a,stroke:#334155,color:#9ca3af
    style STORAGE fill:#1e293b,stroke:#374151,color:#9ca3af
```

### Single-Server Architecture

The entire application runs as a **single FastAPI process** serving both the API and the static frontend. There is no reverse proxy, no message queue, and no separate frontend server. This simplicity is intentional — the application is designed for direct deployment via Docker or a Python virtual environment.

```mermaid
graph LR
    BROWSER["Browser"] -->|"HTTP :8000"| UVICORN["Uvicorn<br/>ASGI Server"]
    UVICORN --> FASTAPI["FastAPI App"]
    FASTAPI --> ROUTES["API Routes<br/>/predict<br/>/download_image<br/>/health"]
    FASTAPI --> STATIC["StaticFiles Mount<br/>frontend/ → /"]
    
    style BROWSER fill:#1e293b,stroke:#38bdf8,color:#f3f4f6
    style UVICORN fill:#111827,stroke:#0ea5e9,color:#f3f4f6
    style FASTAPI fill:#0f172a,stroke:#334155,color:#f3f4f6
    style ROUTES fill:#0f172a,stroke:#10b981,color:#f3f4f6
    style STATIC fill:#0f172a,stroke:#f59e0b,color:#f3f4f6
```

---

## 4. Backend Architecture

### 4.1 API Endpoints

```mermaid
graph LR
    subgraph API["FastAPI REST API"]
        direction TB
        H["GET /health"]
        P["POST /predict"]
        V["POST /validate"]
        D["GET /download_image"]
        S["GET /* (static)"]
    end

    H -->|"200"| H_RES["{ status, sr_model_loaded,<br/>seg_model_loaded, device }"]
    P -->|"200"| P_RES["{ status, job_id, metadata,<br/>visualizations, segmentation }"]
    P -->|"400"| P_ERR1["{ detail: 'Image Error: ...' }"]
    P -->|"503"| P_ERR2["{ detail: 'Model not loaded' }"]
    P -->|"500"| P_ERR3["{ detail: 'Inference error: ...' }"]
    V -->|"200"| V_RES["{ status, validation, maps, metrics }"]
    V -->|"400"| V_ERR["{ detail: 'Validation error: ...' }"]
    D -->|"200"| D_RES["FileResponse (dynamic format)"]
    D -->|"404"| D_ERR["{ detail: 'SR output not found' }"]
    S -->|"200"| S_RES["Static HTML/CSS/JS"]

    style API fill:#111827,stroke:#0ea5e9,color:#f3f4f6
    style H fill:#10b981,stroke:#059669,color:#fff
    style P fill:#0ea5e9,stroke:#0284c7,color:#fff
    style V fill:#0ea5e9,stroke:#0284c7,color:#fff
    style D fill:#f59e0b,stroke:#d97706,color:#fff
    style S fill:#8b5cf6,stroke:#7c3aed,color:#fff
```

| Method | Path | Purpose | Auth | Body |
|--------|------|---------|------|------|
| `GET` | `/health` | Health check — reports model load status and device | None | — |
| `POST` | `/predict` | Upload LR image → SR + Segmentation pipeline | None | `multipart/form-data` (`lr_file`) |
| `POST` | `/validate` | Upload HR image → Metrics & Maps generation | None | `multipart/form-data` (`hr_file`, `job_id`) |
| `GET` | `/download_image?job_id=<uuid>` | Download full-resolution SR Output | None | — |
| `GET` | `/*` | Static file serving (frontend SPA) | None | — |

### 4.2 Request / Response Contracts

#### `POST /predict` — Request

```text
Content-Type: multipart/form-data

┌──────────────────────────────────┐
│  Field: lr_file                  │
│  Type:  UploadFile (binary)      │
│  Accepts: .png, .jpg, .jpeg      │
└──────────────────────────────────┘
```

#### `POST /predict` — Response

```json
{
  "status": "success",
  "job_id": "8e28520c-405f-4f7c-8c97-e37067f883c1",
  "metadata": {
    "input": {
      "filename": "satellite_tile.png",
      "width": 128,
      "height": 128,
      "channels": 3,
      "size_bytes": 45120,
      "format": "PNG"
    },
    "output": {
      "width": 512,
      "height": 512,
      "channels": 3,
      "scale_factor": 4,
      "format": "PNG"
    }
  },
  "visualizations": {
    "lr_rgb": "<base64 PNG>",
    "sr_rgb": "<base64 PNG>"
  },
  "segmentation": {
    "available": true,
    "mask": "<base64 PNG — colored class mask>",
    "overlay": "<base64 PNG — blended SR + mask>",
    "classes": [
      { "class_id": 6, "label": "Forest", "color": "#00b400", "percentage": 42.15 },
      { "class_id": 7, "label": "Agricultural", "color": "#ffb400", "percentage": 28.73 },
      { "class_id": 1, "label": "Background", "color": "#1e1e1e", "percentage": 15.02 },
      { "class_id": 2, "label": "Building", "color": "#ff0000", "percentage": 8.44 },
      { "class_id": 3, "label": "Road", "color": "#ffff00", "percentage": 3.21 },
      { "class_id": 4, "label": "Water", "color": "#0078ff", "percentage": 1.85 },
      { "class_id": 5, "label": "Barren", "color": "#b4783c", "percentage": 0.60 }
    ]
  }
}
```

### 4.3 Startup Lifecycle

```mermaid
sequenceDiagram
    participant UV as Uvicorn
    participant FA as FastAPI
    participant SR as RCAN Loader
    participant SEG as SegFormer Loader
    participant FS as File System

    UV->>FA: Start ASGI Application
    FA->>FA: @app.on_event("startup")
    
    rect rgb(15, 23, 42)
        Note over FA,SR: SR Model Loading (Required)
        FA->>FS: Check MODEL_PATH exists
        FS-->>FA: ✓ rcan_improved.pth found
        FA->>SR: load_model(path, device)
        SR->>SR: torch.load(checkpoint)
        SR->>SR: RCAN(num_blocks=12, channels=96)
        SR->>SR: model.load_state_dict()
        SR->>SR: model.eval()
        SR-->>FA: (model, device, normalization)
        FA->>FA: MODEL = model ✓
    end

    rect rgb(15, 23, 42)
        Note over FA,SEG: Segmentation Model Loading (Optional)
        FA->>SEG: load_segmentation_model(device)
        SEG->>SEG: SegformerForSemanticSegmentation.from_pretrained()
        SEG->>SEG: model.eval()
        SEG-->>FA: (model, device)
        FA->>FA: SEG_MODEL = model ✓
    end

    Note over FA: If SEG fails → SEG_MODEL = None<br/>SR pipeline still works

    FA-->>UV: Application startup complete
    UV->>UV: Listening on 0.0.0.0:8000
```

> [!IMPORTANT]
> **Fault Isolation:** Segmentation model loading is wrapped in a `try/except`. If it fails (e.g., missing weights, incompatible transformers version), the SR pipeline continues to function normally. The `/predict` response will include `"segmentation": { "available": false, "error": "..." }`.

---

## 5. Model Architectures

### 5.1 RCAN — Residual Channel Attention Network

The super-resolution model is a custom **RCAN** (Residual Channel Attention Network) originally trained on Sentinel-2 satellite imagery with 4 spectral bands. It achieves **4× spatial upscaling**.

#### Architecture Diagram

```mermaid
graph TD
    INPUT["Input Tensor<br/>(B, 4, H, W)"] --> HEAD

    subgraph HEAD_BLOCK["Shallow Feature Extraction"]
        HEAD["Conv2d(4 → 96, k=3, p=1)"]
    end

    HEAD --> BODY_IN["features"]
    HEAD --> RESIDUAL["Global Residual<br/>(skip connection)"]

    subgraph BODY["Deep Feature Extraction — 12× RCAB Blocks"]
        direction TB
        RCAB1["RCAB #1"]
        RCAB2["RCAB #2"]
        DOTS["⋮"]
        RCAB12["RCAB #12"]
        RCAB1 --> RCAB2 --> DOTS --> RCAB12
    end

    BODY_IN --> BODY
    BODY --> BODY_CONV["Conv2d(96 → 96, k=3, p=1)"]
    
    BODY_CONV --> ADD["⊕ Element-wise Add"]
    RESIDUAL --> ADD

    subgraph UPSAMPLE["4× Upsampling (2× + 2×)"]
        UP1["Conv2d(96 → 384, k=3, p=1)<br/>PixelShuffle(2) → (96, 2H, 2W)<br/>ReLU"]
        UP2["Conv2d(96 → 384, k=3, p=1)<br/>PixelShuffle(2) → (96, 4H, 4W)<br/>ReLU"]
        UP1 --> UP2
    end

    ADD --> UPSAMPLE

    UPSAMPLE --> TAIL["Conv2d(96 → 4, k=3, p=1)"]
    TAIL --> OUTPUT["Output Tensor<br/>(B, 4, 4H, 4W)"]

    style INPUT fill:#0ea5e9,stroke:#0284c7,color:#fff
    style OUTPUT fill:#10b981,stroke:#059669,color:#fff
    style BODY fill:#1e1b4b,stroke:#4f46e5,color:#c7d2fe
    style UPSAMPLE fill:#1e1b4b,stroke:#7c3aed,color:#c7d2fe
    style HEAD_BLOCK fill:#1e1b4b,stroke:#334155,color:#c7d2fe
```

#### RCAB Block Detail

Each **Residual Channel Attention Block (RCAB)** combines local feature extraction with a squeeze-and-excitation style channel attention mechanism:

```mermaid
graph LR
    X["Input<br/>(96 ch)"] --> CONV1["Conv2d(96→96, k=3)"]
    CONV1 --> RELU["ReLU"]
    RELU --> CONV2["Conv2d(96→96, k=3)"]
    
    CONV2 --> GAP["AdaptiveAvgPool2d(1)<br/>Global Average Pooling"]
    GAP --> ATT1["Conv2d(96→12, k=1)<br/>Squeeze (÷8)"]
    ATT1 --> RELU2["ReLU"]
    RELU2 --> ATT2["Conv2d(12→96, k=1)<br/>Excitation"]
    ATT2 --> SIG["Sigmoid"]
    
    CONV2 --> MUL["⊗ Channel Attention"]
    SIG --> MUL
    
    MUL --> SCALE["× 0.1<br/>Residual Scaling"]
    SCALE --> ADD["⊕ Skip Connection"]
    X --> ADD
    ADD --> OUT["Output<br/>(96 ch)"]

    style X fill:#1e293b,stroke:#38bdf8,color:#f3f4f6
    style OUT fill:#1e293b,stroke:#10b981,color:#f3f4f6
    style SCALE fill:#7c2d12,stroke:#ea580c,color:#fed7aa
```

#### Key Specifications

| Parameter | Value |
|-----------|-------|
| Input channels | 4 (RGB + zero-padded 4th channel) |
| Output channels | 4 (first 3 extracted as RGB) |
| Feature channels | 96 |
| RCAB blocks | 12 |
| Attention reduction ratio | 8 (96 → 12 → 96) |
| Residual scaling factor | 0.1 |
| Upscaling method | 2× PixelShuffle × 2 stages = 4× total |
| Total parameters | ~2,776,276 (~2.78M) |
| Checkpoint size | ~10.6 MB |
| Normalization | Input ÷ 255.0, Output clamp [0, 1] × 255 |

---

### 5.2 SegFormer-B2 — Semantic Segmentation

The segmentation model is a **SegFormer-B2** pretrained on ImageNet and fine-tuned on the **LoveDA** dataset for land cover classification. It is loaded from local weights via HuggingFace `transformers`.

#### Architecture Overview

```mermaid
graph TD
    INPUT["SR Image (RGB)<br/>Resized to 512×512"] --> PATCH

    subgraph ENCODER["Hierarchical Transformer Encoder"]
        direction TB
        
        subgraph S1["Stage 1"]
            PATCH["Overlapping Patch Embed"]
            T1["3× Transformer Blocks<br/>hidden=64, heads=1"]
            PATCH --> T1
        end
        
        subgraph S2["Stage 2"]
            PE2["Patch Embed (↓2×)"]
            T2["4× Transformer Blocks<br/>hidden=128, heads=2"]
            PE2 --> T2
        end
        
        subgraph S3["Stage 3"]
            PE3["Patch Embed (↓2×)"]
            T3["6× Transformer Blocks<br/>hidden=320, heads=5"]
            PE3 --> T3
        end
        
        subgraph S4["Stage 4"]
            PE4["Patch Embed (↓2×)"]
            T4["3× Transformer Blocks<br/>hidden=512, heads=8"]
            PE4 --> T4
        end
        
        T1 --> PE2
        T2 --> PE3
        T3 --> PE4
    end

    subgraph DECODER["All-MLP Decoder"]
        direction TB
        MLP1["MLP (64 → 768)"]
        MLP2["MLP (128 → 768)"]
        MLP3["MLP (320 → 768)"]
        MLP4["MLP (512 → 768)"]
        FUSE["Concatenate + Fuse<br/>Conv2d(4×768 → 768)"]
        CLS["Classification Head<br/>Conv2d(768 → 8)"]
        MLP1 --> FUSE
        MLP2 --> FUSE
        MLP3 --> FUSE
        MLP4 --> FUSE
        FUSE --> CLS
    end

    T1 --> MLP1
    T2 --> MLP2
    T3 --> MLP3
    T4 --> MLP4

    CLS --> LOGITS["Logits<br/>(B, 8, H/4, W/4)"]
    LOGITS --> INTERP["Bilinear Interpolation<br/>→ (B, 8, H_orig, W_orig)"]
    INTERP --> ARGMAX["Argmax → Class Mask<br/>(H_orig, W_orig)"]

    style INPUT fill:#0ea5e9,stroke:#0284c7,color:#fff
    style ARGMAX fill:#10b981,stroke:#059669,color:#fff
    style ENCODER fill:#1e1b4b,stroke:#4f46e5,color:#c7d2fe
    style DECODER fill:#1e1b4b,stroke:#7c3aed,color:#c7d2fe
```

#### Land Cover Classes

| Class ID | Label | Color | Hex |
|----------|-------|-------|-----|
| 0 | Ignore | ⬛ Black | `#000000` |
| 1 | Background | ⬛ Dark Gray | `#1e1e1e` |
| 2 | Building | 🟥 Red | `#ff0000` |
| 3 | Road | 🟨 Yellow | `#ffff00` |
| 4 | Water | 🟦 Blue | `#0078ff` |
| 5 | Barren | 🟫 Brown | `#b4783c` |
| 6 | Forest | 🟩 Green | `#00b400` |
| 7 | Agricultural | 🟧 Orange | `#ffb400` |

#### Key Specifications

| Parameter | Value |
|-----------|-------|
| Architecture | SegFormer-B2 (Mix Transformer) |
| Training data | LoveDA (urban + rural land cover) |
| Input size | 512 × 512 (resized) |
| Normalization | ImageNet mean/std |
| Encoder stages | 4 (depths: 3, 4, 6, 3) |
| Hidden sizes | 64, 128, 320, 512 |
| Attention heads | 1, 2, 5, 8 |
| Decoder hidden | 768 |
| Output classes | 8 (including Ignore) |
| Parameters | ~27M |
| Checkpoint size | ~104 MB |

---

## 6. Inference Pipelines

### 6.1 SR Inference Pipeline

```mermaid
flowchart TD
    A["User uploads RGB image<br/>(H × W × 3)"] --> B["Open with PIL → RGB"]
    B --> C["Convert to float32 array<br/>(H, W, 3)"]
    C --> D["Generate LR preview<br/>(base64 PNG)"]
    
    C --> E["3→4 Channel Adapter<br/>Zero-pad 4th channel"]
    E --> F["Transpose to<br/>(4, H, W)"]
    F --> G["Normalize<br/>÷ 255.0"]
    G --> H["Create tensor<br/>(1, 4, H, W)"]
    
    H --> I{{"RCAN Forward Pass<br/>torch.no_grad()"}}
    
    I --> J["Output tensor<br/>(1, 4, 4H, 4W)"]
    J --> K["Squeeze + Clamp [0, 1]"]
    K --> SAVE_NPY["Save 4-band float32<br/>temp_outputs/{job_id}_sr_tensor.npy<br/>(For Validation)"]
    K --> L["4→3 Channel Extraction<br/>Take channels [:3]"]
    L --> M["Transpose to<br/>(4H, 4W, 3)"]
    M --> N["Denormalize<br/>× 255 → uint8"]
    
    N --> O["Save output file<br/>(e.g., .tif, .jp2, .png)"]
    N --> P["Generate SR preview<br/>(base64 PNG)"]
    N --> Q["Return PIL Image<br/>→ Segmentation Pipeline"]
    SAVE_NPY --> O
    
    style A fill:#0ea5e9,stroke:#0284c7,color:#fff
    style I fill:#7c3aed,stroke:#6d28d9,color:#fff
    style Q fill:#10b981,stroke:#059669,color:#fff
    style SAVE_NPY fill:#f59e0b,stroke:#d97706,color:#fff
```

> [!NOTE]
> **Channel Adapter Rationale:** The RCAN model was trained on 4-band Sentinel-2 imagery (Blue, Green, Red, NIR). For standard RGB input, the 4th channel is zero-padded before inference. After inference, only the first 3 output channels are extracted as the final RGB result. The model weights are never modified.

### 6.2 Segmentation Inference Pipeline

```mermaid
flowchart TD
    A["SR PIL Image (RGB)<br/>(4H × 4W × 3)"] --> B["Resize to 512×512"]
    B --> C["ToTensor → (3, 512, 512)"]
    C --> D["Normalize<br/>ImageNet mean/std"]
    D --> E["Unsqueeze → (1, 3, 512, 512)"]
    
    E --> F{{"SegFormer Forward Pass<br/>torch.no_grad()"}}
    
    F --> G["Logits<br/>(1, 8, 128, 128)"]
    G --> H["Bilinear Interpolate<br/>→ (1, 8, 4H, 4W)"]
    H --> I["Argmax dim=1<br/>→ Class Mask (4H, 4W)"]

    I --> J["Palette Lookup<br/>mask → RGB array"]
    J --> K["Mask Image<br/>(4H, 4W, 3)"]
    
    I --> L["Class Percentages<br/>(exclude Ignore class)"]
    
    K --> M["Alpha Blend<br/>0.55×SR + 0.45×Mask"]
    A --> M
    M --> N["Overlay Image<br/>(4H, 4W, 3)"]
    
    K --> O["Encode base64 PNG"]
    N --> P["Encode base64 PNG"]
    
    O --> Q["JSON Response<br/>mask + overlay + classes"]
    P --> Q
    L --> Q

    style A fill:#10b981,stroke:#059669,color:#fff
    style F fill:#7c3aed,stroke:#6d28d9,color:#fff
    style Q fill:#f59e0b,stroke:#d97706,color:#fff
```

### 6.3 Validation Pipeline

```mermaid
flowchart TD
    A["User uploads HR GeoTIFF<br/>(4-band)"] --> B["Validate Alignment<br/>(Compare with SR meta)"]
    B --> C["Normalize HR<br/>÷ 3000.0"]
    C --> D["Load SR Tensor<br/>temp_outputs/{job_id}_sr_tensor.npy"]
    
    D --> E["Compute Reconstruction Error<br/>(Mean Absolute Error B2/B3/B4/B8)"]
    D --> F["Compute Confidence Map<br/>(Percentile Normalized Error)"]
    D --> G["Compute SAM Map<br/>(Spectral Angle Mapper)"]
    D --> H["Compute NDVI Error Map"]
    D --> I["Compute PSNR, SSIM, RMSE"]
    
    E --> J["Colorize & Base64 Encode Maps<br/>(inferno, magma, Blues)"]
    F --> J
    G --> J
    H --> J
    
    J --> K["Return JSON Response<br/>{ metrics, maps }"]
    
    style A fill:#0ea5e9,stroke:#0284c7,color:#fff
    style D fill:#7c3aed,stroke:#6d28d9,color:#fff
    style K fill:#10b981,stroke:#059669,color:#fff
```

### 6.4 End-to-End Data Flow

```mermaid
sequenceDiagram
    participant B as Browser
    participant F as FastAPI
    participant SR as RCAN Model
    participant SEG as SegFormer Model
    participant FS as File System

    B->>F: POST /predict (multipart: lr_file)
    
    rect rgb(15, 23, 42)
        Note over F: Save upload to temp file
        F->>FS: Write upload to temp_outputs/upload_<name>
    end

    rect rgb(30, 58, 95)
        Note over F,SR: Super-Resolution Pipeline
        F->>F: PIL.open() → RGB
        F->>F: 3→4 Channel Adapter (zero-pad)
        F->>F: Normalize (÷255), create tensor
        F->>SR: model(input_tensor)
        SR-->>F: output_tensor (1, 4, 4H, 4W)
        F->>F: 4→3 Channel Extraction
        F->>F: Clamp, denormalize → uint8
        F->>FS: Save SR PNG (temp_outputs/{job_id}.png)
        F->>F: Generate base64 previews
    end

    rect rgb(30, 40, 80)
        Note over F,SEG: Segmentation Pipeline (if model loaded)
        F->>F: Resize SR image → 512×512
        F->>F: Normalize (ImageNet stats)
        F->>SEG: model(pixel_values)
        SEG-->>F: logits (1, 8, 128, 128)
        F->>F: Interpolate → original SR size
        F->>F: Argmax → class mask
        F->>F: Apply color palette → mask image
        F->>F: Alpha blend → overlay image
        F->>F: Calculate class percentages
        F->>F: Encode mask + overlay as base64
    end

    rect rgb(15, 23, 42)
        Note over F: Cleanup
        F->>FS: Delete uploaded temp file
    end

    F-->>B: JSON { status, job_id, metadata,<br/>visualizations, segmentation }

    Note over B: User clicks Download
    B->>F: GET /download_image?job_id=<uuid>
    F->>FS: Read temp_outputs/{job_id}.png
    F-->>B: FileResponse (PNG binary)
```

---

## 7. Frontend Architecture

### 7.1 Component Hierarchy

```mermaid
graph TD
    APP["#app<br/>Root Container"]
    
    APP --> HEADER["header.app-header<br/>Title + Model Badges"]
    APP --> LAYOUT["div.main-layout<br/>(CSS Grid: 320px | 1fr)"]
    APP --> SEGMENTATION["section.seg-section<br/>Land Cover Analysis"]
    
    LAYOUT --> SIDEBAR["aside.sidebar-left"]
    LAYOUT --> VIEWER["main.main-viewer"]
    
    subgraph SIDEBAR_PANELS["Sidebar — Mutually Exclusive Panels"]
        UPLOAD["#upload-panel<br/>Workspace + Drop Zone"]
        PROCESS["#processing-panel<br/>Step Indicators"]
        META["#metadata-panel<br/>Stats Table + Download"]
    end
    SIDEBAR --> SIDEBAR_PANELS
    
    subgraph VIEWER_SYSTEM["Main Viewer"]
        EMPTY["#stage-empty<br/>'No image loaded'"]
        CONTAINER["#viewer-container"]
        
        CONTAINER --> TOOLBAR["div.toolbar<br/>Mode Toggles + Sync"]
        CONTAINER --> WORKSPACE["#workspace<br/>Mode-specific Layout"]
        CONTAINER --> CONTROLS["div.bottom-controls<br/>Zoom/Fit/Reset/Fullscreen"]
        
        WORKSPACE --> PANE_O["#pane-original<br/>Original Image Viewer"]
        WORKSPACE --> DIVIDER["#split-divider<br/>Split Mode Handle"]
        WORKSPACE --> PANE_SR["#pane-sr<br/>SR Image Viewer"]
        WORKSPACE --> OVERLAY_CTRL["#overlay-controls<br/>Opacity Slider"]
    end
    VIEWER --> VIEWER_SYSTEM

    subgraph SEG_SYSTEM["Segmentation Display"]
        SEG_LOAD["#seg-loading<br/>Spinner"]
        SEG_ERR["#seg-error<br/>Error Message"]
        SEG_RES["#seg-results"]
        SEG_RES --> SEG_IMGS["div.seg-images<br/>Mask + Overlay"]
        SEG_RES --> SEG_INFO["div.seg-info-row<br/>Legend + Stats"]
    end
    SEGMENTATION --> SEG_SYSTEM

    style APP fill:#0f172a,stroke:#334155,color:#f3f4f6
    style SIDEBAR_PANELS fill:#111827,stroke:#1f2937,color:#9ca3af
    style VIEWER_SYSTEM fill:#111827,stroke:#1f2937,color:#9ca3af
    style SEG_SYSTEM fill:#111827,stroke:#1f2937,color:#9ca3af
```

### 7.2 UI State Machine

The application follows a linear state machine. The sidebar panels are **mutually exclusive** — only one is visible at a time.

```mermaid
stateDiagram-v2
    [*] --> IDLE: Page Load

    state IDLE {
        note right of IDLE
            • upload-panel: VISIBLE
            • processing-panel: HIDDEN
            • metadata-panel: HIDDEN
            • stage-empty: VISIBLE
            • viewer-container: HIDDEN
            • segmentation-section: HIDDEN
            • start-btn: DISABLED
        end note
    }

    IDLE --> FILE_SELECTED: User drops/selects image

    state FILE_SELECTED {
        note right of FILE_SELECTED
            • Upload zone shows filename
            • start-btn: ENABLED
            • Local preview loaded into viewers
            • viewer-container: VISIBLE
            • stage-empty: HIDDEN
            • Images fit to screen
        end note
    }

    FILE_SELECTED --> PROCESSING: Click "Start Super-Resolution"

    state PROCESSING {
        note right of PROCESSING
            • upload-panel: HIDDEN
            • processing-panel: VISIBLE
            • Steps animate sequentially:
              1. Analyzing Input (active → done)
              2. Running Super-Resolution (active → done)
              3. Generating Result (active → done)
        end note
    }

    PROCESSING --> RESULTS: API returns successfully

    state RESULTS {
        note right of RESULTS
            • processing-panel: HIDDEN
            • metadata-panel: VISIBLE
            • SR image loaded into viewer
            • Stats table populated
            • segmentation-section: VISIBLE
            • Mask, overlay, legend, stats populated
            • Auto-scroll to segmentation
        end note
    }

    PROCESSING --> ERROR: API error / network failure

    state ERROR {
        note right of ERROR
            • Alert displayed
            • Page reloads → back to IDLE
        end note
    }

    ERROR --> IDLE: location.reload()
```

### 7.3 Viewer System

The image comparison workspace supports three modes. The viewer uses **CSS transforms** (`translate` + `scale`) with `transform-origin: 0 0` for hardware-accelerated rendering.

```mermaid
graph TD
    subgraph MODES["Viewer Modes"]
        SIDE["SIDE BY SIDE<br/>(Default)"]
        SPLIT["SPLIT VIEW"]
        OVERLAY["OVERLAY"]
    end
    
    SIDE --> SIDE_DESC["Two panes in flex layout<br/>Independent or synced<br/>Each with own transform"]
    
    SPLIT --> SPLIT_DESC["Both panes absolute-positioned<br/>SR clipped via CSS polygon()<br/>Draggable divider controls clip %<br/>SR: pointer-events: none"]
    
    OVERLAY --> OVERLAY_DESC["Both panes stacked<br/>SR opacity controlled by slider<br/>Range: 0.0 → 1.0<br/>SR: pointer-events: none"]

    subgraph SYNC["Sync Behavior"]
        ON["SYNC ON (default)"]
        OFF["SYNC OFF"]
    end
    
    ON --> ON_DESC["All controls affect both viewers<br/>Transforms synced via dimension ratio<br/>Active indicator: HIDDEN"]
    
    OFF --> OFF_DESC["Controls affect active viewer only<br/>Active set by mousedown on pane<br/>Active indicator: VISIBLE<br/>Active pane has cyan border"]
    
    style MODES fill:#1e1b4b,stroke:#4f46e5,color:#c7d2fe
    style SYNC fill:#1e1b4b,stroke:#7c3aed,color:#c7d2fe
```

#### Zoom Mathematics

Cursor-anchored zoom ensures the point under the cursor stays fixed during zoom:

$$\text{panX}_{\text{new}} = x - (x - \text{panX}_{\text{old}}) \times \frac{\text{scale}_{\text{new}}}{\text{scale}_{\text{old}}}$$

$$\text{panY}_{\text{new}} = y - (y - \text{panY}_{\text{old}}) \times \frac{\text{scale}_{\text{new}}}{\text{scale}_{\text{old}}}$$

Where $(x, y)$ is the cursor position relative to the pane's bounding rect.

#### Sync Transform Calculation

When syncing two viewers with different native resolutions (e.g., Original 128×128 and SR 512×512):

$$\text{scale}_{\text{dest}} = \text{scale}_{\text{source}} \times \frac{1}{\text{ratio}_W}$$

Where $\text{ratio}_W = \frac{W_{\text{dest}}}{W_{\text{source}}}$. Pan coordinates are identical because the transform is applied in screen-space.

### 7.4 Segmentation UI

```mermaid
flowchart TD
    TRIGGER["populateUI() completes"] --> CALL["populateSegmentation(data.segmentation)"]
    
    CALL --> SHOW["Show #segmentation-section"]
    SHOW --> CHECK{"seg.available<br/>== true?"}
    
    CHECK -->|No| ERR["Show #seg-error<br/>Display error message"]
    
    CHECK -->|Yes| RESULTS["Show #seg-results"]
    RESULTS --> IMG1["Set #seg-mask-img.src<br/>(base64 mask PNG)"]
    RESULTS --> IMG2["Set #seg-overlay-img.src<br/>(base64 overlay PNG)"]
    RESULTS --> LEGEND["Build #seg-legend<br/>Color swatches + labels"]
    RESULTS --> STATS["Build #seg-stats<br/>Percentage bars"]
    
    STATS --> SCROLL["setTimeout 300ms<br/>scrollIntoView({ smooth })"]

    style TRIGGER fill:#0ea5e9,stroke:#0284c7,color:#fff
    style ERR fill:#ef4444,stroke:#dc2626,color:#fff
    style SCROLL fill:#10b981,stroke:#059669,color:#fff
```

---

## 8. Design System

### Color Tokens

```mermaid
graph LR
    subgraph PALETTE["CSS Custom Properties (:root)"]
        direction TB
        BG["--bg-color<br/>#0b0f19"]
        PANEL["--panel-bg<br/>#111827"]
        BORDER["--border-color<br/>#1f2937"]
        TEXT1["--text-primary<br/>#f3f4f6"]
        TEXT2["--text-secondary<br/>#9ca3af"]
        ACCENT["--accent-color<br/>#0ea5e9"]
        HOVER["--accent-hover<br/>#38bdf8"]
        DANGER["--danger<br/>#ef4444"]
        SUCCESS["--success<br/>#10b981"]
    end

    style BG fill:#0b0f19,stroke:#1f2937,color:#f3f4f6
    style PANEL fill:#111827,stroke:#1f2937,color:#f3f4f6
    style BORDER fill:#1f2937,stroke:#374151,color:#f3f4f6
    style TEXT1 fill:#f3f4f6,stroke:#d1d5db,color:#111827
    style TEXT2 fill:#9ca3af,stroke:#6b7280,color:#111827
    style ACCENT fill:#0ea5e9,stroke:#0284c7,color:#fff
    style HOVER fill:#38bdf8,stroke:#0ea5e9,color:#111827
    style DANGER fill:#ef4444,stroke:#dc2626,color:#fff
    style SUCCESS fill:#10b981,stroke:#059669,color:#fff
```

### Typography

| Usage | Font | Weights |
|-------|------|---------|
| Body text, headings, labels | **Inter** | 300, 400, 500, 600 |
| Badges, stats, zoom level, monospace data | **JetBrains Mono** | 400, 500 |

### Layout Grid

```text
┌─────────────────────────────────────────────────────────────────┐
│  HEADER (60px fixed height)                                     │
│  [Title + Subtitle]                           [Scale] [Arch]    │
├────────────┬────────────────────────────────────────────────────┤
│            │                                                    │
│  SIDEBAR   │              MAIN VIEWER                          │
│  (320px)   │              (1fr flex)                            │
│            │                                                    │
│  Upload    │   ┌──────────┐   ┌──────────┐                     │
│  Panel     │   │ ORIGINAL │   │    SR    │                     │
│  ─────     │   │          │   │          │                     │
│  Process   │   │          │   │          │                     │
│  Panel     │   └──────────┘   └──────────┘                     │
│  ─────     │                                                    │
│  Metadata  │   [−] 100% [+]  [FIT] [1:1] [RESET] [FULLSCREEN] │
│  Panel     │                                                    │
├────────────┴────────────────────────────────────────────────────┤
│  SEGMENTATION SECTION (full width, scrollable)                  │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │ Segmentation     │  │ Overlay on       │                    │
│  │ Mask             │  │ SR Image         │                    │
│  └──────────────────┘  └──────────────────┘                    │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │ Legend           │  │ Area Distribution │                    │
│  │ ■ Building       │  │ Forest    ████ 42%│                    │
│  │ ■ Road           │  │ Agri      ███  29%│                    │
│  │ ■ Water          │  │ Building  ██    8%│                    │
│  └──────────────────┘  └──────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Deployment Architecture

### Docker Deployment

```mermaid
graph TD
    subgraph DOCKER["Docker Container"]
        BASE["python:3.10-slim"]
        BASE --> DEPS["pip install -r requirements.txt<br/>(PyTorch CPU, FastAPI, transformers)"]
        DEPS --> COPY["COPY . /app"]
        COPY --> UVICORN["CMD: uvicorn backend.main:app<br/>--host 0.0.0.0 --port 8000"]
    end

    subgraph ENV["Environment"]
        E1["MODEL_PATH=/app/model/rcan_improved.pth"]
        E2["PYTHONUNBUFFERED=1"]
        E3["EXPOSE 8000"]
    end

    BROWSER["Browser<br/>localhost:8000"] -->|"HTTP"| DOCKER

    style DOCKER fill:#111827,stroke:#0ea5e9,color:#f3f4f6
    style ENV fill:#1e293b,stroke:#334155,color:#9ca3af
```

### Launch Options

```mermaid
graph TD
    START["run.sh / run.bat"] --> CHOICE{"User Choice"}
    
    CHOICE -->|"1. Docker"| DOCKER_PATH
    CHOICE -->|"2. Python venv"| VENV_PATH
    
    subgraph DOCKER_PATH["Docker Path"]
        D1["docker build -t srm-app ."]
        D2["docker run --rm -p 8000:8000 srm-app"]
        D1 --> D2
    end
    
    subgraph VENV_PATH["Virtual Environment Path"]
        V1["python3 -m venv .venv"]
        V2["source .venv/bin/activate"]
        V3["pip install -r requirements.txt"]
        V4["uvicorn backend.main:app<br/>--host 0.0.0.0 --port 8000"]
        V1 --> V2 --> V3 --> V4
    end
    
    DOCKER_PATH --> READY["Application Ready<br/>http://localhost:8000"]
    VENV_PATH --> READY

    style START fill:#0ea5e9,stroke:#0284c7,color:#fff
    style READY fill:#10b981,stroke:#059669,color:#fff
```

---

## 10. Technology Stack

```mermaid
mindmap
  root((Tech Stack))
    Backend
      Python 3.10+
      FastAPI
        ASGI framework
        Auto OpenAPI docs
      Uvicorn
        ASGI server
        Single worker
      PyTorch
        RCAN model
        CPU inference
      HuggingFace Transformers
        SegFormer loading
        Local weights only
      Torchvision
        Image transforms
        Normalization
      Pillow
        Image I/O
        Format conversion
      NumPy
        Array operations
        Palette lookup
    Frontend
      HTML5
        Semantic structure
        Drag and drop API
      Vanilla JavaScript
        No framework
        DOM manipulation
        Fetch API
      CSS3
        Custom properties
        CSS Grid layout
        CSS transforms
        clip-path polygon
        Animations
      Google Fonts
        Inter
        JetBrains Mono
    Infrastructure
      Docker
        python:3.10-slim
        Single container
        CPU PyTorch wheels
      File System
        temp_outputs/
        Static serving
      Cross-platform
        run.sh (Linux/Mac)
        run.bat (Windows)
```

---

> **Document Version:** 1.0
>
> **Last Updated:** September 2026
>
> **Scope:** Read-only architectural reference. This document does not modify any application code or configuration.
