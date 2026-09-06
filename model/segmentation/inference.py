"""Segmentation inference pipeline — mirrors the verified notebook exactly."""

import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model.segmentation.config import (
    PALETTE,
    CLASS_LABELS,
    CLASS_COLORS_HEX,
    IGNORE_CLASS_ID,
)

# ── Preprocessing (identical to notebook Cell 3) ──────────────────────────────

_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ── Core inference ────────────────────────────────────────────────────────────

def predict_mask(model, device, pil_image: Image.Image) -> np.ndarray:
    """
    Run SegFormer inference on a PIL RGB image.
    Returns a (H, W) numpy array of class IDs at the original image resolution.
    """
    input_tensor = _transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=input_tensor)

    # Resize logits back to original image dimensions (notebook Cell 3)
    logits = F.interpolate(
        outputs.logits,
        size=(pil_image.height, pil_image.width),
        mode="bilinear",
        align_corners=False,
    )

    prediction = logits.argmax(dim=1)[0].cpu().numpy()
    return prediction


# ── Visualization ─────────────────────────────────────────────────────────────

def generate_mask_image(mask: np.ndarray) -> np.ndarray:
    """
    Convert a (H, W) class-ID mask into an (H, W, 3) uint8 RGB image
    using the palette from the verified notebook (Cell 4).
    """
    return PALETTE[mask]


def generate_overlay_image(
    sr_array: np.ndarray,
    mask_color_array: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    """
    Blend the SR image with the colored segmentation mask.
    Matches notebook Cell 4:
        overlay = 0.55 * original + 0.45 * mask
    """
    overlay = (
        alpha * sr_array.astype(np.float32)
        + (1.0 - alpha) * mask_color_array.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)
    return overlay


# ── Statistics ────────────────────────────────────────────────────────────────

def calculate_class_percentages(mask: np.ndarray) -> list:
    """
    Calculate the percentage of pixels belonging to each class.
    Excludes the Ignore class (ID 0) from the denominator.
    Returns a list of dicts sorted by descending percentage.
    """
    unique, counts = np.unique(mask, return_counts=True)
    count_map = dict(zip(unique.tolist(), counts.tolist()))

    # Total valid pixels (excluding Ignore)
    ignore_count = count_map.get(IGNORE_CLASS_ID, 0)
    total_valid = mask.size - ignore_count
    if total_valid <= 0:
        total_valid = mask.size  # fallback

    results = []
    for class_id, label in CLASS_LABELS.items():
        if class_id == IGNORE_CLASS_ID:
            continue
        pixel_count = count_map.get(class_id, 0)
        percentage = round((pixel_count / total_valid) * 100, 2)
        results.append({
            "class_id": class_id,
            "label": label,
            "color": CLASS_COLORS_HEX[class_id],
            "percentage": percentage,
        })

    results.sort(key=lambda x: x["percentage"], reverse=True)
    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _array_to_base64_png(img_array: np.ndarray) -> str:
    pil_img = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_segmentation(model, device, sr_pil_image: Image.Image) -> dict:
    """
    Full segmentation pipeline on the SR output image.
    Returns a dict ready to be embedded in the /predict JSON response.
    """
    # 1. Predict class-ID mask
    mask = predict_mask(model, device, sr_pil_image)

    # 2. Generate colored mask image
    mask_rgb = generate_mask_image(mask)

    # 3. Generate overlay
    sr_array = np.array(sr_pil_image)
    overlay_rgb = generate_overlay_image(sr_array, mask_rgb, alpha=0.55)

    # 4. Calculate class percentages
    class_stats = calculate_class_percentages(mask)

    # 5. Encode images to base64
    return {
        "available": True,
        "mask": _array_to_base64_png(mask_rgb),
        "overlay": _array_to_base64_png(overlay_rgb),
        "classes": class_stats,
    }
