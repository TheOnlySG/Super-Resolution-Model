"""Load the pretrained SegFormer-B2 (LoveDA) model from local weights."""

import torch
from transformers import SegformerForSemanticSegmentation
from model.segmentation.config import SEG_MODEL_DIR


def load_segmentation_model(device="cpu"):
    """
    Load SegFormer from the local model directory.
    Returns (model, device_str).
    
    The model directory must contain:
      - config.json
      - pytorch_model.bin
    """
    model = SegformerForSemanticSegmentation.from_pretrained(
        SEG_MODEL_DIR,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return model, device
