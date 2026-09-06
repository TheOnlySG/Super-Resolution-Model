import os
import numpy as np

# Path to the local model directory containing config.json, preprocessor_config.json, pytorch_model.bin
SEG_MODEL_DIR = os.path.join(os.path.dirname(__file__))

# Class ID 0 is "Ignore" and should be excluded from legend/stats
IGNORE_CLASS_ID = 0

# Class labels from the model's config.json id2label mapping
CLASS_LABELS = {
    0: "Ignore",
    1: "Background",
    2: "Building",
    3: "Road",
    4: "Water",
    5: "Barren",
    6: "Forest",
    7: "Agricultural",
}

# Exact palette from the verified notebook (Cell 4)
# Each entry is an RGB tuple matching the notebook's numpy array
CLASS_COLORS_RGB = {
    0: (0, 0, 0),         # Ignore — black
    1: (30, 30, 30),      # Background — dark gray
    2: (255, 0, 0),       # Building — red
    3: (255, 255, 0),     # Road — yellow
    4: (0, 120, 255),     # Water — blue
    5: (180, 120, 60),    # Barren — brown
    6: (0, 180, 0),       # Forest — green
    7: (255, 180, 0),     # Agricultural — orange
}

# Hex versions for frontend legend
CLASS_COLORS_HEX = {
    k: "#{:02x}{:02x}{:02x}".format(*v)
    for k, v in CLASS_COLORS_RGB.items()
}

# Numpy palette array for fast mask coloring (identical to notebook Cell 4)
PALETTE = np.array([
    [0,   0,   0],       # 0 Ignore
    [30,  30,  30],      # 1 Background
    [255, 0,   0],       # 2 Building
    [255, 255, 0],       # 3 Road
    [0,   120, 255],     # 4 Water
    [180, 120, 60],      # 5 Barren
    [0,   180, 0],       # 6 Forest
    [255, 180, 0],       # 7 Agricultural
], dtype=np.uint8)
