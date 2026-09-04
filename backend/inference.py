import io
import torch
import numpy as np
from PIL import Image
from backend.model import RCAN

# Fallback normalization value from notebook if not present in checkpoint
DEFAULT_NORMALIZATION = 3000.0

def load_model(checkpoint_path, device="cpu"):
    """
    Load the RCAN model from a checkpoint dictionary.
    Returns the initialized model, device, and normalization factor.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # The metadata in the checkpoint is actually incorrect (it says 20 blocks, 128 channels)
    # but the state_dict contains 12 blocks and 96 channels. We'll hardcode what the state_dict expects.
    num_blocks = 12
    channels = 96
    normalization = checkpoint.get("normalization", DEFAULT_NORMALIZATION)
    
    print(f"Loading RCAN(num_blocks={num_blocks}, channels={channels})")
    
    model = RCAN(num_blocks=num_blocks, channels=channels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, device, normalization

def preprocess_image(image_bytes, normalization=DEFAULT_NORMALIZATION):
    """
    Preprocess raw image bytes into a [1, 4, H, W] float32 tensor.
    If the image is 3-channel (RGB), we duplicate the Red channel to synthesize a 4th NIR channel.
    """
    # Open image
    img = Image.open(io.BytesIO(image_bytes))
    
    # Convert to numpy array
    img_np = np.array(img).astype(np.float32)
    
    # Check if we have 3 channels or 4 channels
    if len(img_np.shape) == 2:
        # Grayscale, duplicate to 4 channels
        img_np = np.stack((img_np,)*4, axis=-1)
    elif img_np.shape[2] == 3:
        # RGB, duplicate Red to 4th channel (synthesize NIR)
        nir = img_np[:, :, 0:1] # Red channel
        img_np = np.concatenate([img_np, nir], axis=2)
    elif img_np.shape[2] > 4:
        # Take first 4 channels
        img_np = img_np[:, :, :4]
        
    # Scale from 0-255 to the model's expected scale (originally 0-3000 for sentinel)
    # The model expects inputs scaled down by the normalization factor.
    # If the user uploads a regular 8-bit RGB image, it's in 0-255. 
    # Let's scale it so it roughly matches the expected input range.
    # Actually, we should just divide by 255.0 to get it into [0, 1].
    # But wait, the notebook divided by 3000.0, so the input to the model was in roughly [0, 1] range anyway (since reflectance maxes out around 3000-4000).
    # Since standard images are 0-255, dividing by 255.0 gives [0, 1].
    img_np = img_np / 255.0
    
    # Convert to C, H, W
    img_np = np.transpose(img_np, (2, 0, 1))
    
    # Add batch dimension
    tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)
    
    return tensor

def stretch_rgb(img_np):
    """
    Applies 2-98 percentile contrast stretching to the RGB channels.
    img_np shape: (3, H, W)
    """
    output = np.zeros_like(img_np)
    for c in range(3):
        low = np.percentile(img_np[c, :, :], 2)
        high = np.percentile(img_np[c, :, :], 98)
        output[c, :, :] = np.clip((img_np[c, :, :] - low) / (high - low + 1e-8), 0, 1)
    return output

def postprocess_output(tensor):
    """
    Convert the output tensor back to an image (bytes) ready to return via API.
    """
    # Remove batch dim and move to CPU
    out_tensor = tensor.squeeze(0).cpu()
    
    # Clamp to [0, 1]
    out_tensor = out_tensor.clamp(0, 1)
    
    # Convert to numpy
    out_np = out_tensor.numpy()
    
    # The model outputs 4 channels. We only want RGB for visualization (first 3 channels)
    rgb = out_np[:3, :, :]
    
    # Apply percentile stretching (as in the notebook)
    stretched = stretch_rgb(rgb)
    
    # Convert back to H, W, C and scale to 0-255
    img_array = (np.transpose(stretched, (1, 2, 0)) * 255).astype(np.uint8)
    
    # Convert to PIL Image
    pil_img = Image.fromarray(img_array)
    
    # Save to bytes
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return img_byte_arr

def run_inference(model, image_bytes, device):
    """
    End-to-end inference for a single image.
    """
    # 1. Preprocess
    input_tensor = preprocess_image(image_bytes)
    input_tensor = input_tensor.to(device)
    
    # 2. Inference
    with torch.no_grad():
        output_tensor = model(input_tensor)
        
    # 3. Postprocess
    output_bytes = postprocess_output(output_tensor)
    
    return output_bytes
