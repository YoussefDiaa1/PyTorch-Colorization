import torch
import numpy as np
import cv2
from PIL import Image

def lab_to_rgb(L_channel, ab_channels):
    """
    Converts L*a*b* tensors back to an RGB PIL Image.
    
    Args:
        L_channel (torch.Tensor): The L channel tensor (1, H, W) or (B, 1, H, W).
        ab_channels (torch.Tensor): The a and b channels tensor (2, H, W) or (B, 2, H, W).
        
    Returns:
        PIL.Image: The resulting color image.
    """
    # Ensure we are working with a single image (remove batch dimension if present)
    if L_channel.dim() == 4:
        L_channel = L_channel.squeeze(0)
    if ab_channels.dim() == 4:
        ab_channels = ab_channels.squeeze(0)
        
    # Move to CPU and convert to numpy
    L_np = L_channel.cpu().numpy().transpose(1, 2, 0) # (H, W, 1)
    ab_np = ab_channels.cpu().numpy().transpose(1, 2, 0) # (H, W, 2)
    
    # Denormalize L channel: [0, 1] -> [0, 100]
    L_np = L_np * 100.0
    
    # Denormalize a and b channels: [-1, 1] -> [-128, 127]
    # ab_channels were normalized as: (lab[:, :, 1:] + 128) / 255.0 * 2 - 1
    # Inverse: (ab_np + 1) / 2 * 255.0 - 128
    ab_np = (ab_np + 1) / 2 * 255.0 - 128
    
    # Concatenate L, a, and b channels
    lab_img = np.concatenate([L_np, ab_np], axis=2).astype(np.uint8)
    
    # Convert L*a*b* to RGB using OpenCV
    rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
    
    # Convert numpy array to PIL Image
    return Image.fromarray(rgb_img)

def preprocess_image(image):
    """
    Preprocesses a PIL Image for model input (RGB -> L channel tensor).
    """
    # Resize and convert to numpy
    image = image.resize((96, 96))
    img_np = np.array(image)
    
    # Convert RGB to L*a*b* using OpenCV
    lab_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    
    # Normalize L channel to [0, 1]
    L = lab_img[:, :, 0:1] / 100.0
    
    # Convert to PyTorch tensor and permute to (C, H, W)
    L_tensor = torch.from_numpy(L).float().permute(2, 0, 1)
    
    # Add batch dimension
    return L_tensor.unsqueeze(0)

if __name__ == '__main__':
    # Dummy test for the utility functions
    print("Color utility functions created.")
    print("lab_to_rgb(L_channel, ab_channels) and preprocess_image(image) are ready.")
