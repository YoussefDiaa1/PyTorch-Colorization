import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import cv2

# Custom transform to convert RGB to L*a*b*
class RGBToLab(object):
    """Convert a PIL Image from RGB to L*a*b* color space."""
    def __call__(self, img):
        # Convert PIL Image to numpy array (H, W, C)
        img_np = np.array(img)
        # Convert RGB to L*a*b* using OpenCV
        lab_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        # Normalize L channel to [0, 1] (L is 0-100)
        # Normalize a and b channels to [-1, 1] (a/b are -128 to 127)
        L = lab_img[:, :, 0:1] / 100.0
        ab = (lab_img[:, :, 1:] + 128) / 255.0 * 2 - 1 # Scale to [-1, 1]

        # Stack channels back (H, W, C)
        lab_img_norm = np.concatenate([L, ab], axis=2)
        
        # Convert back to PIL Image (optional, but good for chaining transforms)
        # For PyTorch, we'll convert to tensor next, so we can skip PIL conversion
        
        # Convert to PyTorch tensor and permute to (C, H, W)
        lab_tensor = torch.from_numpy(lab_img_norm).float().permute(2, 0, 1)
        
        # The input to the model will be the L channel (grayscale)
        # The target will be the a and b channels
        L_channel = lab_tensor[0:1, :, :]
        ab_channels = lab_tensor[1:, :, :]
        
        return L_channel, ab_channels

def custom_collate_fn(batch):
    """
    Custom collate function to handle the (L, ab) tuple returned by the transform
    and the (image, label) tuple returned by the STL10 dataset.
    """
    # batch is a list of [((L, ab), label), ...]
    
    # Separate the L, ab channels and the labels
    L_ab_tuples = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Separate L and ab channels
    L_channels = [item[0] for item in L_ab_tuples]
    ab_channels = [item[1] for item in L_ab_tuples]
    
    # Stack them into tensors
    L_batch = torch.stack(L_channels)
    ab_batch = torch.stack(ab_channels)
    labels_batch = torch.tensor(labels)
    
    # Return the L channel (input) and ab channels (target)
    return L_batch, ab_batch, labels_batch

def get_stl10_dataloaders(batch_size=64, num_workers=4):
    """
    Loads the STL-10 dataset and returns training and testing DataLoaders.
    The images are transformed to L*a*b* color space.
    """
    
    # Standard transformations for STL-10 (96x96 images)
    # We only need to convert to tensor and apply our custom L*a*b* transform
    transform = transforms.Compose([
        transforms.Resize(96), # STL-10 is already 96x96, but good practice
        transforms.ToTensor(), # Converts to [0, 1] float tensor (C, H, W)
        transforms.ToPILImage(), # Convert back to PIL for cv2/numpy conversion
        RGBToLab(), # Custom transform to L*a*b* and split L/ab
    ])

    # The STL10 dataset returns a PIL Image, which is what our transform expects.
    # We use the 'unlabeled' split as it's much larger and suitable for self-supervised tasks like colorization.
    # The dataset returns (image, label), but we only care about the image.
    
    # STL-10 Unlabeled Dataset
    unlabeled_dataset = datasets.STL10(
        root='./Colorization_Project/data', 
        split='unlabeled', 
        download=True, 
        transform=transforms.Compose([
            transforms.ToTensor(), # Convert to [0, 1] float tensor (C, H, W)
            transforms.ToPILImage(), # Convert back to PIL for cv2/numpy conversion
            RGBToLab(), # Custom transform to L*a*b* and split L/ab
        ])
    )
    
    # We will split the unlabeled dataset into training and validation sets
    train_size = int(0.9 * len(unlabeled_dataset))
    val_size = len(unlabeled_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(unlabeled_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    print(f"Total dataset size: {len(unlabeled_dataset)}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    return train_loader, val_loader

if __name__ == '__main__':
    # Example usage and verification
    print("Testing data loader...")
    
    # Dummy check for the transform output shape
    # The actual check will be done after installing dependencies
    
    # To avoid dependency issues during file creation, we'll just print the function signature
    print("Data loader function created: get_stl10_dataloaders(batch_size=64, num_workers=4)")
    print("Next step: Install dependencies (PyTorch, torchvision, opencv-python) and test the loader.")
