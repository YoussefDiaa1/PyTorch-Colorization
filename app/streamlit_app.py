import streamlit as st
import torch
from PIL import Image
import numpy as np
import os
import sys

# Add project root to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.colorization_model import ColorizationAutoencoder
from utils.color_utils import lab_to_rgb, preprocess_image

# --- Configuration ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'weights', 'latest_model.pth')
DEVICE = torch.device("cpu") # Streamlit apps typically run on CPU

@st.cache_resource
def load_model():
    """Loads the trained model and sets it to evaluation mode."""
    try:
        model = ColorizationAutoencoder().to(DEVICE)
        # Load the state dict, ensuring it's mapped to the CPU
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at: {MODEL_PATH}. Please train the model first.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def colorize_image(model, L_tensor):
    """
    Colorizes the grayscale L channel tensor using the model.
    
    Args:
        model (nn.Module): The trained colorization model.
        L_tensor (torch.Tensor): The preprocessed L channel tensor (1, 1, 96, 96).
        
    Returns:
        PIL.Image: The colorized image.
    """
    with torch.no_grad():
        # Predict the a and b channels
        ab_tensor = model(L_tensor.to(DEVICE))
        
        # Convert L and ab channels back to RGB image
        # L_tensor is (1, 1, 96, 96), ab_tensor is (1, 2, 96, 96)
        colorized_image = lab_to_rgb(L_tensor, ab_tensor)
        return colorized_image

# --- Streamlit App ---
st.set_page_config(page_title="PyTorch Colorization Autoencoder", layout="wide")

st.title("🎨 PyTorch Colorization Autoencoder")
st.markdown("""
This application uses a trained Autoencoder model to colorize grayscale images.
The model was trained on the STL-10 dataset using the **L*a*b* color space** to achieve better color results and avoid the "brownish" output common with RGB-based models.
""")

# Load the model
model = load_model()

if model:
    st.sidebar.header("Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose a grayscale image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file).convert("RGB")
        
        # Preprocess the image
        L_tensor = preprocess_image(image)
        
        # Colorize
        colorized_image = colorize_image(model, L_tensor)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Original Grayscale (L Channel)")
            # Convert L channel tensor back to a displayable grayscale image
            L_display_np = L_tensor.squeeze().cpu().numpy() * 255.0
            L_display_img = Image.fromarray(L_display_np.astype(np.uint8), mode='L').convert('RGB')
            st.image(L_display_img, caption="Input (L Channel)", use_column_width=True)
            
        with col2:
            st.header("Colorized Result (L*a*b* -> RGB)")
            st.image(colorized_image, caption="Output (Colorized)", use_column_width=True)
            
        st.success("Colorization complete!")
    else:
        st.info("Please upload an image to start colorization.")
        
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Model Status:** Loaded successfully")
    st.sidebar.markdown(f"**Device:** {DEVICE}")
    st.sidebar.markdown(f"**Model Path:** `{MODEL_PATH}`")
else:
    st.error("Cannot run the application without a loaded model. Please train the model first and ensure 'latest_model.pth' exists in the 'weights' folder.")
