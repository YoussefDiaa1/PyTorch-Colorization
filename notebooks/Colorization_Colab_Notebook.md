# PyTorch Autoencoder Colorization Project (STL-10) - Google Colab Guide

This notebook provides a step-by-step guide to set up, train, and run the Streamlit deployment for the colorization project in a Google Colab environment.

## 1. Setup and Installation

### 1.1. Check GPU Availability

Ensure you are running a GPU runtime. Go to `Runtime` -> `Change runtime type` and select `GPU` as the hardware accelerator.

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
```

### 1.2. Clone the Repository

Replace `your-username` with your actual GitHub username after you push the project. For now, we will use a placeholder.

```bash
# Clone the repository (assuming you have pushed the project to GitHub)
!git clone https://github.com/your-username/Colorization_Project.git
%cd Colorization_Project
```

### 1.3. Install Dependencies

We need to install PyTorch, Torchvision, OpenCV (for L*a*b* conversion), and Streamlit.

```bash
!pip install -r requirements.txt
```

### 1.4. Import Project Modules

Add the project directory to the Python path so we can import our custom modules.

```python
import sys
sys.path.append('/content/Colorization_Project')
```

## 2. Model Training

The `train.py` script will download the STL-10 dataset (approx. 2.6 GB) and start the training process.

**Note:** The training process can take a significant amount of time depending on the GPU. The script is configured to save checkpoints in the `weights/` directory.

```bash
!python train.py
```

**Expected Output:**
The script will print the dataset sizes, training progress (loss), and save the best model checkpoint as `weights/best_model_epoch_X.pth` and the latest as `weights/latest_model.pth`.

## 3. Running the Streamlit Application

We will use `ngrok` to expose the Streamlit application running on Colab to a public URL.

### 3.1. Install `ngrok`

```bash
!pip install pyngrok
```

### 3.2. Run the Streamlit App

This command will start the Streamlit app and use `ngrok` to create a public tunnel.

```bash
# Run the Streamlit app in the background
!nohup streamlit run app/streamlit_app.py &

# Wait a few seconds for the app to start
import time
time.sleep(5)

# Get the ngrok public URL
from pyngrok import ngrok
# You may need to set your ngrok auth token if you haven't already
# ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN") 

# Open a tunnel to the Streamlit port (8501 is the default)
public_url = ngrok.connect(8501)
print(f"Streamlit App Public URL: {public_url}")
```

**Instructions:**
1.  Click the provided `Streamlit App Public URL`.
2.  The web application will open in a new tab.
3.  Upload a grayscale image (e.g., from the STL-10 dataset or any other image) and see the colorized result!

## 4. Saving to GitHub

After training and testing, you can save your trained model weights and the entire project structure to your GitHub repository.

### 4.1. Configure Git

```bash
!git config --global user.email "your-email@example.com"
!git config --global user.name "your-username"
```

### 4.2. Commit and Push

**Important:** Before pushing, you should remove the large dataset files to keep your repository clean.

```bash
# Remove the downloaded STL-10 dataset (it's large!)
!rm -rf data/stl10_binary

# Add all project files
!git add .

# Commit changes
!git commit -m "Final project commit: Added trained model, Streamlit app, and Colab guide."

# Push to your repository (you will need to enter your GitHub credentials/token)
!git push origin main
```

This completes the professional setup for your presentation! Good luck with your juries!
