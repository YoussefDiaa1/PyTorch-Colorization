# PyTorch Autoencoder Colorization Project (STL-10)

## 🎨 Project Overview

This project implements a deep learning solution for image colorization using a **Convolutional Autoencoder** architecture built with **PyTorch**. The goal is to colorize grayscale images from the **STL-10 dataset** with high fidelity, specifically addressing the common issue of "brownish" or desaturated color output often seen in simpler models.

The key to achieving superior color quality is the utilization of the **L*a*b* color space**, which separates the luminance (L) channel from the color information (a and b channels). The model is trained to predict the 'a' and 'b' color channels from the grayscale 'L' channel input.

The project is structured for professional presentation, including a clean codebase, a dedicated training script, and a user-friendly **Streamlit** web application for demonstration.

## ✨ Key Features

*   **L*a*b* Color Space Implementation:** Images are converted to L*a*b* space for training, which is crucial for decoupling brightness and color information, leading to more vibrant and accurate colorization.
*   **Convolutional Autoencoder:** A robust encoder-decoder architecture is used for feature extraction and color prediction.
*   **STL-10 Dataset:** Utilizes the large unlabeled split of the STL-10 dataset for self-supervised training.
*   **Professional Structure:** Organized into `models`, `utils`, `weights`, and `app` directories for maintainability and clarity.
*   **Streamlit Deployment:** A simple web application for real-time demonstration of the colorization model.

## 🚀 Project Structure

```
Colorization_Project/
├── app/
│   └── streamlit_app.py     # Streamlit web application for deployment
├── models/
│   └── colorization_model.py # PyTorch Autoencoder model definition
├── utils/
│   ├── data_loader.py       # STL-10 data loading and L*a*b* preprocessing
│   └── color_utils.py       # L*a*b* <-> RGB conversion utilities
├── weights/                 # Directory for saving model checkpoints (.pth)
├── runs/                    # Directory for TensorBoard logs
├── train.py                 # Main script for training the model
├── requirements.txt         # List of Python dependencies
└── README.md                # This file
```

## 🛠️ Setup and Installation

This project is designed to be run on a platform like Google Colab or any environment with GPU access.

### Prerequisites

*   Python 3.8+
*   `pip`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Colorization_Project.git
    cd Colorization_Project
    ```

2.  **Create `requirements.txt`:**
    ```bash
    echo "torch>=1.10.0" > requirements.txt
    echo "torchvision>=0.11.0" >> requirements.txt
    echo "opencv-python>=4.5.0" >> requirements.txt
    echo "numpy>=1.21.0" >> requirements.txt
    echo "Pillow>=8.4.0" >> requirements.txt
    echo "streamlit>=1.0.0" >> requirements.txt
    echo "tensorboard>=2.7.0" >> requirements.txt
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage

### 1. Training the Model

The `train.py` script handles data downloading (STL-10), model initialization, training, and checkpoint saving.

```bash
python train.py
```

*   **Note:** The STL-10 dataset is large (~2.6 GB) and will be downloaded to the `./data` directory on the first run.
*   Model checkpoints will be saved in the `./weights` directory. The latest model will be saved as `latest_model.pth`.

### 2. Running the Streamlit App

Once you have a trained model checkpoint (`latest_model.pth` in the `weights` folder), you can run the Streamlit application for demonstration.

```bash
streamlit run app/streamlit_app.py
```

*   The application will start on a local port (usually `8501`).
*   Upload a grayscale image, and the model will colorize it in real-time.

## 💡 Presentation Tips for Juries

To impress your jury, focus on the **technical decisions** that led to improved results:

1.  **The L*a*b* Color Space:** Emphasize that using L*a*b* is a superior approach to RGB for colorization. Explain how separating Luminance (L) from Chrominance (a, b) allows the model to focus purely on learning color, which directly solves the "brownish photo" problem (a symptom of the model struggling to learn color in the coupled RGB space).
2.  **Autoencoder Architecture:** Briefly explain the Encoder-Decoder structure and how it compresses the image features before reconstructing the color channels. Mention the use of **Batch Normalization** and **ReLU** for stable training.
3.  **Data Strategy:** Highlight the use of the large **unlabeled** portion of the STL-10 dataset, which is a common and effective strategy for self-supervised tasks like colorization.
4.  **Professional Deployment:** The Streamlit app demonstrates a complete, end-to-end project, moving beyond just a training script to a deployable product. This shows maturity and practical application of your work.

---
*Project maintained by [Your Name/Team Name]*
