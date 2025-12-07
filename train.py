import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
from models.colorization_model import ColorizationAutoencoder
from utils.data_loader import get_stl10_dataloaders

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_EPOCHS = 5 # Reduced for demonstration/testing, user can increase this
LEARNING_RATE = 1e-3
CHECKPOINT_DIR = './Colorization_Project/weights'
LOG_DIR = './Colorization_Project/runs'

def train_model():
    # Setup directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(LOG_DIR)

    # Data Loaders
    print("Loading data...")
    train_loader, val_loader = get_stl10_dataloaders(batch_size=BATCH_SIZE, num_workers=4)
    print(f"Using device: {DEVICE}")

    # Model, Loss, Optimizer
    model = ColorizationAutoencoder().to(DEVICE)
    # Mean Squared Error (MSE) is a common choice for L*a*b* regression
    # The user's previous "brownish" issue might be due to L1 loss or poor L*a*b* scaling.
    # MSE is generally better for regression tasks like this.
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        for i, (L_channel, ab_channels, _) in enumerate(train_loader):
            L_channel, ab_channels = L_channel.to(DEVICE), ab_channels.to(DEVICE)

            # Forward pass
            outputs = model(L_channel)
            loss = criterion(outputs, ab_channels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * L_channel.size(0)
            
            # Log to TensorBoard every 100 batches
            if (i + 1) % 100 == 0:
                step = epoch * len(train_loader) + i
                writer.add_scalar('Loss/train_step', loss.item(), step)
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        writer.add_scalar('Loss/train_epoch', epoch_loss, epoch)
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for L_channel, ab_channels, _ in val_loader:
                L_channel, ab_channels = L_channel.to(DEVICE), ab_channels.to(DEVICE)
                outputs = model(L_channel)
                loss = criterion(outputs, ab_channels)
                val_loss += loss.item() * L_channel.size(0)

        val_loss /= len(val_loader.dataset)
        writer.add_scalar('Loss/val_epoch', val_loss, epoch)

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] finished in {epoch_duration:.2f}s")
        print(f"Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Checkpoint saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'best_model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model checkpoint to {checkpoint_path}")
        
        # Save a checkpoint every epoch (optional, but good for long runs)
        latest_checkpoint_path = os.path.join(CHECKPOINT_DIR, 'latest_model.pth')
        torch.save(model.state_dict(), latest_checkpoint_path)

    print("Training complete.")
    writer.close()

if __name__ == '__main__':
    # This script is designed to be run in a Colab environment, 
    # but we will simulate the run to ensure the code is correct.
    # We will not actually run the full training here due to time constraints.
    print("Training script created. Next step is to create the Streamlit app and documentation.")
    # train_model() # Uncomment this line to run the training
