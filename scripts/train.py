# scripts/train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from netisg.config import TrainerConfig
from netisg.model import PointNetOffsetPredictor
from netisg.data_loader import get_dataloader

def train():
    """Main training loop for the NETISG framework."""
    config = TrainerConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting NETISG Training on {device} ---")

    # --- Setup Model ---
    model = PointNetOffsetPredictor(num_points=config.num_points).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()

    # --- Setup Dataloaders ---
    train_loader = get_dataloader(config, split='train')
    # val_loader = get_dataloader(config, split='val') # Optional: for validation

    # --- Training Loop ---
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch in progress_bar:
            points = batch["points"].to(device)
            target_offsets = batch["target_offsets"].to(device)

            # Reshape points for Conv1d: (B, N, C) -> (B, C, N)
            points_transposed = points.transpose(1, 2)
            
            # --- Forward Pass ---
            optimizer.zero_grad()
            predicted_offsets = model(points_transposed)
            
            # --- Loss Calculation ---
            loss = criterion(predicted_offsets, target_offsets)
            
            # --- Backward Pass ---
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Complete. Average Training Loss: {avg_loss:.6f}")

        # --- Save Model Checkpoint ---
        torch.save(model.state_dict(), config.model_checkpoint)
        print(f"✅ Model saved to {config.model_checkpoint}")

    print("--- Training Finished ---")

if __name__ == "__main__":
    train()

