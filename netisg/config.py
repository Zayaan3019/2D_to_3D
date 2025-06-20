# netisg/config.py
from dataclasses import dataclass, field
import os

@dataclass
class Config:
    # --- Image Processing & 3D Lifting ---
    POINT_CLOUD_SAMPLES: int = 15000

    # --- Deep Offset Prediction Network ---
    K_NEIGHBORS: int = 16
    CONTRACTION_STEPS: int = 10
    CONTRACTION_STRENGTH: float = 0.2

    # --- Final Skeleton Refinement ---
    FINAL_SKELETON_VOXEL_SIZE: float = 0.02
    
@dataclass
class TrainerConfig:
    # --- Dataset and Dataloader ---
    # This path MUST point to the root of your RigNet dataset
    dataset_path: str = r"C:/Users/Mohamed Zayaan/Downloads/RigNet_Kaggle"
    num_workers: int = 4
    num_points: int = 2048 # Number of points to sample from each mesh

    # --- Training Parameters ---
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 0.001
    
    # --- Checkpoints ---
    checkpoint_dir: str = "./checkpoints"
    model_checkpoint: str = field(init=False)

    def __post_init__(self):
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.model_checkpoint = os.path.join(self.checkpoint_dir, "netisg_model.pth")











