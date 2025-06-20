# scripts/run_netisg.py
import argparse
import os
import sys
import torch

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from netisg.config import Config, TrainerConfig
from netisg.image_processor import lift_image_to_3d
from netisg.skeletonizer import generate_skeleton
from netisg.visualizer import visualize_3d_result
from netisg.model import PointNetOffsetPredictor

def main():
    parser = argparse.ArgumentParser(description="Run the NETISG-XII Universal Deep Learning Framework.")
    parser.add_argument("--input", type=str, required=True, help="Path to any input image file.")
    args = parser.parse_args()

    print("--- Initializing NETISG-XII: The Universal Deep Hybrid Framework ---", flush=True)
    
    config = Config()
    train_config = TrainerConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Trained Model ---
    print(f"🧠 Loading trained model from {train_config.model_checkpoint} onto {device}...", flush=True)
    model = PointNetOffsetPredictor(num_points=train_config.num_points).to(device)
    
    if not os.path.exists(train_config.model_checkpoint):
        print(f"❌ Error: Model checkpoint not found at {train_config.model_checkpoint}", flush=True)
        print("Please run the training script first: python scripts/train.py", flush=True)
        sys.exit(1)
        
    model.load_state_dict(torch.load(train_config.model_checkpoint, map_location=device))
    model.eval()
    print("✅ Model loaded successfully.", flush=True)

    try:
        # Step 1: Lift image to a 3D point cloud
        print(f"📷 Loading and lifting image '{args.input}' to 3D point cloud...", flush=True)
        point_cloud = lift_image_to_3d(args.input, config)
        if point_cloud is None or len(point_cloud.points) == 0:
            print("❌ Error: Point cloud is empty after lifting. Exiting.", flush=True)
            sys.exit(1)
        print(f"✅ Point cloud created with {len(point_cloud.points)} points.", flush=True)

        # Step 2: Generate the skeleton using the TRAINED Deep Offset Prediction engine
        print("🦴 Generating skeleton from point cloud...", flush=True)
        skeleton_points, skeleton_topology = generate_skeleton(point_cloud, model, config, device)
        print("✅ Skeleton generation complete.", flush=True)

        # Step 3: Visualize the final result
        print("👁️ Launching 3D visualization window...", flush=True)
        visualize_3d_result(point_cloud, skeleton_points, skeleton_topology)
        print("\n--- NETISG-XII Pipeline Completed Successfully! ---", flush=True)

    except Exception as e:
        print(f"\n❌ An error occurred during the pipeline: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()




