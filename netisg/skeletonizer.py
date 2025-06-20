# netisg/skeletonizer.py
import torch
import numpy as np
import open3d as o3d
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from tqdm import tqdm
from .config import Config
from .model import PointNetOffsetPredictor # Import the actual model

@torch.no_grad()
def _predict_offsets(points: np.ndarray, model: PointNetOffsetPredictor, device) -> np.ndarray:
    """
    Uses the trained PointNet model to predict where each point should move.
    """
    model.eval()
    
    # Normalize points for the model
    centroid = np.mean(points, axis=0)
    points_norm = points - centroid
    scale = np.max(np.linalg.norm(points_norm, axis=1))
    if scale > 0:
        points_norm /= scale

    # Prepare batch for the model
    points_tensor = torch.from_numpy(points_norm).float().unsqueeze(0).to(device) # (1, N, 3)
    points_tensor = points_tensor.transpose(1, 2) # (1, 3, N)

    # Predict offsets
    predicted_offsets_norm = model(points_tensor) # (1, N, 3)
    
    # Denormalize offsets
    offsets = predicted_offsets_norm.squeeze(0).cpu().numpy() * scale
    return offsets

def generate_skeleton(point_cloud: o3d.geometry.PointCloud, model: PointNetOffsetPredictor, config: Config, device) -> tuple:
    """
    Generates a 3D skeleton using the trained Deep Offset Prediction pipeline.
    """
    print("🚀 Stage 2: Running Deep Offset Prediction and Contraction...")
    
    points = np.asarray(point_cloud.points)
    
    # Iteratively contract the point cloud using the model
    for i in tqdm(range(config.CONTRACTION_STEPS), desc="Contracting Point Cloud"):
        # The number of points might change, so we need to handle that.
        # For simplicity, we assume a fixed number of points matching the model's training.
        # A more advanced implementation would use subsampling/interpolation.
        if len(points) != model.num_points:
             # Simple resampling to match model input size
            indices = np.random.choice(len(points), model.num_points, replace=len(points) < model.num_points)
            points_subset = points[indices]
            offsets = _predict_offsets(points_subset, model, device)
            points[indices] += config.CONTRACTION_STRENGTH * offsets
        else:
            offsets = _predict_offsets(points, model, device)
            points += config.CONTRACTION_STRENGTH * offsets

    # Create a point cloud of the contracted (skeleton) points
    skeleton_pcd = o3d.geometry.PointCloud()
    skeleton_pcd.points = o3d.utility.Vector3dVector(points)
    
    # Downsample the dense skeleton to get clean nodes for the graph
    skeleton_nodes_pcd = skeleton_pcd.voxel_down_sample(config.FINAL_SKELETON_VOXEL_SIZE)
    nodes = np.asarray(skeleton_nodes_pcd.points)

    print("🚀 Stage 3: Building Final Topology with Minimum Spanning Tree...")
    dist_matrix = np.linalg.norm(nodes[:, np.newaxis, :] - nodes[np.newaxis, :, :], axis=2)
    graph = csr_matrix(dist_matrix)
    mst = minimum_spanning_tree(graph)
    
    lines = np.array(mst.nonzero()).T
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(nodes),
        lines=o3d.utility.Vector2iVector(lines)
    )
    
    print("✅ Skeleton and topology extracted successfully.")
    return skeleton_nodes_pcd, line_set






