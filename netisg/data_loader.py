# netisg/data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import open3d as o3d
from .config import TrainerConfig

def normalize_point_cloud(points):
    """Normalizes a point cloud to fit into a unit sphere centered at the origin."""
    centroid = np.mean(points, axis=0)
    points -= centroid
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 1e-6:
        points /= max_dist
    return points, centroid, max_dist

class RigNetDataset(Dataset):
    """
    Dataset for loading 3D models and skeletons from the RigNet dataset,
    parsing skeleton data from individual .txt files in 'rig_info_remesh'.
    """
    def __init__(self, config: TrainerConfig, split='train'):
        self.root_dir = config.dataset_path
        self.num_points = config.num_points
        self.split = split
        
        self.split_file = os.path.join(self.root_dir, f"{self.split}_final.txt")

        # --- Validate Dataset Structure ---
        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"Split file not found at {self.split_file}. Check dataset_path in config.")

        print(f"--- Loading dataset for '{self.split}' split ---")
        
        # --- Load Model Names ---
        with open(self.split_file, 'r') as f:
            self.model_names = [line.strip() for line in f.readlines()]
        print(f"Found {len(self.model_names)} models in {os.path.basename(self.split_file)}.")

        # --- Define directory paths ---
        self.mesh_dir = os.path.join(self.root_dir, "obj_remesh")
        self.rig_dir = os.path.join(self.root_dir, "rig_info_remesh")

        if not os.path.exists(self.rig_dir):
            raise FileNotFoundError(f"Rig info directory not found at {self.rig_dir}")

    def _parse_rig_txt(self, rig_path):
        """Parses a .txt rig file to extract joint positions."""
        joints = []
        if not os.path.exists(rig_path):
            return np.array([])
            
        with open(rig_path, 'r') as f:
            for line in f:
                if line.strip().startswith('joint'):
                    parts = line.strip().split()
                    # Format is: "joint joint_name X Y Z"
                    xyz = [float(parts[2]), float(parts[3]), float(parts[4])]
                    joints.append(xyz)
        return np.array(joints)

    def __len__(self):
        return len(self.model_names)

    def __getitem__(self, idx):
        model_name = self.model_names[idx]
        
        # --- Load Ground Truth Skeleton from .txt file ---
        rig_path = os.path.join(self.rig_dir, f"{model_name}.txt")
        gt_skeleton_points = self._parse_rig_txt(rig_path)

        # --- Load Mesh and Sample Point Cloud ---
        mesh_path = os.path.join(self.mesh_dir, f"{model_name}.obj")
        mesh = o3d.io.read_triangle_mesh(mesh_path)

        # --- Data Validity Check ---
        # Skip if mesh is invalid or skeleton data is missing
        if not mesh.has_triangles() or len(mesh.vertices) == 0 or gt_skeleton_points.shape[0] == 0:
            print(f"Warning: Skipping invalid data for model {model_name}. Mesh has {len(mesh.vertices)} vertices, skeleton has {gt_skeleton_points.shape[0]} joints.")
            return self.__getitem__((idx + 1) % len(self))
            
        pcd = mesh.sample_points_uniformly(number_of_points=self.num_points)
        points = np.asarray(pcd.points)
        
        # --- Normalize both to the same space ---
        points, centroid, scale = normalize_point_cloud(points)
        if scale > 1e-6:
            gt_skeleton_points = (gt_skeleton_points - centroid) / scale
        else:
            gt_skeleton_points = gt_skeleton_points - centroid

        # --- Find Target Offsets ---
        pcd_tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_skeleton_points)))
        target_offsets = np.zeros_like(points)
        for i, point in enumerate(points):
            _, k_idx, _ = pcd_tree.search_knn_vector_3d(point, 1)
            closest_skel_point = gt_skeleton_points[k_idx[0]]
            target_offsets[i] = closest_skel_point - point

        return {
            "points": torch.from_numpy(points).float(),
            "target_offsets": torch.from_numpy(target_offsets).float(),
        }

def get_dataloader(config: TrainerConfig, split='train'):
    dataset = RigNetDataset(config, split=split)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=(split == 'train'),
        drop_last=True
    )






