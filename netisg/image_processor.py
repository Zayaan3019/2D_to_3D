# netisg/image_processor.py
import cv2
import numpy as np
import open3d as o3d
from rembg import remove
from PIL import Image
import io
from .config import Config

def lift_image_to_3d(image_path: str, config: Config) -> o3d.geometry.PointCloud:
    """
    Lifts any 2D image to a 3D point cloud using universal background removal.
    """
    print("🚀 Stage 1: Performing Universal Semantic Segmentation...")
    
    with open(image_path, 'rb') as i:
        input_data = i.read()
    output_data = remove(input_data)
    
    output_image = Image.open(io.BytesIO(output_data)).convert("RGBA")
    mask = np.array(output_image)[:, :, 3]
    
    h, w = mask.shape
    y_coords, x_coords = np.where(mask > 0)

    if len(x_coords) == 0:
        raise ValueError("Segmentation failed. No foreground object found.")

    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    z_coords = dist_transform[y_coords, x_coords]
    
    points = np.vstack((x_coords, -y_coords, z_coords)).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Downsample if the cloud is too dense
    if len(pcd.points) > config.POINT_CLOUD_SAMPLES:
        pcd = pcd.farthest_point_down_sample(config.POINT_CLOUD_SAMPLES)
    
    # Also get colors from the original image for impressive visualization
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Could not read the original image for coloring at: {image_path}")

    # --- FIX APPLIED HERE ---
    # First, convert the Open3D Vector3dVector to a NumPy array.
    points_np = np.asarray(pcd.points)
    
    # Now, we can perform NumPy-style indexing on the new `points_np` array.
    # We also clip the indices to ensure they are within the image bounds.
    y_indices = np.clip((-points_np[:, 1]).astype(int), 0, h - 1)
    x_indices = np.clip(points_np[:, 0].astype(int), 0, w - 1)
    
    colors = original_image[y_indices, x_indices]
    
    # Assign the correctly sampled colors back to the point cloud.
    # Open3D expects RGB colors in the range [0, 1].
    pcd.colors = o3d.utility.Vector3dVector(colors[:, ::-1] / 255.0) # BGR to RGB and scale

    print(f"✅ Separated object and lifted to a 3D point cloud with {len(pcd.points)} points.")
    return pcd

