# netisg/visualizer.py
import open3d as o3d

def visualize_3d_result(
    original_pcd: o3d.geometry.PointCloud, 
    skeleton: o3d.geometry.PointCloud, 
    topology: o3d.geometry.LineSet
):
    """
    Creates an impressive, interactive 3D visualization of the results.
    """
    skeleton.paint_uniform_color([1.0, 0.0, 0.0]) # Bright Red
    topology.paint_uniform_color([0.0, 1.0, 1.0]) # Bright Cyan

    print("🚀 Launching interactive 3D visualizer... (Close the window to exit)")
    o3d.visualization.draw_geometries(
        [original_pcd, skeleton, topology],
        window_name="NETISG-XII Result",
        width=1600,
        height=900
    )

