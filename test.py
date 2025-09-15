import open3d as o3d

def visualize_pcd(pcd_path: str):
    """
    Read and visualize a point cloud file (.pcd).
    
    Args:
        pcd_path (str): Path to the .pcd file
    """
    # Load
    pcd = o3d.io.read_point_cloud(pcd_path)

    if not pcd.has_points():
        print(f"[!] No points found in {pcd_path}")
        return

    # Visualize
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="PCD Viewer",
        width=800,
        height=600,
        left=50,
        top=50,
        point_show_normal=False
    )

if __name__ == "__main__":
    visualize_pcd("archive/2025_09_12_Hand_DDIM/easy_check_tray_img_grad_scale1e-1_clamp1e-2_k3_thresh96_grippinglossnew_frame150_seed310_alpha_20_ddim256/softzoo/design/geometry_Batch_0_Sampling_0075_Local_0000.pcd")
