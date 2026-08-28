import time
import yaml
import numpy as np
import open3d as o3d
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mlx_db3 import load_session


def load_config(config_path="config.yaml"):
    """Loads YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# =========================================================
# MODULE: OBJECT DETECTION
# =========================================================

def detect_factory_tables(xyz_points, cfg_det, frame_idx=0):
    """
    Segmentation & Box Hierarchy driven by YAML configuration.
    - Red Box   : Entire combined structure
    - Green Box : Table Base
    - Cyan Box  : Steel Plate
    """
    start_time = time.perf_counter()
    
    # 1. Downsample
    t0 = time.perf_counter()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_points)
    pcd = pcd.voxel_down_sample(voxel_size=cfg_det["voxel_size"])
    downsample_time = time.perf_counter() - t0
    
    # 2. Ground RANSAC
    t1 = time.perf_counter()
    g_cfg = cfg_det["ground_ransac"]
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=g_cfg["distance_threshold"],
        ransac_n=g_cfg["ransac_n"],
        num_iterations=g_cfg["num_iterations"]
    )
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    inlier_cloud.paint_uniform_color([0.0, 0.4, 0.4])
    ransac_time = time.perf_counter() - t1
    
    # 3. PASS 1: Broad clustering
    t2 = time.perf_counter()
    p1_cfg = cfg_det["pass1_dbscan"]
    labels = np.array(outlier_cloud.cluster_dbscan(
        eps=p1_cfg["eps"],
        min_points=p1_cfg["min_points"],
        print_progress=False
    ))
    max_label = labels.max()
    cluster_time = time.perf_counter() - t2
    
    # 4. Box generation and Pass 2
    t3 = time.perf_counter()
    obbs = []
    
    if max_label >= 0:
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0 
        outlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
        outlier_points = np.asarray(outlier_cloud.points) 
        
        for i in range(max_label + 1):
            cluster_idx = np.where(labels == i)[0]
            
            if len(cluster_idx) > p1_cfg["min_cluster_points"]:
                cluster_points = outlier_points[cluster_idx]
                sub_cloud = o3d.geometry.PointCloud()
                sub_cloud.points = o3d.utility.Vector3dVector(cluster_points)
                
                if len(sub_cloud.points) > 0:
                    obb = sub_cloud.get_oriented_bounding_box()
                    dims = np.sort(obb.extent) 
                    
                    t_dim = cfg_det["table_dimensions"]

                    if (t_dim["height"]["min"] < dims[0] < t_dim["height"]["max"]) and \
                        (t_dim["width"]["min"] < dims[1] < t_dim["width"]["max"]) and \
                        (t_dim["length"]["min"] < dims[2] < t_dim["length"]["max"]):
                        
                        # Calculate the direction of the shortest axis
                        shortest_axis_idx = np.argmin(obb.extent)
                        shortest_axis_dir = obb.R[:, shortest_axis_idx]
                        
                        # Ensure the shortest axis is roughly vertical (Z-component is dominant).
                        if abs(shortest_axis_dir[2]) > 0.1:
                            
                            # 1. OVERALL BOUNDING BOX (RED)
                            obb.color = np.array([1.0, 0.0, 0.0])
                            obbs.append(obb)

                            if shortest_axis_dir[2] < 0: 
                                shortest_axis_dir = -shortest_axis_dir

                            # -------------------------------------------------
                            # PASS 2: Multi-Pass RANSAC
                            # -------------------------------------------------
                            p2_cfg = cfg_det["pass2_ransac"]
                            plane_model_1, inliers_1 = sub_cloud.segment_plane(
                                distance_threshold=p2_cfg["distance_threshold"],
                                ransac_n=p2_cfg["ransac_n"],
                                num_iterations=p2_cfg["num_iterations"]
                            )
                            plane_cloud_1 = sub_cloud.select_by_index(inliers_1)
                            remainder_cloud = sub_cloud.select_by_index(inliers_1, invert=True)

                            if len(remainder_cloud.points) > p2_cfg["min_points_remainder"]:
                                plane_model_2, inliers_2 = remainder_cloud.segment_plane(
                                    distance_threshold=p2_cfg["distance_threshold"],
                                    ransac_n=p2_cfg["ransac_n"],
                                    num_iterations=p2_cfg["num_iterations"]
                                )
                                plane_cloud_2 = remainder_cloud.select_by_index(inliers_2)
                                
                                pts1 = np.asarray(plane_cloud_1.points)
                                pts2 = np.asarray(plane_cloud_2.points)
                                
                                h1 = np.mean(np.dot(pts1 - obb.center, shortest_axis_dir))
                                h2 = np.mean(np.dot(pts2 - obb.center, shortest_axis_dir))
                                
                                if h1 < h2:
                                    table_pts, plate_pts = pts1, pts2
                                    h_table_mean, h_plate_mean = h1, h2
                                else:
                                    table_pts, plate_pts = pts2, pts1
                                    h_table_mean, h_plate_mean = h2, h1
                                    
                                red_bottom_h = -obb.extent[shortest_axis_idx] / 2.0

                                # 1. GREEN BOX: Table Surface & Base
                                drop_green_dist = h_table_mean - red_bottom_h
                                table_base_pts = table_pts - (drop_green_dist * shortest_axis_dir)
                                green_vol_pts = np.vstack((table_pts, table_base_pts))
                                
                                t_cloud = o3d.geometry.PointCloud()
                                t_cloud.points = o3d.utility.Vector3dVector(green_vol_pts)
                                t_obb = t_cloud.get_oriented_bounding_box()
                                t_obb.color = np.array([0.0, 1.0, 0.0])
                                obbs.append(t_obb)
                                
                                # 2. CYAN BOX: Steel Plate
                                table_max_h = np.max(np.dot(table_pts - obb.center, shortest_axis_dir))
                                drop_cyan_dist = h_plate_mean - table_max_h
                                plate_base_pts = plate_pts - (drop_cyan_dist * shortest_axis_dir)
                                cyan_vol_pts = np.vstack((plate_pts, plate_base_pts))
                                
                                p_cloud = o3d.geometry.PointCloud()
                                p_cloud.points = o3d.utility.Vector3dVector(cyan_vol_pts)
                                p_obb = p_cloud.get_oriented_bounding_box()
                                p_obb.color = np.array([0.0, 0.8, 1.0])
                                obbs.append(p_obb)
                                
                                # DISTANCE MEASUREMENT DISPLAY
                                distance_m = h_plate_mean - table_max_h
                                if distance_m > 0:
                                    print("\n" + "="*50)
                                    print(f" [MEASUREMENT DATA]")
                                    print(f"  ► Distance (Highest Table Point → Plate) : {distance_m:.3f} m ({distance_m * 100.0:.1f} cm)")
                                    print("="*50 + "\n")

    else:
        outlier_cloud.paint_uniform_color([0, 0, 0])
        
    filter_time = time.perf_counter() - t3
    total_time = time.perf_counter() - start_time
    fps = 1.0 / total_time if total_time > 0 else 0
    
    print(f"[Frame {frame_idx:03d}] Downsample: {downsample_time*1000:.1f}ms | RANSAC: {ransac_time*1000:.1f}ms | Cluster: {cluster_time*1000:.1f}ms | Filter/OBB: {filter_time*1000:.1f}ms | Total: {total_time*1000:.1f}ms ({fps:.1f} FPS)")
        
    return inlier_cloud, outlier_cloud, obbs


# =========================================================
# MODULE : DATA LOADING & ALIGNMENT
# =========================================================

def load_frames_with_timestamps(db3_path):
    """Loads a db3 file and returns a list of (timestamp_ns, pointcloud) tuples."""
    print(f"Loading {db3_path}...")
    try:
        session = load_session(db3_path)
    except Exception as e:
        print(f"Failed to load session: {e}")
        return []

    extracted = []
    for frame in session.frames:
        valid_xyz = frame.xyz_m[frame.valid_mask]
        valid_intensity = frame.intensity[frame.valid_mask]
        if len(valid_xyz) > 0:
            pc = np.hstack((valid_xyz, valid_intensity[:, np.newaxis]))
            extracted.append((frame.timestamp_ns, pc))
            
    print(f"Loaded {len(extracted)} valid frames.")
    return extracted


def process_single_bag(db3_path):
    """Loads a single db3 file and formats it for the visualizer."""
    timestamped_frames = load_frames_with_timestamps(db3_path)
    return [frame for _, frame in timestamped_frames]


def combine_two_db3_files(cfg_exec, cfg_calib):
    """Synchronizes two db3 files by time, applies calibration, and merges them."""
    path1 = cfg_exec["db_file_1"]
    path2 = cfg_exec["db_file_2"]
    max_time_diff_ns = cfg_exec["max_time_diff_ns"]
    
    frames1 = load_frames_with_timestamps(path1)
    frames2 = load_frames_with_timestamps(path2)

    if not frames1 or not frames2:
        print("Error: One or both files contained no frames.")
        return []

    print("\nSynchronizing frames by timestamp...")
    synced_pairs = []
    idx2 = 0
    
    for t1, f1 in frames1:
        best_diff = float('inf')
        best_idx2 = -1
        for i in range(idx2, len(frames2)):
            t2, f2 = frames2[i]
            diff = abs(t1 - t2)
            if diff < best_diff:
                best_diff = diff
                best_idx2 = i
            elif t2 > t1 + max_time_diff_ns:
                break
                
        if best_diff <= max_time_diff_ns and best_idx2 != -1:
            synced_pairs.append((f1, frames2[best_idx2][1]))
            idx2 = best_idx2 + 1 
            
    print(f"Found {len(synced_pairs)} synchronized frame pairs.")
    
    if not synced_pairs:
        return []

    transform_matrix = np.array(cfg_calib["transform_matrix"])

    merged_sequence = []
    for f1, f2 in synced_pairs:
        pcd_f2 = o3d.geometry.PointCloud()
        pcd_f2.points = o3d.utility.Vector3dVector(f2[:, :3])
        pcd_f2.transform(transform_matrix)
        
        aligned_xyz = np.asarray(pcd_f2.points)
        aligned_f2 = np.hstack((aligned_xyz, f2[:, 3:4]))
        merged_frame = np.vstack((f1, aligned_f2))
        merged_sequence.append(merged_frame)
        
    print(f"Successfully created {len(merged_sequence)} merged frames!")
    return merged_sequence


def create_merged_lineset(obbs):
    """Merges all bounding boxes into a single 3D LineSet."""
    if not obbs:
        merged_ls = o3d.geometry.LineSet()
        merged_ls.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]]))
        merged_ls.lines = o3d.utility.Vector2iVector(np.array([[0, 0]], dtype=np.int32))
        merged_ls.colors = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]]))
        return merged_ls

    points, lines, colors = [], [], []
    for i, obb in enumerate(obbs):
        obb_ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
        points.append(np.asarray(obb_ls.points))
        lines.append(np.asarray(obb_ls.lines) + (i * 8))
        c = np.asarray(obb.color)
        colors.append(np.tile(c, (12, 1)))
        
    merged_ls = o3d.geometry.LineSet()
    if points:
        merged_ls.points = o3d.utility.Vector3dVector(np.vstack(points))
        merged_ls.lines = o3d.utility.Vector2iVector(np.vstack(lines))
        merged_ls.colors = o3d.utility.Vector3dVector(np.vstack(colors))
    return merged_ls


# =========================================================
# MODULE : VISUALIZATION
# =========================================================

def visualize_lidar_sequence(frames, config):
    """Open3D viewer configured via YAML."""
    if not frames:
        print("No frames available for visualization.")
        return

    cfg_vis = config["visualization"]
    cfg_det = config["detection"]

    print(f"\nStarting Open3D visualization for {len(frames)} frames...")
    print("=== Controls ===")
    print(" [Space]       - Pause / Play")
    print(" [Right Arrow] - Next frame (when paused)")
    print(" [Left Arrow]  - Previous frame (when paused)")
    print(" [C]           - Toggle Mode (Intensity / Height / Distance / Clustering / Isolated Targets)")
    print("================\n")
    
    class ViewerState:
        def __init__(self):
            self.is_playing = True
            self.frame_idx = 0
            self.needs_update = True
            self.color_mode = 0 

    state = ViewerState()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name="Factory LiDAR Analytics",
        width=cfg_vis["window_width"],
        height=cfg_vis["window_height"]
    )
    
    initial_xyz = frames[0][:, :3]
    pcd = o3d.geometry.PointCloud()
    if len(initial_xyz) > 0:
        pcd.points = o3d.utility.Vector3dVector(initial_xyz)
    else:
        pcd.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]]))
        
    vis.add_geometry(pcd)
    
    bbox_lineset = o3d.geometry.LineSet()
    bbox_lineset.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]]))
    bbox_lineset.lines = o3d.utility.Vector2iVector(np.array([[0, 0]], dtype=np.int32))
    bbox_lineset.colors = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]]))
    vis.add_geometry(bbox_lineset)

    opt = vis.get_render_option()
    opt.background_color = np.asarray(cfg_vis["background_color"])
    opt.point_size = cfg_vis["point_size"]

    def toggle_play(vis_instance):
        state.is_playing = not state.is_playing
        return False
    def next_frame(vis_instance):
        if not state.is_playing and state.frame_idx < len(frames) - 1:
            state.frame_idx += 1
            state.needs_update = True
        return False
    def prev_frame(vis_instance):
        if not state.is_playing and state.frame_idx > 0:
            state.frame_idx -= 1
            state.needs_update = True
        return False
    def toggle_color(vis_instance):
        state.color_mode = (state.color_mode + 1) % 5
        modes = ["Intensity", "Height", "Distance", "Table Detection & Clustering", "Isolated Targets Only"]
        print(f"Mode changed to: {modes[state.color_mode]}")
        state.needs_update = True
        return False

    vis.register_key_callback(32, toggle_play)
    vis.register_key_callback(262, next_frame)
    vis.register_key_callback(263, prev_frame)
    vis.register_key_callback(67, toggle_color)

    while vis.poll_events():
        if state.is_playing:
            state.needs_update = True
            
        if state.needs_update:
            frame = frames[state.frame_idx]
            xyz = frame[:, :3]
            intensity = frame[:, 3]
            
            if len(xyz) > 0:
                if state.color_mode in [3, 4]:
                    inliers, outliers, obbs = detect_factory_tables(xyz, cfg_det, state.frame_idx)
                    
                    global_dist = np.linalg.norm(xyz, axis=1)
                    d_min, d_max = np.percentile(global_dist, 2), np.percentile(global_dist, 98)
                    
                    if state.color_mode == 3:
                        all_pts = np.vstack([np.asarray(inliers.points), np.asarray(outliers.points)])
                        if len(all_pts) > 0:
                            pcd.points = o3d.utility.Vector3dVector(all_pts)
                            dist = np.linalg.norm(all_pts, axis=1)
                            norm = np.clip((dist - d_min) / (d_max - d_min + 1e-6), 0, 1)
                            pcd.colors = o3d.utility.Vector3dVector(cm.jet(norm)[:, :3])
                            
                    elif state.color_mode == 4:
                        full_pcd = o3d.geometry.PointCloud()
                        full_pcd.points = o3d.utility.Vector3dVector(xyz)
                        
                        isolated_pts = []
                        for obb in obbs:
                            cropped_pcd = full_pcd.crop(obb)
                            pts = np.asarray(cropped_pcd.points)
                            if len(pts) > 0:
                                isolated_pts.append(pts)
                                
                        if isolated_pts:
                            all_isolated_pts = np.vstack(isolated_pts)
                            pcd.points = o3d.utility.Vector3dVector(all_isolated_pts)
                            dist = np.linalg.norm(all_isolated_pts, axis=1)
                            norm = np.clip((dist - d_min) / (d_max - d_min + 1e-6), 0, 1)
                            pcd.colors = o3d.utility.Vector3dVector(cm.jet(norm)[:, :3])
                        else:
                            pcd.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]]))
                            pcd.colors = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]]))

                    new_lineset = create_merged_lineset(obbs)
                    bbox_lineset.points = new_lineset.points
                    bbox_lineset.lines = new_lineset.lines
                    bbox_lineset.colors = new_lineset.colors
                    
                else:
                    pcd.points = o3d.utility.Vector3dVector(xyz)
                    bbox_lineset.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]]))
                    bbox_lineset.lines = o3d.utility.Vector2iVector(np.array([[0, 0]], dtype=np.int32))
                    bbox_lineset.colors = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]]))

                    if state.color_mode == 0:
                        if len(intensity) > 0 and np.max(intensity) > 0:
                            p98 = np.percentile(intensity, 98)
                            norm = np.clip(intensity / p98, 0, 1) if p98 > 0 else intensity
                            colors = np.column_stack((norm, norm, norm))
                            pcd.colors = o3d.utility.Vector3dVector(colors)
                        else:
                            pcd.paint_uniform_color([1.0, 1.0, 1.0])
                    elif state.color_mode == 1:
                        z = xyz[:, 2]
                        z_min, z_max = np.percentile(z, 2), np.percentile(z, 98)
                        norm = np.clip((z - z_min) / (z_max - z_min + 1e-6), 0, 1)
                        pcd.colors = o3d.utility.Vector3dVector(cm.jet(norm)[:, :3])
                    elif state.color_mode == 2:
                        dist = np.linalg.norm(xyz, axis=1)
                        d_min, d_max = np.percentile(dist, 2), np.percentile(dist, 98)
                        norm = np.clip((dist - d_min) / (d_max - d_min + 1e-6), 0, 1)
                        pcd.colors = o3d.utility.Vector3dVector(cm.jet(norm)[:, :3])
                    
            vis.update_geometry(pcd)
            vis.update_geometry(bbox_lineset)
            state.needs_update = False
            
            if state.is_playing:
                state.frame_idx += 1
                if state.frame_idx >= len(frames):
                    state.is_playing = False
                    state.frame_idx = len(frames) - 1
                    
        vis.update_renderer()
        time.sleep(cfg_vis["playback_speed"] if state.is_playing else 0.01)
            
    vis.destroy_window()


# =========================================================
# MAIN EXECUTION
# =========================================================

if __name__ == "__main__":
    config = load_config("./config.yaml")
    cfg_exec = config["execution"]
    
    mode = cfg_exec["mode"]
    print(f"=== Starting Script in {mode} Mode ===")

    frames_to_show = []

    if mode == "SINGLE_1":
        frames_to_show = process_single_bag(cfg_exec["db_file_1"])
    elif mode == "SINGLE_2":
        frames_to_show = process_single_bag(cfg_exec["db_file_2"])
    elif mode == "COMBINE":
        frames_to_show = combine_two_db3_files(cfg_exec, config["calibration"])
    else:
        print("Invalid MODE selected in config.yaml.")

    if frames_to_show:
        visualize_lidar_sequence(frames_to_show, config)