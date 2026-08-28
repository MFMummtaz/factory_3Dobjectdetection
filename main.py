import time
import yaml
import numpy as np
import open3d as o3d
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

from mlx_db3 import load_session


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# =========================================================
# MODULE: OBJECT DETECTION
# =========================================================

def detect_factory_tables(xyz_points, cfg_det):
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
    
    # 3. Clustering
    t2 = time.perf_counter()
    p1_cfg = cfg_det["pass1_dbscan"]
    labels = np.array(outlier_cloud.cluster_dbscan(
        eps=p1_cfg["eps"],
        min_points=p1_cfg["min_points"],
        print_progress=False
    ))
    max_label = labels.max()
    cluster_time = time.perf_counter() - t2
    
    # 4. Box Generation & Distance Measurements
    t3 = time.perf_counter()
    obbs = []
    distances = []
    
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
                        
                        shortest_axis_idx = np.argmin(obb.extent)
                        shortest_axis_dir = obb.R[:, shortest_axis_idx]
                        
                        if abs(shortest_axis_dir[2]) > 0.1:
                            obb.color = np.array([1.0, 0.0, 0.0])
                            obbs.append(obb)

                            if shortest_axis_dir[2] < 0: 
                                shortest_axis_dir = -shortest_axis_dir

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
                                drop_green_dist = h_table_mean - red_bottom_h
                                table_base_pts = table_pts - (drop_green_dist * shortest_axis_dir)
                                green_vol_pts = np.vstack((table_pts, table_base_pts))
                                
                                t_cloud = o3d.geometry.PointCloud()
                                t_cloud.points = o3d.utility.Vector3dVector(green_vol_pts)
                                t_obb = t_cloud.get_oriented_bounding_box()
                                t_obb.color = np.array([0.0, 1.0, 0.0])
                                obbs.append(t_obb)
                                
                                table_max_h = np.max(np.dot(table_pts - obb.center, shortest_axis_dir))
                                drop_cyan_dist = h_plate_mean - table_max_h
                                plate_base_pts = plate_pts - (drop_cyan_dist * shortest_axis_dir)
                                cyan_vol_pts = np.vstack((plate_pts, plate_base_pts))
                                
                                p_cloud = o3d.geometry.PointCloud()
                                p_cloud.points = o3d.utility.Vector3dVector(cyan_vol_pts)
                                p_obb = p_cloud.get_oriented_bounding_box()
                                p_obb.color = np.array([0.0, 0.8, 1.0])
                                obbs.append(p_obb)
                                
                                distance_m = h_plate_mean - table_max_h
                                if distance_m > 0:
                                    distances.append(distance_m)

    else:
        outlier_cloud.paint_uniform_color([0, 0, 0])
        
    filter_time = time.perf_counter() - t3
    total_time = time.perf_counter() - start_time
    fps = 1.0 / total_time if total_time > 0 else 0
    
    metrics = {
        "downsample_ms": downsample_time * 1000,
        "ransac_ms": ransac_time * 1000,
        "cluster_ms": cluster_time * 1000,
        "filter_ms": filter_time * 1000,
        "total_ms": total_time * 1000,
        "fps": fps,
        "distances": distances
    }
        
    return inlier_cloud, outlier_cloud, obbs, metrics


# =========================================================
# MODULE: DATA LOADING & ALIGNMENT
# =========================================================

def load_frames_with_timestamps(db3_path):
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
    return extracted

def process_single_bag(db3_path):
    timestamped_frames = load_frames_with_timestamps(db3_path)
    return [frame for _, frame in timestamped_frames]

def combine_two_db3_files(cfg_exec, cfg_calib):
    path1 = cfg_exec["db_file_1"]
    path2 = cfg_exec["db_file_2"]
    max_time_diff_ns = cfg_exec["max_time_diff_ns"]
    
    frames1 = load_frames_with_timestamps(path1)
    frames2 = load_frames_with_timestamps(path2)
    if not frames1 or not frames2: return []

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
    return merged_sequence

def create_merged_lineset(obbs):
    merged_ls = o3d.geometry.LineSet()
    if not obbs:
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
        
    merged_ls.points = o3d.utility.Vector3dVector(np.vstack(points))
    merged_ls.lines = o3d.utility.Vector2iVector(np.vstack(lines))
    merged_ls.colors = o3d.utility.Vector3dVector(np.vstack(colors))
    return merged_ls


# =========================================================
# MODULE: MODERN TKINTER DASHBOARD
# =========================================================

class ModernLidarDashboard:
    def __init__(self, root, total_frames, modes, state):
        self.root = root
        self.state = state
        self.total_frames = total_frames
        
        # Window setup
        self.root.title("LiDAR Control & Telemetry")
        self.root.geometry("380x620")
        self.root.configure(bg="#1e1e1e")
        self.root.attributes('-topmost', True)
        
        # Style setup
        self.style = ttk.Style()
        self.style.theme_use("classic")
        self.configure_dark_theme()

        # Layout Containers
        self.build_playback_card()
        self.build_mode_card(modes)
        self.build_perf_card()
        self.build_telemetry_card()

    def configure_dark_theme(self):
        BG_COLOR = "#1e1e1e"
        CARD_BG = "#252526"
        TEXT_COLOR = "#d4d4d4"
        ACCENT_COLOR = "#007acc"

        self.style.configure(".", background=BG_COLOR, foreground=TEXT_COLOR, font=("Segoe UI", 9))
        self.style.configure("Card.TFrame", background=CARD_BG, relief="flat")
        self.style.configure("Header.TLabel", font=("Segoe UI", 10, "bold"), foreground="#4ec9b0", background=CARD_BG)
        self.style.configure("Value.TLabel", font=("Consolas", 10, "bold"), foreground="#ce9178", background=CARD_BG)
        self.style.configure("FPS.TLabel", font=("Consolas", 16, "bold"), foreground="#569cd6", background=CARD_BG)
        
        # Buttons
        self.style.configure("TButton", background="#333333", foreground="#ffffff", borderwidth=0, font=("Segoe UI", 9, "bold"))
        self.style.map("TButton", background=[("active", ACCENT_COLOR)])
        
        # Combobox
        self.style.configure("TCombobox", fieldbackground="#333333", background="#333333", foreground="#ffffff", arrowcolor="#ffffff")

    def create_card(self, title):
        card = ttk.Frame(self.root, style="Card.TFrame", padding=12)
        card.pack(fill="x", padx=12, pady=6)
        lbl = ttk.Label(card, text=title, style="Header.TLabel")
        lbl.pack(anchor="w", pady=(0, 6))
        return card

    def build_playback_card(self):
        card = self.create_card("PLAYBACK CONTROLS")
        
        # Frame counter label
        self.lbl_frame = ttk.Label(card, text=f"Frame: 0 / {self.total_frames - 1}", background="#252526")
        self.lbl_frame.pack(anchor="w", pady=(0, 4))
        
        # Frame Scrubber Slider
        self.slider_var = tk.DoubleVar(value=0)
        self.slider = ttk.Scale(
            card, from_=0, to=max(self.total_frames - 1, 1), 
            variable=self.slider_var, command=self.on_slider_drag
        )
        self.slider.pack(fill="x", pady=(0, 8))

        # Control Buttons
        btn_box = ttk.Frame(card, style="Card.TFrame")
        btn_box.pack(fill="x")
        
        self.btn_prev = ttk.Button(btn_box, text="◄ Prev", width=8, command=self.on_prev)
        self.btn_prev.pack(side="left", padx=(0, 4))

        self.btn_play = ttk.Button(btn_box, text="Pause", width=10, command=self.on_toggle_play)
        self.btn_play.pack(side="left", fill="x", expand=True, padx=2)

        self.btn_next = ttk.Button(btn_box, text="Next ►", width=8, command=self.on_next)
        self.btn_next.pack(side="left", padx=(4, 0))

    def build_mode_card(self, modes):
        card = self.create_card("VISUALIZATION MODE")
        self.combo_mode = ttk.Combobox(card, values=modes, state="readonly")
        self.combo_mode.current(0)
        self.combo_mode.pack(fill="x")
        self.combo_mode.bind("<<ComboboxSelected>>", self.on_mode_change)

    def build_perf_card(self):
        card = self.create_card("PERFORMANCE METRICS")
        
        # FPS Display
        fps_frame = ttk.Frame(card, style="Card.TFrame")
        fps_frame.pack(fill="x", pady=(0, 6))
        ttk.Label(fps_frame, text="Real-time Inference:", background="#252526").pack(side="left")
        self.lbl_fps = ttk.Label(fps_frame, text="0.0 FPS", style="FPS.TLabel")
        self.lbl_fps.pack(side="right")

        # Grid breakdown
        grid = ttk.Frame(card, style="Card.TFrame")
        grid.pack(fill="x")
        
        self.perf_labels = {}
        stages = [
            ("Downsample:", "ds"), ("Ground RANSAC:", "ransac"),
            ("Clustering:", "cluster"), ("Filtering/Box:", "filter"),
            ("Total Frame Latency:", "total")
        ]
        
        for i, (label_text, key) in enumerate(stages):
            ttk.Label(grid, text=label_text, background="#252526").grid(row=i, column=0, sticky="w", pady=2)
            lbl_val = ttk.Label(grid, text="- ms", style="Value.TLabel")
            lbl_val.grid(row=i, column=1, sticky="e", pady=2)
            grid.columnconfigure(1, weight=1)
            self.perf_labels[key] = lbl_val

    def build_telemetry_card(self):
        card = self.create_card("TARGET MEASUREMENTS")
        self.lbl_telemetry = ttk.Label(card, text="No targets detected.", background="#252526", foreground="#9cdcfe", font=("Consolas", 9))
        self.lbl_telemetry.pack(anchor="w")

    # --- UI Callbacks ---
    def on_toggle_play(self):
        self.state.is_playing = not self.state.is_playing
        self.btn_play.config(text="Pause" if self.state.is_playing else "Play")

    def on_prev(self):
        if not self.state.is_playing and self.state.frame_idx > 0:
            self.state.frame_idx -= 1
            self.state.needs_update = True
            self.slider_var.set(self.state.frame_idx)

    def on_next(self):
        if not self.state.is_playing and self.state.frame_idx < self.total_frames - 1:
            self.state.frame_idx += 1
            self.state.needs_update = True
            self.slider_var.set(self.state.frame_idx)

    def on_slider_drag(self, val):
        idx = int(float(val))
        if idx != self.state.frame_idx:
            self.state.frame_idx = idx
            self.state.needs_update = True

    def on_mode_change(self, event):
        self.state.color_mode = self.combo_mode.current()
        self.state.needs_update = True

    # --- External Data Update ---
    def update_dashboard(self, frame_idx, metrics=None):
        self.lbl_frame.config(text=f"Frame: {frame_idx} / {self.total_frames - 1}")
        if self.state.is_playing:
            self.slider_var.set(frame_idx)

        if metrics:
            self.lbl_fps.config(text=f"{metrics['fps']:.1f} FPS")
            self.perf_labels["ds"].config(text=f"{metrics['downsample_ms']:.1f} ms")
            self.perf_labels["ransac"].config(text=f"{metrics['ransac_ms']:.1f} ms")
            self.perf_labels["cluster"].config(text=f"{metrics['cluster_ms']:.1f} ms")
            self.perf_labels["filter"].config(text=f"{metrics['filter_ms']:.1f} ms")
            self.perf_labels["total"].config(text=f"{metrics['total_ms']:.1f} ms")
            
            if metrics["distances"]:
                lines = [f"► Steel Plate {i+1}: {d*100.0:.2f} cm gap" for i, d in enumerate(metrics["distances"])]
                self.lbl_telemetry.config(text="\n".join(lines))
            else:
                self.lbl_telemetry.config(text="No Steel Plate detected.")
        else:
            self.lbl_fps.config(text="N/A")
            for lbl in self.perf_labels.values():
                lbl.config(text="-")
            self.lbl_telemetry.config(text="Detection Mode Off")


# =========================================================
# MODULE: VISUALIZATION & DUAL LOOP
# =========================================================

def visualize_lidar_sequence(frames, config):
    if not frames:
        print("No frames available for visualization.")
        return

    cfg_vis = config["visualization"]
    cfg_det = config["detection"]

    modes = ["Intensity", "Height (Z)", "Distance", "Table Detection & DBSCAN", "Isolated Targets Only"]

    class State:
        def __init__(self):
            self.is_playing = True
            self.frame_idx = 0
            self.needs_update = True
            self.color_mode = 0

    state = State()

    # 1. Initialize Modern Tkinter Window
    root = tk.Tk()
    dashboard = ModernLidarDashboard(root, len(frames), modes, state)

    # 2. Initialize Standard Open3D Visualizer Window
    vis = o3d.visualization.Visualizer()
    vis.create_window("Factory LiDAR Viewport", width=cfg_vis["window_width"], height=cfg_vis["window_height"])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(frames[0][:, :3])
    vis.add_geometry(pcd)
    
    bbox_lineset = o3d.geometry.LineSet()
    bbox_lineset.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]]))
    bbox_lineset.lines = o3d.utility.Vector2iVector(np.array([[0, 0]], dtype=np.int32))
    vis.add_geometry(bbox_lineset)

    opt = vis.get_render_option()
    opt.background_color = np.asarray(cfg_vis["background_color"])
    opt.point_size = cfg_vis["point_size"]

    # 3. Unified Synchronous Event Loop
    while vis.poll_events():
        try:
            root.update()
        except tk.TclError:
            break # User closed the Tkinter UI window

        if state.is_playing:
            state.needs_update = True
            
        if state.needs_update:
            frame = frames[state.frame_idx]
            xyz = frame[:, :3]
            intensity = frame[:, 3]
            
            if len(xyz) > 0:
                if state.color_mode in [3, 4]:
                    inliers, outliers, obbs, metrics = detect_factory_tables(xyz, cfg_det)
                    dashboard.update_dashboard(state.frame_idx, metrics)
                    
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
                            cropped = full_pcd.crop(obb)
                            pts = np.asarray(cropped.points)
                            if len(pts) > 0: isolated_pts.append(pts)
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
                    dashboard.update_dashboard(state.frame_idx, metrics=None)
                    
                    pcd.points = o3d.utility.Vector3dVector(xyz)
                    bbox_lineset.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]]))
                    bbox_lineset.lines = o3d.utility.Vector2iVector(np.array([[0, 0]], dtype=np.int32))
                    bbox_lineset.colors = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]]))

                    if state.color_mode == 0:
                        if len(intensity) > 0 and np.max(intensity) > 0:
                            p98 = np.percentile(intensity, 98)
                            norm = np.clip(intensity / p98, 0, 1) if p98 > 0 else intensity
                            pcd.colors = o3d.utility.Vector3dVector(np.column_stack((norm, norm, norm)))
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
                    dashboard.btn_play.config(text="Play")
                    state.frame_idx = len(frames) - 1
                    
        vis.update_renderer()
        time.sleep(cfg_vis["playback_speed"] if state.is_playing else 0.01)
            
    vis.destroy_window()
    try:
        root.destroy()
    except tk.TclError:
        pass


# =========================================================
# MAIN EXECUTION
# =========================================================

if __name__ == "__main__":
    config = load_config("./config.yaml")
    cfg_exec = config["execution"]
    
    print(f"=== Starting Script in {cfg_exec['mode']} Mode ===")

    frames_to_show = []
    if cfg_exec["mode"] == "SINGLE_1":
        frames_to_show = process_single_bag(cfg_exec["db_file_1"])
    elif cfg_exec["mode"] == "SINGLE_2":
        frames_to_show = process_single_bag(cfg_exec["db_file_2"])
    elif cfg_exec["mode"] == "COMBINE":
        frames_to_show = combine_two_db3_files(cfg_exec, config["calibration"])
    else:
        print("Invalid MODE selected in config.yaml.")

    if frames_to_show:
        visualize_lidar_sequence(frames_to_show, config)