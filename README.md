![Demo](https://private-user-images.githubusercontent.com/87586726/640726664-9717cf6b-c490-404e-8526-35354e9df3cd.gif?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODc2NDEwMTQsIm5iZiI6MTc4NzY0MDcxNCwicGF0aCI6Ii84NzU4NjcyNi82NDA3MjY2NjQtOTcxN2NmNmItYzQ5MC00MDRlLTg1MjYtMzUzNTRlOWRmM2NkLmdpZj9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA4MjUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwODI1VDA2NTE1NFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTZhOGVkNTA3ODIwMTI4MDc1YzQyNmIzYTQ4NzA5NTkyZTJhZTg5OGJmMDIyN2M2YTM2YmY2YWFkMjJkZTI0MGEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRmdpZiJ9.UeOOJ8uzskikQBqTy4ebv89dlUcWkwS2bezp1dCA3c4)

# Factory LiDAR Analytics

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![Domain](https://img.shields.io/badge/domain-3D%20Point%20Cloud%20Processing-orange)
![Engine](https://img.shields.io/badge/viewer-Open3D-green)

A modular 3D point cloud processing pipeline built for dual-3D LiDAR industrial monitoring systems. This repository handles multi-sensor temporal synchronization, spatial coordinate alignment, hierarchical object segmentation (table assemblies and steel plates), and volumetric bounding box extraction to deliver accurate real-time vertical height measurements.

## 📋 Table of Contents
- [Overview](#overview)
- [Repository Structure](#-repository-structure)
- [Prerequisites & Installation](#-prerequisites--installation)
- [Configuration Guide (`config.yaml`)](#-configuration-guide-configyaml)
- [Pipeline Architecture (`main.py`)](#-pipeline-architecture-mainpy)
- [Interactive Viewer Controls](#-interactive-viewer-controls)
- [Usage Instructions](#-usage-instructions)

---

## 🌐 Overview

In industrial manufacturing, monitoring raw materials and equipment positioning requires millimetric precision across multiple sensors. **Factory LiDAR Analytics** addresses key operational challenges:
* **Sensor Fusion & Alignment:** Combines multi-sensor ROS `.db3` recordings into a unified coordinate frame using $4 \times 4$ rigid transformation matrices $[R \mid T]$.
* **Temporal Synchronization:** Filters frames using nanosecond-level timestamp matching (`max_time_diff_ns`).
* **Hierarchical Detection:** Isolates complex objects (e.g., table tops, steel plates) through ground removal, DBSCAN clustering, dimensional filtering, and surface RANSAC.
* **3D Visual Analytics:** Interactive Open3D visualizer supporting live 3D bounding box overlays, height extraction, and multi-mode colormaps.

---

## 📁 Repository Structure

```text
.
├── config.yaml         # Centralized YAML configuration for hardware, calibration, and detection settings
├── main.py             # Application entry point (loaders, spatial registration, segmentation, viewer)
├── requirements.txt    # Python library requirements
└── README.md           # Project documentation
```

---

## ⚡ Prerequisites & Installation

### System Requirements
* **Python**: 3.10 or higher (tested in Ubuntu 22.04)
* **Dependencies**: Open3D, NumPy, PyYAML, and relevant `.db3` recording parsers.

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MFMummtaz/factory-lidar-analytics.git
   cd factory-lidar-analytics
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate 
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Configuration Guide (`config.yaml`)

All parameters governing data streams, rigid registration, algorithm tolerances, and display preferences are configured centrally in `config.yaml`.

| Section | Parameter | Type | Description |
| :--- | :--- | :--- | :--- |
| **`execution`** | `mode` | `String` | Operational mode: `SINGLE_1` (Sensor 1), `SINGLE_2` (Sensor 2), or `COMBINE` (Dual synchronized) |
| | `db_file_1` | `Path` | File path to primary sensor raw `.db3` dataset |
| | `db_file_2` | `Path` | File path to secondary sensor raw `.db3` dataset |
| | `max_time_diff_ns` | `Integer` | Max allowable timestamp difference ($50,000,000\text{ ns} = 50\text{ ms}$) for dual stream sync |
| **`calibration`** | `transform_matrix` | `List[4x4]` | $4 \times 4$ rigid matrix $[R \mid T]$ mapping Sensor 2 frame into Sensor 1 reference frame |
| **`detection`** | `voxel_size` | `Float` | Grid cell dimension (meters) for initial point downsampling |
| | `ground_ransac` | `Object` | RANSAC distance threshold, sample count, and max iterations for floor removal |
| | `pass1_dbscan` | `Object` | Clustering distance (`eps`), min point neighbors, and cluster size bounds for initial assembly isolation |
| | `table_dimensions` | `List` | Min/Max bounding dimensions `[width, length, height]` (meters) to reject invalid candidate clusters |
| | `pass2_ransac` | `Object` | Fine-grained RANSAC parameters to separate table top and steel plate surfaces |
| **`visualization`**| `window_width` / `height` | `Integer` | Open3D render window geometry |
| | `point_size` | `Float` | Rendered point size in pixels |
| | `playback_speed` | `Float` | Automated playback frame delay/rate |
| | `background_color` | `List[3]` | RGB background color vector |

---

## 🏗️ Pipeline Architecture (`main.py`)

The pipeline processes continuous frame sequences through four modular phases:

```text
[ Raw DB3 Streams ] ---> [ Sync & 4x4 Registration ] ---> [ Ground RANSAC ]
                                                                 │
                                                                 ▼
[ Interactive Open3D Visualizer ] <--- [ Bounding Boxes & Height ] <--- [ Multi-Pass Surface RANSAC ]
```

1. **Configuration Ingestion:** Reads operational, hardware, and algorithmic parameters from `config.yaml`.
2. **Data Loading & Spatial Registration:**
   * Ingests XYZ coordinates and reflectivity intensity values from `.db3` files.
   * Matches timestamp pairs within `max_time_diff_ns`.
   * Multiplies Sensor 2 point vectors by the $4 \times 4$ rigid transformation matrix $[R \mid T]$.
3. **Hierarchical Object Detection:**
   * **Ground Plane Removal:** Fits a plane using RANSAC to exclude floor points.
   * **Pass 1 (DBSCAN Assembly Clustering):** Identifies target candidate clusters (`eps`, `min_points`).
   * **Dimensional Gate:** Compares cluster oriented bounding boxes against `table_dimensions`.
   * **Pass 2 (Surface Extraction RANSAC):** Differentiates top table plane from stacked steel plate surfaces.
   * **Volumetric Extrusion:** Calculates exact height offset and fits oriented bounding boxes (Green: Table Base, Cyan: Steel Plate).
4. **Visualization Engine:** Streams processed frames through Open3D, constructing custom bounding box `LineSet` objects and printing real-time height measurements to standard output.

---

## 🎮 Interactive Viewer Controls

### Keyboard Bindings

| Key | Action | Description |
| :---: | :--- | :--- |
| `Space` | Play / Pause | Toggle continuous automated sequence playback |
| `Right Arrow` | Next Frame | Step forward by 1 frame (when paused) |
| `Left Arrow` | Previous Frame | Step backward by 1 frame (when paused) |
| `C` | Cycle Color Modes | Switch dynamically through visual colormaps (Modes 0–4) |

### Render Display Modes (`C` Key)

* **Mode 0 — Reflectivity Intensity Map:** Greyscale gradient representing returned LiDAR beam signal intensity.
* **Mode 1 — Z-Height Map:** Jet colormap encoding absolute vertical height along the Z-axis.
* **Mode 2 — Radial Distance Map:** Jet colormap encoding Euclidean distance from the sensor origin.
* **Mode 3 — Full Pipeline Output:** Displays full point cloud with active bounding boxes (Red: Candidate, Green: Table Base, Cyan: Steel Plate).
* **Mode 4 — Target Isolation Mode:** Crops point cloud to isolate target objects and measurements exclusively.

---

## 🚀 Usage Instructions

1. Activate the python environment
2. **Configure Parameters:** Open `config.yaml` and set paths for your raw `.db3` files along with the desired mode (`SINGLE_1`, `SINGLE_2`, or `COMBINE`) also choose the correct calibration parameters according to the choosen .db3 files.
3. **Execute Processing Script:**
   ```bash
   python main.py
   ```
4. **Inspect Results:**
   * Press `Space` to pause/resume frame streaming.
   * Press `C` to switch to **Mode 3** or **Mode 4** to inspect bounding box fits and live metric logging in the console output.
