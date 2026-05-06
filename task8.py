"""
Task 8 – LiDAR Motion Compensation & Object Reconstruction

This script reconstructs a distortion-free 3D point cloud of a moving vehicle
(Passenger_Car:1) from LiDAR scans.

Problem:
- Each PCD frame is captured over ~0.1s.
- Vehicle + sensor motion introduces distortion ("motion blur").
- Each point has an implicit timestamp within the scan.
- Bounding boxes + timestamps are provided in labels.json.

Goal:
- Compensate motion using linear interpolation (translation + rotation).
- Transform all car points into a consistent global coordinate system.
- Accumulate corrected points across frames.
- Export final reconstructed point cloud.

Output:
- A single .pcd file containing the reconstructed object.
"""

import os
import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


# ============================================================
# Data Loading Utilities
# ============================================================

def load_labels(label_path: str):
    """
    Load JSON label file.

    Args:
        label_path (str): Path to labels.json

    Returns:
        list: List of label entries.
    """
    with open(label_path, "r") as f:
        data = json.load(f)
    return data["labels"]


def get_files_from_json(labels, target_id: str, pcd_folder: str):
    """
    Extract all valid PCD files associated with a given object ID.

    Args:
        labels (list): Label data from JSON.
        target_id (str): Object label ID (e.g., "Passenger_Car:1").
        pcd_folder (str): Path to PCD directory.

    Returns:
        list: Sorted list of PCD filenames.
    """
    file_list = []

    for entry in labels:
        if entry["label_id"] == target_id:
            file_name = entry["file_id"]
            if os.path.exists(os.path.join(pcd_folder, file_name)):
                file_list.append(file_name)

    return sorted(file_list)


def get_label_info(labels, file_id: str, target_id: str):
    """
    Retrieve label information and timestamp for a specific frame.

    Args:
        labels (list): Label dataset.
        file_id (str): PCD filename.
        target_id (str): Object ID.

    Returns:
        tuple: (label dict, timestamp float)
    """
    for entry in labels:
        if entry["file_id"] == file_id and entry["label_id"] == target_id:
            timestamp = float(file_id.replace(".pcd", ""))
            return entry, timestamp

    return None, None


# ============================================================
# Motion Compensation
# ============================================================

def deskew_car_points_global(
    curr_label,
    next_label,
    points,
    t_start,
    t_next,
    pos_ref,
    rot_ref
):
    """
    Compensate LiDAR motion distortion for a single frame segment
    and transform points into a consistent global frame.

    Assumes:
        - Linear translation between keyframes
        - Slerp-based rotation interpolation
        - Uniform point timing distribution within scan

    Args:
        curr_label (dict): Current frame label.
        next_label (dict): Next frame label.
        points (np.ndarray): Raw point cloud (Nx3).
        t_start (float): Start timestamp.
        t_next (float): Next frame timestamp.
        pos_ref (np.ndarray): Global reference position.
        rot_ref (Rotation): Global reference rotation.

    Returns:
        np.ndarray: Motion-compensated global points.
    """

    # --- Extract current and next bounding boxes ---
    box = curr_label["three_d_bbox"]
    pos_start = np.array([box["cx"], box["cy"], box["cz"]])
    q_start = [
        box["quaternion"]["x"],
        box["quaternion"]["y"],
        box["quaternion"]["z"],
        box["quaternion"]["w"]
    ]

    n_box = next_label["three_d_bbox"]
    pos_next = np.array([n_box["cx"], n_box["cy"], n_box["cz"]])
    q_next = [
        n_box["quaternion"]["x"],
        n_box["quaternion"]["y"],
        n_box["quaternion"]["z"],
        n_box["quaternion"]["w"]
    ]

    # --- Filter points inside current bounding box ---
    rot_matrix = R.from_quat(q_start).as_matrix()
    extent = np.array([box["l"], box["w"], box["h"]])

    obb = o3d.geometry.OrientedBoundingBox(pos_start, rot_matrix, extent)

    indices = obb.get_point_indices_within_bounding_box(
        o3d.utility.Vector3dVector(points)
    )

    car_points = points[indices]

    # Approximate timestamps within scan (0.1s window)
    car_times = t_start + np.linspace(0, 0.1, len(car_points))

    # --- Rotation interpolation ---
    slerp = Slerp([t_start, t_next], R.from_quat([q_start, q_next]))

    compensated_points = []

    for p, t_p in zip(car_points, car_times):

        # Normalized interpolation factor
        alpha = (t_p - t_start) / (t_next - t_start)

        # Interpolated pose
        interp_pos = pos_start + alpha * (pos_next - pos_start)
        interp_rot = slerp(t_p)

        # Undo motion at measurement time
        p_local = interp_rot.inv().apply(p - interp_pos)

        # Re-anchor in start frame
        p_corrected = R.from_quat(q_start).apply(p_local) + pos_start

        # Transform into global reference frame
        q_curr = [
            curr_label["three_d_bbox"]["quaternion"][k]
            for k in ["x", "y", "z", "w"]
        ]

        R_curr = R.from_quat(q_curr)
        R_rel = rot_ref * R_curr.inv()

        p_global = R_rel.apply(p_corrected - pos_start) + pos_ref

        compensated_points.append(p_global)

    return np.array(compensated_points)


# ============================================================
# Reconstruction Pipeline
# ============================================================

def accumulate_car_global(
    labels,
    file_list,
    target_id,
    output_path,
    pcd_folder
):
    """
    Reconstruct full object point cloud across all frames.

    Steps:
        1. Use first frame as global reference.
        2. Deskew each frame segment.
        3. Accumulate corrected points.
        4. Save final PCD.

    Args:
        labels (list): Dataset labels.
        file_list (list): Ordered PCD files.
        target_id (str): Object ID.
        output_path (str): Output PCD path.
        pcd_folder (str): Input folder path.
    """

    accumulated = []

    # Reference frame
    first_label, _ = get_label_info(labels, file_list[0], target_id)

    pos_ref = np.array([
        first_label["three_d_bbox"]["cx"],
        first_label["three_d_bbox"]["cy"],
        first_label["three_d_bbox"]["cz"]
    ])

    rot_ref = R.from_quat([
        first_label["three_d_bbox"]["quaternion"][k]
        for k in ["x", "y", "z", "w"]
    ])

    # Process frame pairs
    for i in range(len(file_list) - 1):

        curr_file = file_list[i]
        next_file = file_list[i + 1]

        curr_label, t_curr = get_label_info(labels, curr_file, target_id)
        next_label, t_next = get_label_info(labels, next_file, target_id)

        if curr_label is None or next_label is None:
            continue

        pcd = o3d.io.read_point_cloud(os.path.join(pcd_folder, curr_file))
        points = np.asarray(pcd.points)

        corrected = deskew_car_points_global(
            curr_label,
            next_label,
            points,
            t_curr,
            t_next,
            pos_ref,
            rot_ref
        )

        accumulated.append(corrected)

    if not accumulated:
        print("No valid points found.")
        return

    all_points = np.vstack(accumulated)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(all_points)

    o3d.io.write_point_cloud(output_path, cloud)

    visualize_result(all_points, pos_ref)


# ============================================================
# Visualization
# ============================================================

def visualize_result(all_points, pos_ref):
    """
    Visualize reconstructed point cloud with color gradient.
    """

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(all_points)

    pts = np.asarray(cloud.points)

    # Color by X-axis position
    norm = (pts[:, 0] - pts[:, 0].min()) / (pts[:, 0].ptp() + 1e-8)
    cloud.colors = o3d.utility.Vector3dVector(
        plt.get_cmap("jet")(norm)[:, :3]
    )

    vis = o3d.visualization.Visualizer()
    vis.create_window("Car Reconstruction", 1280, 720)

    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])
    opt.point_size = 2.0

    vis.add_geometry(cloud)

    vis.run()
    vis.destroy_window()


# ============================================================
# Entry Point
# ============================================================

def main():
    label_path = "labels.json"
    pcd_folder = "pcd"
    target_id = "Passenger_Car:1"
    output_file = "Passenger_Car_1_compensated.pcd"

    labels = load_labels(label_path)
    file_list = get_files_from_json(labels, target_id, pcd_folder)

    if len(file_list) < 2:
        print("Not enough frames.")
        return

    accumulate_car_global(
        labels,
        file_list,
        target_id,
        output_file,
        pcd_folder
    )


if __name__ == "__main__":
    main()