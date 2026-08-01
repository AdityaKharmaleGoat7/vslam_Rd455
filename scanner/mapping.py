"""TSDF volume handling: point-cloud extraction/export and the fine
re-integration pass.

Export goes through Open3D's *legacy* writer on purpose: it stores colors
as uchar red/green/blue, which every external viewer (MeshLab,
CloudCompare, web viewers) understands. The tensor writer emits float
colors that those viewers render as no color at all.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c


def extract_cloud(vbg, weight_threshold=3.0):
    """Extract a point cloud from a voxel block grid onto the CPU."""
    pcd = vbg.extract_point_cloud(weight_threshold)
    return pcd.to(o3c.Device("CPU:0"))


def save_cloud(vbg, path, mesh=False, clean=False):
    """Extract, optionally de-noise, and save with 8-bit RGB (uchar PLY).

    Args:
        vbg: the TSDF ``VoxelBlockGrid`` (or ``Model.voxel_grid``).
        path: output ``.ply`` path.
        mesh: also export a triangle mesh next to the point cloud.
        clean: run statistical outlier removal before saving.

    Returns:
        Number of points saved (0 if the volume was empty).
    """
    pcd = extract_cloud(vbg).to_legacy()
    if clean and len(pcd.points) > 0:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
    n = len(pcd.points)
    if n == 0:
        print("[warn] point cloud is empty, nothing saved")
        return 0
    o3d.io.write_point_cloud(str(path), pcd)
    print(f"[saved] {path}  ({n:,} points, 8-bit RGB)")
    if mesh:
        mesh_path = Path(path).with_suffix(".mesh.ply")
        tri = vbg.extract_triangle_mesh(3.0).to(o3c.Device("CPU:0")).to_legacy()
        o3d.io.write_triangle_mesh(str(mesh_path), tri)
        print(f"[saved] {mesh_path}")
    return n


def refine(frames_dir, poses, intrinsic, depth_scale, args, device):
    """Re-fuse all recorded frames at the finer ``--refine-voxel`` size.

    Frames are streamed from disk one at a time; the grid uses uint16
    weight/color attributes to keep the finer volume within low-RAM
    budgets. Missing frames (files not yet flushed at crash time) are
    skipped.
    """
    n = len(poses)
    print(f"[refine] re-integrating {n} frames at "
          f"{args.refine_voxel * 1000:.0f} mm voxels ...")
    vbg = o3d.t.geometry.VoxelBlockGrid(
        ("tsdf", "weight", "color"),
        (o3c.float32, o3c.uint16, o3c.uint16),
        ((1,), (1,), (3,)),
        args.refine_voxel, 16, args.block_count * 2, device)
    t0 = time.time()
    for i in range(n):
        depth_np = cv2.imread(str(frames_dir / f"depth_{i:05d}.png"),
                              cv2.IMREAD_UNCHANGED)
        color_bgr = cv2.imread(str(frames_dir / f"color_{i:05d}.jpg"))
        if depth_np is None or color_bgr is None:
            continue
        depth = o3d.t.geometry.Image(depth_np.astype(np.uint16)).to(device)
        color = o3d.t.geometry.Image(
            np.ascontiguousarray(color_bgr[:, :, ::-1])).to(device)
        extrinsic = o3c.Tensor(np.linalg.inv(poses[i]), o3c.Dtype.Float64)
        blocks = vbg.compute_unique_block_coordinates(
            depth, intrinsic, extrinsic, depth_scale, args.depth_max)
        vbg.integrate(blocks, depth, color, intrinsic, intrinsic,
                      extrinsic, depth_scale, args.depth_max,
                      args.trunc_multiplier)
        if (i + 1) % 25 == 0 or i == n - 1:
            print(f"[refine] {i + 1}/{n}  ({(i + 1) / (time.time() - t0):.1f} fps)")
    return vbg
