"""The offline half of the scanner: pose graph optimization + fine
re-integration, shared by the live capture path and ``--refine-only``.

``--refine-only <frames_dir>`` rebuilds a point cloud from a previous
session's frames directory (``meta.json`` + ``poses.npy`` + frames) without
touching the camera — the recovery path after a crash, or a way to re-run
refinement with different parameters.
"""

import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.core as o3c

from .frameio import load_meta
from .mapping import refine, save_cloud
from .posegraph import optimize_trajectory


def offline_pipeline(frames_dir, poses, intrinsic, depth_scale, args, device):
    """Optimize the trajectory (unless ``--no-posegraph``), save it, re-fuse
    all frames at the fine voxel size, and save the point cloud.

    Returns:
        Number of points saved.
    """
    if not args.no_posegraph and len(poses) > 1:
        try:
            poses = optimize_trajectory(frames_dir, poses, intrinsic,
                                        depth_scale, args, device)
        except Exception as e:
            print(f"[warn] pose graph optimization failed, refining with "
                  f"tracked poses: {e}")
    np.save(Path(args.output).with_suffix(".trajectory.npy"), np.stack(poses))
    vbg = refine(frames_dir, poses, intrinsic, depth_scale, args, device)
    return save_cloud(vbg, args.output, mesh=args.mesh, clean=True)


def view_result(path):
    """Open the saved point cloud in the Open3D viewer."""
    pcd = o3d.io.read_point_cloud(str(path))
    o3d.visualization.draw_geometries([pcd],
                                      window_name="Reconstruction result")


def refine_only(args, device):
    """Entry point for ``--refine-only``: run the offline pipeline on a
    previously recorded frames directory."""
    frames_dir = Path(args.refine_only)
    meta_path = frames_dir / "meta.json"
    poses_path = frames_dir / "poses.npy"
    if not meta_path.exists() or not poses_path.exists():
        sys.exit(f"[error] {frames_dir} needs meta.json and poses.npy "
                 f"(recorded by a capture session)")
    meta = load_meta(meta_path)
    intrinsic = o3c.Tensor([[meta["fx"], 0.0, meta["ppx"]],
                            [0.0, meta["fy"], meta["ppy"]],
                            [0.0, 0.0, 1.0]], o3c.Dtype.Float64)
    depth_scale = meta["depth_scale"]
    args.depth_max = meta["depth_max"]
    poses = list(np.load(poses_path))
    # a crash can leave more poses than flushed frames; refine skips gaps,
    # but trim the tail so progress reporting is honest
    n_files = len(list(frames_dir.glob("depth_*.png")))
    poses = poses[:n_files]
    print(f"[info] refine-only: {len(poses)} frames from {frames_dir}")
    n = offline_pipeline(frames_dir, poses, intrinsic, depth_scale, args, device)
    if n and not args.no_view:
        view_result(args.output)
