"""Offline trajectory optimization.

Frame-to-model tracking drifts; re-fusing frames with drifted poses bakes
the drift into the final cloud. This module builds a keyframe pose graph —
odometry edges between consecutive keyframes plus loop-closure edges
between keyframe pairs that are spatially close but temporally distant —
optimizes it with Levenberg-Marquardt, and spreads the keyframe
corrections onto every in-between frame by interpolation (slerp on
rotation, lerp on translation).
"""

import numpy as np
import open3d as o3d
import open3d.core as o3c

from .config import (FITNESS_MIN, LOOP_FITNESS_MIN, LOOP_MAX_ATTEMPTS,
                     LOOP_MIN_GAP_FRAMES, LOOP_RADIUS_M, RMSE_MAX)
from .frameio import load_rgbd
from .tracking import rgbd_odometry
from .transforms import interp_transform, rotation_angle_deg


def select_keyframes(poses, every, trans_thr, rot_thr_deg):
    """Pick keyframe indices: every ``every`` frames, or sooner when the
    relative motion since the last keyframe exceeds the translation (m) or
    rotation (deg) threshold. Always includes the first and last frame."""
    kf = [0]
    for i in range(1, len(poses)):
        rel = np.linalg.inv(poses[kf[-1]]) @ poses[i]
        if (i - kf[-1] >= every
                or np.linalg.norm(rel[:3, 3]) > trans_thr
                or rotation_angle_deg(rel[:3, :3]) > rot_thr_deg):
            kf.append(i)
    if kf[-1] != len(poses) - 1:
        kf.append(len(poses) - 1)
    return kf


def optimize_trajectory(frames_dir, poses, intrinsic, depth_scale, args, device):
    """Optimize the tracked trajectory against recorded frames.

    Args:
        frames_dir: directory of recorded ``depth_*.png``/``color_*.jpg``.
        poses: list of (4, 4) tracked camera-to-world poses, one per frame.
        intrinsic: 3x3 ``o3c.Tensor`` camera matrix.
        depth_scale: raw depth units per meter.
        args: parsed CLI namespace (keyframe thresholds, depth_max).
        device: Open3D device for odometry.

    Returns:
        A new list of corrected camera-to-world poses (same length).
    """
    reg = o3d.pipelines.registration
    poses = [np.asarray(p) for p in poses]
    kf = select_keyframes(poses, args.kf_every, args.kf_trans, args.kf_rot)
    print(f"[posegraph] {len(kf)} keyframes from {len(poses)} frames")

    cache = {}  # small LRU of loaded keyframe RGBDs (low-RAM)

    def rgbd(i):
        if i not in cache:
            if len(cache) >= 8:
                cache.pop(next(iter(cache)))
            cache[i] = load_rgbd(frames_dir, i, device)
        return cache[i]

    def info_matrix(src, tgt, trans):
        try:
            return o3d.t.pipelines.odometry.compute_odometry_information_matrix(
                src.depth, tgt.depth, intrinsic,
                o3c.Tensor(trans, o3c.Dtype.Float64),
                0.07, depth_scale, args.depth_max).numpy()
        except Exception:
            return np.eye(6)

    pg = reg.PoseGraph()
    for k in kf:
        pg.nodes.append(reg.PoseGraphNode(poses[k]))

    # odometry edges between consecutive keyframes, refined by RGBD odometry
    for n in range(len(kf) - 1):
        a, b = kf[n], kf[n + 1]
        T_ab = np.linalg.inv(poses[b]) @ poses[a]  # maps frame-a points into frame b
        sa, tb = rgbd(a), rgbd(b)
        info = np.eye(6)
        if sa is not None and tb is not None:
            try:
                res = rgbd_odometry(sa, tb, intrinsic, T_ab,
                                    depth_scale, args.depth_max)
                if res.fitness >= FITNESS_MIN:
                    T_ab = res.transformation.numpy()
                info = info_matrix(sa, tb, T_ab)
            except RuntimeError:
                pass
        pg.edges.append(reg.PoseGraphEdge(n, n + 1, T_ab, info,
                                          uncertain=False))

    # loop closures: spatially close, temporally distant keyframe pairs
    candidates = []
    for i in range(len(kf)):
        for j in range(i + 1, len(kf)):
            if kf[j] - kf[i] < LOOP_MIN_GAP_FRAMES:
                continue
            d = np.linalg.norm(poses[kf[i]][:3, 3] - poses[kf[j]][:3, 3])
            if d < LOOP_RADIUS_M:
                candidates.append((d, i, j))
    candidates.sort()
    loops = 0
    for _, i, j in candidates[:LOOP_MAX_ATTEMPTS]:
        a, b = kf[i], kf[j]
        sa, tb = rgbd(a), rgbd(b)
        if sa is None or tb is None:
            continue
        init = np.linalg.inv(poses[b]) @ poses[a]
        try:
            res = rgbd_odometry(sa, tb, intrinsic, init,
                                depth_scale, args.depth_max)
        except RuntimeError:
            continue
        if res.fitness < LOOP_FITNESS_MIN or res.inlier_rmse > RMSE_MAX:
            continue
        T_ab = res.transformation.numpy()
        pg.edges.append(reg.PoseGraphEdge(i, j, T_ab,
                                          info_matrix(sa, tb, T_ab),
                                          uncertain=True))
        loops += 1
    print(f"[posegraph] {loops} loop closure edges "
          f"({len(candidates)} candidate pairs)")

    option = reg.GlobalOptimizationOption(
        max_correspondence_distance=0.03,
        edge_prune_threshold=0.25,
        preference_loop_closure=2.0,
        reference_node=0)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        reg.global_optimization(
            pg, reg.GlobalOptimizationLevenbergMarquardt(),
            reg.GlobalOptimizationConvergenceCriteria(), option)

    # propagate keyframe corrections to in-between frames
    corrections = [pg.nodes[n].pose @ np.linalg.inv(poses[kf[n]])
                   for n in range(len(kf))]
    new_poses = list(poses)
    for n in range(len(kf) - 1):
        a, b = kf[n], kf[n + 1]
        for f in range(a, b + 1):
            t = (f - a) / max(b - a, 1)
            C = interp_transform(corrections[n], corrections[n + 1], t)
            new_poses[f] = C @ poses[f]
    drift = np.linalg.norm(corrections[-1][:3, 3])
    print(f"[posegraph] optimized; final-keyframe correction {drift * 100:.1f} cm")
    return new_poses
