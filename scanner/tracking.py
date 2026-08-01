"""RGBD odometry helpers shared by live tracking, relocalization, and the
pose graph.

``Model.track_frame_to_model`` (Open3D 0.19) does not accept an initial
transformation, so every code path that needs an init — the IMU rotation
prior, relocalization, loop-closure verification — goes through
:func:`rgbd_odometry`, a thin wrapper over
``o3d.t.pipelines.odometry.rgbd_odometry_multi_scale``.
"""

import open3d as o3d
import open3d.core as o3c

_CRITERIA = None


def odo_criteria():
    """Multi-scale convergence criteria matching the realtime tracker's
    defaults (6/3/1 iterations coarse-to-fine, cheaper than the library
    default of 10/5/3)."""
    global _CRITERIA
    if _CRITERIA is None:
        C = o3d.t.pipelines.odometry.OdometryConvergenceCriteria
        _CRITERIA = [C(6), C(3), C(1)]
    return _CRITERIA


def rgbd_odometry(src_rgbd, tgt_rgbd, intrinsic, init, depth_scale, depth_max):
    """Hybrid (depth + intensity) multi-scale odometry, source -> target.

    Args:
        src_rgbd: source ``o3d.t.geometry.RGBDImage``.
        tgt_rgbd: target ``o3d.t.geometry.RGBDImage``.
        intrinsic: 3x3 ``o3c.Tensor`` camera matrix (CPU, Float64).
        init: 4x4 numpy initial guess mapping source points into the target
            frame (e.g. the IMU rotation prior or a drifted relative pose).
        depth_scale: raw depth units per meter.
        depth_max: clip depth beyond this (m).

    Returns:
        ``OdometryResult`` with ``transformation``, ``fitness``,
        ``inlier_rmse``.

    Raises:
        RuntimeError: when the 6x6 system is singular (degenerate geometry);
            callers treat this as "tracking failed".
    """
    return o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
        src_rgbd, tgt_rgbd, intrinsic,
        o3c.Tensor(init, o3c.Dtype.Float64),
        depth_scale, depth_max, odo_criteria(),
        o3d.t.pipelines.odometry.Method.Hybrid)
