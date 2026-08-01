"""Gyro-aided tracking support.

The D455's motion module streams on its own callback at a much higher rate
(~200 Hz) than the video streams. :class:`GyroIntegrator` accumulates the
angular-velocity samples that arrive between two video frames into a single
rotation, which the capture loop consumes as the initial guess for RGBD
odometry (``Model.track_frame_to_model`` accepts no init transform, so the
prior is fed through the lower-level odometry path — see
:mod:`scanner.tracking`).
"""

import threading

import numpy as np
import pyrealsense2 as rs

from .config import GYRO_FPS, GYRO_MAX_DT, GYRO_MAX_PRIOR_DEG
from .transforms import rotation_angle_deg, so3_exp


class GyroIntegrator:
    """Accumulates gyro rotation between video frames into a pose prior.

    The callback composes ``R <- R @ Exp(w * dt)`` per sample (exact for
    piecewise-constant angular velocity) in the gyro frame; ``pop_prior``
    conjugates the result into the color-camera frame.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.R = np.eye(3)          # accumulated Exp(w*dt) products, gyro frame
        self.last_ts = None
        self.R_cg = np.eye(3)       # gyro -> color rotation

    def callback(self, frame):
        """librealsense callback; invoked per motion frame at GYRO_FPS."""
        m = frame.as_motion_frame()
        if not m:
            return
        w = m.get_motion_data()
        ts = m.get_timestamp() / 1000.0
        with self.lock:
            if self.last_ts is not None:
                dt = min(max(ts - self.last_ts, 0.0), GYRO_MAX_DT)
                self.R = self.R @ so3_exp(np.array([w.x, w.y, w.z]) * dt)
            self.last_ts = ts

    def pop_prior(self):
        """Return and reset the accumulated rotation as a 4x4 prior mapping
        current-frame points into previous-frame coordinates (the init for
        source=current -> target=previous odometry). Implausibly large
        rotations (sensor glitch, dropped frames) collapse to identity."""
        with self.lock:
            Rg, self.R = self.R, np.eye(3)
        R = self.R_cg @ Rg @ self.R_cg.T
        if rotation_angle_deg(R) > GYRO_MAX_PRIOR_DEG:
            return np.eye(4)
        T = np.eye(4)
        T[:3, :3] = R
        return T

    def reset(self):
        """Drop the accumulated rotation (used when tracking is lost — the
        prior is only meaningful relative to the last *tracked* frame)."""
        with self.lock:
            self.R = np.eye(3)


def start_imu(color_stream_profile):
    """Start a second pipeline for the motion module.

    Returns:
        ``(pipeline, GyroIntegrator)`` on success, ``(None, None)`` if the
        gyro stream is unavailable (missing motion module, USB bandwidth,
        permissions) — the caller then tracks without an IMU prior.
    """
    try:
        integ = GyroIntegrator()
        imu_pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, GYRO_FPS)
        profile = imu_pipe.start(cfg, integ.callback)
        try:
            ext = profile.get_stream(rs.stream.gyro).get_extrinsics_to(
                color_stream_profile)
            # rs2_extrinsics.rotation is column-major
            integ.R_cg = np.asarray(ext.rotation, np.float64).reshape(3, 3).T
        except Exception:
            pass  # near-identity on the D455; keep identity fallback
        print(f"[info] gyro stream enabled ({GYRO_FPS} Hz rotation prior)")
        return imu_pipe, integ
    except Exception as e:
        print(f"[warn] gyro unavailable, tracking without IMU prior ({e})")
        return None, None
