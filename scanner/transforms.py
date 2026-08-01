"""Small SO(3)/SE(4) helpers built on ``cv2.Rodrigues`` (no scipy dependency)."""

import cv2
import numpy as np


def so3_exp(rvec):
    """Exponential map: rotation vector (axis * angle, rad) -> 3x3 rotation."""
    return cv2.Rodrigues(np.asarray(rvec, np.float64))[0]


def rotation_angle_deg(R):
    """Geodesic angle (deg) of a 3x3 rotation matrix."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def interp_transform(Ta, Tb, t):
    """Interpolate between two 4x4 rigid transforms at fraction ``t`` in [0, 1].

    Rotation is interpolated along the geodesic (equivalent to quaternion
    slerp), translation linearly. Used to spread pose-graph corrections from
    keyframes onto the frames between them.
    """
    Ra, Rb = Ta[:3, :3], Tb[:3, :3]
    rvec = cv2.Rodrigues(Ra.T @ Rb)[0]
    out = np.eye(4)
    out[:3, :3] = Ra @ so3_exp(rvec * t)
    out[:3, 3] = (1.0 - t) * Ta[:3, 3] + t * Tb[:3, 3]
    return out
