"""Tuning constants for tracking, relocalization, loop closure, and the IMU.

These are deliberately module-level constants rather than CLI flags: they
rarely need to change per scan, but they are the first thing to adjust when
adapting the scanner to a new environment. See the "Tuning" section of the
README for guidance on which to try first.
"""

# --- tracking health -------------------------------------------------------
FITNESS_MIN = 0.3
"""Reject a tracking result whose inlier fraction is below this."""

RMSE_MAX = 0.08
"""Reject a tracking result whose inlier RMSE is above this. Hybrid odometry
mixes depth (m) and normalized-intensity residuals; typical good frames land
around 0.01-0.05, so tighten toward 0.05 once verified on real scans."""

MAX_STEP_M = 0.15
"""Reject a per-frame translation larger than this (m). At 30 FPS this
corresponds to ~4.5 m/s of camera motion — implausible for handheld use."""

# --- relocalization --------------------------------------------------------
RELOC_AFTER = 10
"""Consecutive lost frames before relocalization against stored keyframes."""

RELOC_FITNESS_MIN = 0.5
"""Minimum odometry fitness to accept a relocalization match."""

RELOC_KEYFRAMES = 6
"""Ring-buffer size of keyframes kept in memory for relocalization
(~2 MB each at 848x480)."""

# --- pose graph loop closure ----------------------------------------------
LOOP_MIN_GAP_FRAMES = 90
"""Keyframe pairs must be at least this many frames apart to count as a
potential loop closure (temporally distant)."""

LOOP_RADIUS_M = 0.6
"""... and their (drifted) camera positions must be within this distance
(spatially close)."""

LOOP_FITNESS_MIN = 0.6
"""Minimum odometry fitness to accept a loop-closure edge. Keep this high:
a false loop closure damages the reconstruction more than a missed one."""

LOOP_MAX_ATTEMPTS = 60
"""Cap on odometry attempts between candidate keyframe pairs (closest
pairs are tried first)."""

# --- IMU -------------------------------------------------------------------
GYRO_FPS = 200
"""Gyro stream rate (Hz). The D455 motion module supports 200/400."""

GYRO_MAX_DT = 0.05
"""Clamp gaps between gyro samples to this (s) so a hiccup in the motion
stream cannot inject a huge rotation step."""

GYRO_MAX_PRIOR_DEG = 45.0
"""Distrust (and discard) an inter-frame rotation prior larger than this."""
