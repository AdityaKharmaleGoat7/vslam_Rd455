"""Handheld realtime 3D scanner for the Intel RealSense D455.

The package implements a two-phase reconstruction pipeline:

1. **Capture** (:mod:`scanner.capture`) — realtime frame-to-model dense SLAM
   (KinectFusion-style) with a gyro rotation prior, tracking health checks,
   keyframe relocalization, and background frame recording.
2. **Offline** (:mod:`scanner.pipeline`) — keyframe pose graph optimization
   with loop closures (:mod:`scanner.posegraph`) followed by re-integration
   of every recorded frame at a finer TSDF voxel size
   (:mod:`scanner.mapping`).

Module map
----------
========================  ====================================================
:mod:`scanner.cli`        argument parsing and the ``main()`` entry point
:mod:`scanner.config`     tuning constants (tracking, loop closure, IMU)
:mod:`scanner.camera`     RealSense pipeline, depth filters, exposure lock
:mod:`scanner.imu`        gyro stream and rotation-prior integration
:mod:`scanner.capture`    the realtime capture session (tracking loop, HUD)
:mod:`scanner.tracking`   RGBD odometry helpers shared by all trackers
:mod:`scanner.posegraph`  keyframe selection and trajectory optimization
:mod:`scanner.mapping`    TSDF integration and point-cloud export
:mod:`scanner.pipeline`   the offline refine pass and ``--refine-only`` mode
:mod:`scanner.frameio`    background frame writer, frame/metadata (de)serialization
:mod:`scanner.transforms` small SO(3)/SE(3) helpers (no scipy dependency)
========================  ====================================================
"""

__version__ = "1.0.0"
