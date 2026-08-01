"""Command-line interface.

``main()`` is the single entry point, reachable three equivalent ways::

    ./venv/bin/python realtime_pointcloud.py [options]
    ./venv/bin/python -m scanner [options]
    d455-scan [options]              # after `pip install -e .`
"""

import argparse

import open3d as o3d
import open3d.core as o3c

DESCRIPTION = """Realtime RGB-D 3D reconstruction with an Intel RealSense D455.

Hold the camera and move slowly around the scene. Camera pose is tracked
frame-to-model (KinectFusion-style dense SLAM) and every frame is fused
into a scalable TSDF voxel grid. On exit the trajectory is optimized with
a pose graph (loop closures between keyframes) and all frames are
re-integrated at a finer voxel size, then exported as a colored point
cloud (.ply).

Keys (in the preview window):
  q / ESC  finish scan, save point cloud
  s        save an intermediate snapshot of the point cloud
  r        reset the reconstruction (start over, keep streaming)
"""


def parse_args(argv=None):
    """Build the argument parser and parse ``argv`` (default: sys.argv)."""
    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output", default="scan.ply", help="output point cloud file")
    p.add_argument("--voxel-size", type=float, default=0.006,
                   help="TSDF voxel size (m) during realtime tracking")
    p.add_argument("--refine-voxel", type=float, default=0.004,
                   help="TSDF voxel size (m) for the offline refinement pass")
    p.add_argument("--no-refine", action="store_true",
                   help="skip the offline high-quality refinement pass")
    p.add_argument("--no-posegraph", action="store_true",
                   help="skip pose graph optimization before the refine pass")
    p.add_argument("--refine-only", metavar="FRAMES_DIR", default=None,
                   help="skip capture; run the offline pipeline on a previous "
                        "session's frames directory")
    p.add_argument("--keep-frames", action="store_true",
                   help="keep the recorded RGB-D frames next to the output")
    p.add_argument("--depth-min", type=float, default=0.15,
                   help="ignore depth closer than this (m)")
    p.add_argument("--depth-max", type=float, default=3.0,
                   help="ignore depth farther than this (m); keep <= 4 for D455 quality")
    p.add_argument("--block-count", type=int, default=20000,
                   help="voxel block budget of the TSDF volume "
                        "(~80 KB each; keep modest on low-RAM machines)")
    p.add_argument("--trunc-multiplier", type=float, default=8.0,
                   help="TSDF truncation distance in voxels")
    p.add_argument("--width", type=int, default=848)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--kf-every", type=int, default=15,
                   help="keyframe at least every N frames (pose graph + reloc)")
    p.add_argument("--kf-trans", type=float, default=0.10,
                   help="new keyframe when translation from last exceeds this (m)")
    p.add_argument("--kf-rot", type=float, default=10.0,
                   help="new keyframe when rotation from last exceeds this (deg)")
    p.add_argument("--temporal-filter", action="store_true",
                   help="enable the temporal depth filter (ghosting on a "
                        "moving handheld camera; minimal smoothing is used)")
    p.add_argument("--keep-auto-exposure", action="store_true",
                   help="do not freeze exposure/white balance after warm-up")
    p.add_argument("--no-imu", action="store_true",
                   help="disable the gyro rotation prior for tracking")
    p.add_argument("--mesh", action="store_true",
                   help="also export a triangle mesh next to the point cloud")
    p.add_argument("--no-view", action="store_true",
                   help="do not open the 3D viewer after saving")
    p.add_argument("--headless", action="store_true",
                   help="no preview window (stop with --max-frames or Ctrl+C)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="stop automatically after N frames (0 = run until quit)")
    return p.parse_args(argv)


def main(argv=None):
    """Parse arguments and run either the offline pipeline (--refine-only)
    or a full capture session followed by the offline pipeline."""
    args = parse_args(argv)
    device = o3c.Device("CUDA:0" if o3d.core.cuda.is_available() else "CPU:0")
    print(f"[info] Open3D device: {device}")

    if args.refine_only:
        from .pipeline import refine_only
        refine_only(args, device)
        return

    from .capture import CaptureSession
    session = CaptureSession(args, device)
    session.warm_up()
    session.run()
    session.finalize()
