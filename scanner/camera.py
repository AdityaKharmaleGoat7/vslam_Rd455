"""RealSense D455 video pipeline: streaming, depth post-processing, depth-to-
color alignment, and the exposure/white-balance lock.

Filter ordering matters: the spatial (and optional temporal) filters assume
the sensor's native depth sampling, so they are applied to the *original*
depth frame **before** ``rs.align`` — aligning first resamples depth to the
color viewpoint and breaks those assumptions.
"""

import numpy as np
import open3d.core as o3c
import pyrealsense2 as rs


def start_camera(args):
    """Start depth+color streaming and build the processing chain.

    Returns:
        Tuple of ``(pipeline, profile, align, filters, intrinsic,
        depth_scale)`` where ``intrinsic`` is the color stream's 3x3
        ``o3c.Tensor`` (depth is aligned to color) and ``depth_scale`` is
        raw depth units per meter (Open3D convention).
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    try:
        depth_sensor.set_option(rs.option.visual_preset,
                                float(rs.rs400_visual_preset.high_accuracy))
    except Exception:
        pass  # preset not supported on this firmware; defaults are fine
    depth_scale_m = depth_sensor.get_depth_scale()          # meters per unit
    depth_scale = 1.0 / depth_scale_m                       # units per meter (Open3D convention)

    align = rs.align(rs.stream.color)

    # depth post-processing: filter in disparity space, then back.
    # temporal filter ghosts on a moving handheld camera -> off by default,
    # and minimal smoothing / short persistence when enabled.
    filters = [rs.disparity_transform(True), rs.spatial_filter()]
    if args.temporal_filter:
        tf = rs.temporal_filter()
        try:
            tf.set_option(rs.option.filter_smooth_alpha, 0.6)
            tf.set_option(rs.option.filter_smooth_delta, 20.0)
            tf.set_option(rs.option.holes_fill, 1)
        except Exception:
            pass
        filters.append(tf)
    filters.append(rs.disparity_transform(False))

    # intrinsics of the color stream (depth is aligned to it)
    color_profile = profile.get_stream(rs.stream.color)
    intr = color_profile.as_video_stream_profile().get_intrinsics()
    intrinsic = o3c.Tensor([[intr.fx, 0.0, intr.ppx],
                            [0.0, intr.fy, intr.ppy],
                            [0.0, 0.0, 1.0]], o3c.Dtype.Float64)
    return pipeline, profile, align, filters, intrinsic, depth_scale


def filter_and_align(frames, filters, align, state):
    """Filter the ORIGINAL depth frame, then align.

    Some librealsense builds reject frameset input to the processing
    blocks; the first failure flips ``state["no_frameset_filter"]`` and all
    subsequent frames fall back to filtering the aligned depth (the old,
    slightly incorrect behavior) with a one-time warning.

    Returns:
        ``(depth_frame, color_frame)`` — either may be falsy on a bad grab.
    """
    if not state.get("no_frameset_filter"):
        try:
            f = frames
            for flt in filters:
                f = flt.process(f)
            aligned = align.process(f.as_frameset())
            return aligned.get_depth_frame(), aligned.get_color_frame()
        except Exception:
            state["no_frameset_filter"] = True
            print("[warn] filters do not accept framesets on this "
                  "librealsense build; filtering after align instead")
    aligned = align.process(frames)
    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()
    if depth_frame:
        for flt in filters:
            depth_frame = flt.process(depth_frame)
    return depth_frame, color_frame


def lock_exposure(profile):
    """Freeze auto-exposure and auto white balance at their settled values
    so TSDF colors stay consistent across the scan."""
    try:
        dev = profile.get_device()
        color_sensor = None
        for s in dev.query_sensors():
            if "RGB" in s.get_info(rs.camera_info.name):
                color_sensor = s
                break
        if color_sensor is None:
            raise RuntimeError("no RGB sensor found")
        exp = color_sensor.get_option(rs.option.exposure)
        wb = color_sensor.get_option(rs.option.white_balance)
        color_sensor.set_option(rs.option.enable_auto_exposure, 0)
        color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
        color_sensor.set_option(rs.option.exposure, exp)
        color_sensor.set_option(rs.option.white_balance, wb)
        print(f"[info] exposure locked at {exp:.0f} us, "
              f"white balance at {wb:.0f} K")
    except Exception as e:
        print(f"[warn] could not lock exposure/white balance: {e}")
