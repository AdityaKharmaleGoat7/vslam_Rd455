"""The realtime capture session.

:class:`CaptureSession` owns the camera, the SLAM model, and all per-scan
state. The per-frame flow is::

    grab -> filter+align -> clip depth -> track (IMU-primed odometry or
    frame-to-model) -> health checks -> [relocalize if lost too long] ->
    integrate into TSDF -> record frame (background writer) -> HUD

Crash safety: whatever ends the loop — quit key, Ctrl+C, or an unhandled
exception — the writer queue is flushed and the tracked poses are saved
into the frames directory, so ``--refine-only`` can always rebuild the
scan offline.
"""

import gc
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c
import pyrealsense2 as rs

from .camera import filter_and_align, lock_exposure, start_camera
from .config import (FITNESS_MIN, MAX_STEP_M, RELOC_AFTER,
                     RELOC_FITNESS_MIN, RELOC_KEYFRAMES, RMSE_MAX)
from .frameio import FrameWriter, save_meta
from .imu import start_imu
from .mapping import save_cloud
from .pipeline import offline_pipeline, view_result
from .tracking import rgbd_odometry

WINDOW_TITLE = "RealSense RGB-D reconstruction"


class CaptureSession:
    """One scanning session: camera + IMU + SLAM model + recording."""

    def __init__(self, args, device):
        self.args = args
        self.device = device

        (self.pipeline, self.profile, self.align, self.filters,
         self.intrinsic, self.depth_scale) = start_camera(args)
        print(f"[info] streaming {args.width}x{args.height}@{args.fps}, "
              f"depth range {args.depth_min}-{args.depth_max} m")

        self.imu_pipe, self.gyro = (None, None)
        if not args.no_imu:
            self.imu_pipe, self.gyro = start_imu(
                self.profile.get_stream(rs.stream.color))

        self.T = o3c.Tensor(np.eye(4), o3c.Dtype.Float64)  # camera-to-world
        self.model = o3d.t.pipelines.slam.Model(
            args.voxel_size, 16, args.block_count, self.T, device)
        self.input_frame = o3d.t.pipelines.slam.Frame(
            args.height, args.width, self.intrinsic, device)
        self.raycast_frame = o3d.t.pipelines.slam.Frame(
            args.height, args.width, self.intrinsic, device)

        self.idx = 0                 # count of successfully tracked frames
        self.lost_streak = 0
        self.fps = 0.0
        self.t_prev = time.time()
        self.poses = []              # tracked camera-to-world poses
        self.reloc_buf = []          # ring buffer of {pose, rgbd} keyframes
        self.filter_state = {}
        self.crashed = False

        self.frames_dir = Path(args.output).with_suffix("").parent / (
            Path(args.output).stem + "_frames")
        self.writer = None
        if not args.no_refine:
            self.frames_dir.mkdir(exist_ok=True)
            self.writer = FrameWriter(maxsize=8)

    # -- lifecycle ---------------------------------------------------------

    def warm_up(self):
        """Let auto-exposure settle, then freeze it and write metadata."""
        for _ in range(15):
            self.pipeline.wait_for_frames()
        if not self.args.keep_auto_exposure:
            lock_exposure(self.profile)
        if not self.args.no_refine:
            save_meta(self.frames_dir / "meta.json", self.intrinsic,
                      self.depth_scale, self.args)

    def run(self):
        """The capture loop. Returns normally on quit/interrupt; sets
        ``self.crashed`` on an unhandled exception."""
        try:
            while self._step():
                pass
        except KeyboardInterrupt:
            print("\n[info] interrupted")
        except Exception:
            self.crashed = True
            traceback.print_exc()
        finally:
            self._shutdown()

    def _shutdown(self):
        """Stop streams and flush recordings (crash-safe: always runs)."""
        self.pipeline.stop()
        if self.imu_pipe:
            self.imu_pipe.stop()
        cv2.destroyAllWindows()
        if self.writer:
            self.writer.close()
        if self.poses and not self.args.no_refine:
            np.save(self.frames_dir / "poses.npy", np.stack(self.poses))

    # -- per-frame steps ---------------------------------------------------

    def _step(self):
        """Process one camera frame. Returns False to end the loop."""
        frames = self.pipeline.wait_for_frames()
        depth_frame, color_frame = filter_and_align(
            frames, self.filters, self.align, self.filter_state)
        if not depth_frame or not color_frame:
            return True

        depth_np, color_bgr, depth_img, color_img = self._prepare(
            depth_frame, color_frame)

        tracked, relocalized = True, False
        if self.idx == 0:
            if self.gyro:
                self.gyro.reset()  # discard rotation accumulated pre-scan
        else:
            tracked = self._track()
            if not tracked and self.lost_streak + 1 >= RELOC_AFTER:
                tracked = relocalized = self._relocalize(color_img, depth_img)

        if tracked:
            self._integrate(depth_np, color_bgr, color_img, depth_img)
        else:
            self.lost_streak += 1
            if self.gyro:
                self.gyro.reset()  # prior is stale once tracking is lost

        now = time.time()
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - self.t_prev, 1e-6))
        self.t_prev = now

        if not self.args.headless and not self._hud(depth_np, color_bgr,
                                                    relocalized):
            return False
        return not (self.args.max_frames and self.idx >= self.args.max_frames)

    def _prepare(self, depth_frame, color_frame):
        """Convert librealsense frames to numpy + device images, clipping
        depth outside [depth-min, depth-max] to invalid (0)."""
        args = self.args
        depth_np = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
        # copy: the color buffer belongs to librealsense and is recycled
        color_bgr = np.asanyarray(color_frame.get_data()).copy()
        if depth_np.shape[:2] != (args.height, args.width):
            depth_np = cv2.resize(depth_np, (args.width, args.height),
                                  interpolation=cv2.INTER_NEAREST)
        near_m = np.float32(depth_np) / self.depth_scale
        depth_np[(near_m < args.depth_min) | (near_m > args.depth_max)] = 0

        color_rgb = np.ascontiguousarray(color_bgr[:, :, ::-1])
        depth_img = o3d.t.geometry.Image(depth_np).to(self.device)
        color_img = o3d.t.geometry.Image(color_rgb).to(self.device)
        self.input_frame.set_data_from_image("depth", depth_img)
        self.input_frame.set_data_from_image("color", color_img)
        return depth_np, color_bgr, depth_img, color_img

    def _track(self):
        """Track the input frame against the raycast model frame; apply the
        health checks. Returns True and updates the pose on success."""
        args = self.args
        prior = self.gyro.pop_prior() if self.gyro else np.eye(4)
        try:
            if self.gyro is not None:
                # Model.track_frame_to_model takes no init transform, so the
                # IMU prior goes through the lower-level odometry against
                # the raycast model frame
                src = o3d.t.geometry.RGBDImage(
                    self.input_frame.get_data_as_image("color"),
                    self.input_frame.get_data_as_image("depth"))
                tgt = o3d.t.geometry.RGBDImage(
                    self.raycast_frame.get_data_as_image("color"),
                    self.raycast_frame.get_data_as_image("depth"))
                result = rgbd_odometry(src, tgt, self.intrinsic, prior,
                                       self.depth_scale, args.depth_max)
            else:
                result = self.model.track_frame_to_model(
                    self.input_frame, self.raycast_frame, self.depth_scale,
                    args.depth_max, 0.07,
                    o3d.t.pipelines.odometry.Method.Hybrid)
        except RuntimeError:
            return False
        step = np.linalg.norm(result.transformation.numpy()[:3, 3])
        if (result.fitness < FITNESS_MIN or result.inlier_rmse > RMSE_MAX
                or step > MAX_STEP_M):
            return False
        self.T = self.T @ result.transformation
        return True

    def _relocalize(self, color_img, depth_img):
        """Try to re-acquire tracking against the stored keyframe ring
        buffer (newest first) instead of only the stale last raycast."""
        src = o3d.t.geometry.RGBDImage(color_img, depth_img)
        for kf in reversed(self.reloc_buf):
            try:
                res = rgbd_odometry(src, kf["rgbd"], self.intrinsic,
                                    np.eye(4), self.depth_scale,
                                    self.args.depth_max)
            except RuntimeError:
                continue
            if (res.fitness >= RELOC_FITNESS_MIN
                    and res.inlier_rmse <= RMSE_MAX):
                self.T = o3c.Tensor(kf["pose"] @ res.transformation.numpy(),
                                    o3c.Dtype.Float64)
                print(f"[info] relocalized (fitness {res.fitness:.2f})")
                return True
        return False

    def _integrate(self, depth_np, color_bgr, color_img, depth_img):
        """Fuse the tracked frame into the TSDF, refresh the raycast frame,
        record the pose/keyframe, and queue the frame for disk."""
        args = self.args
        self.lost_streak = 0
        self.model.update_frame_pose(self.idx, self.T)
        self.model.integrate(self.input_frame, self.depth_scale,
                             args.depth_max, args.trunc_multiplier)
        self.model.synthesize_model_frame(
            self.raycast_frame, self.depth_scale, args.depth_min,
            args.depth_max, args.trunc_multiplier, True)
        self.poses.append(self.T.numpy())
        if self.idx % args.kf_every == 0:
            self.reloc_buf.append({
                "pose": self.T.numpy(),
                "rgbd": o3d.t.geometry.RGBDImage(color_img, depth_img)})
            if len(self.reloc_buf) > RELOC_KEYFRAMES:
                self.reloc_buf.pop(0)
        if self.writer:
            self.writer.put(self.frames_dir / f"depth_{self.idx:05d}.png",
                            depth_np, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            self.writer.put(self.frames_dir / f"color_{self.idx:05d}.jpg",
                            color_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.idx += 1

    # -- UI ----------------------------------------------------------------

    def _hud(self, depth_np, color_bgr, relocalized):
        """Draw the preview window and handle keys. Returns False on quit."""
        args = self.args
        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_np,
                                alpha=255.0 / (args.depth_max * self.depth_scale)),
            cv2.COLORMAP_JET)
        depth_vis[depth_np == 0] = 0
        hud = np.hstack([color_bgr, depth_vis])
        t = self.T.numpy()[:3, 3]
        if self.lost_streak == 0:
            status = "RELOCALIZED" if relocalized else "TRACKING"
            color = (0, 220, 0)
        elif self.lost_streak >= RELOC_AFTER:
            status = f"LOST x{self.lost_streak} - relocalizing..."
            color = (0, 0, 255)
        else:
            status = f"LOST x{self.lost_streak} - go back!"
            color = (0, 0, 255)
        cv2.putText(hud, f"{status}  frames:{self.idx}  fps:{self.fps:4.1f}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(hud, f"pos: {t[0]:+.2f} {t[1]:+.2f} {t[2]:+.2f} m   "
                         f"[q]uit  [s]napshot  [r]eset",
                    (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(WINDOW_TITLE, hud)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            return False
        elif key == ord('s'):
            snap = Path(args.output).with_stem(
                Path(args.output).stem + datetime.now().strftime("_%H%M%S"))
            save_cloud(self.model.voxel_grid, snap)
        elif key == ord('r'):
            self._reset()
        return True

    def _reset(self):
        """Start the reconstruction over: fresh model, cleared trajectory,
        keyframes, gyro accumulator, writer queue, and frames directory."""
        print("[info] reconstruction reset")
        self.T = o3c.Tensor(np.eye(4), o3c.Dtype.Float64)
        self.model = o3d.t.pipelines.slam.Model(
            self.args.voxel_size, 16, self.args.block_count, self.T,
            self.device)
        self.idx = 0
        self.lost_streak = 0
        self.poses.clear()
        self.reloc_buf.clear()
        if self.gyro:
            self.gyro.reset()
        if self.writer:
            self.writer.drain()
            shutil.rmtree(self.frames_dir, ignore_errors=True)
            self.frames_dir.mkdir(exist_ok=True)
            save_meta(self.frames_dir / "meta.json", self.intrinsic,
                      self.depth_scale, self.args)

    # -- finalization ------------------------------------------------------

    def finalize(self):
        """Post-capture: save trajectory + metadata, then either export the
        realtime volume (``--no-refine``) or run the offline pipeline."""
        args = self.args
        print(f"[info] integrated {self.idx} frames")
        if self.idx == 0:
            return

        if self.crashed and not args.no_refine:
            print(f"[error] capture crashed. Frames and poses were saved; run\n"
                  f"        {sys.argv[0]} --refine-only {self.frames_dir} "
                  f"--output {args.output}\n"
                  f"        to build the point cloud offline.")
            sys.exit(1)

        np.save(Path(args.output).with_suffix(".trajectory.npy"),
                np.stack(self.poses))
        # intrinsics + depth scale next to the trajectory so saved frames
        # can be reprocessed later even after the frames dir is cleaned up
        save_meta(Path(args.output).with_suffix(".meta.json"),
                  self.intrinsic, self.depth_scale, args)

        if args.no_refine:
            n = save_cloud(self.model.voxel_grid, args.output,
                           mesh=args.mesh, clean=True)
        else:
            # free the realtime volume before allocating the finer one (low RAM)
            self.model = self.input_frame = self.raycast_frame = None
            self.reloc_buf.clear()
            gc.collect()
            try:
                n = offline_pipeline(self.frames_dir, self.poses,
                                     self.intrinsic, self.depth_scale,
                                     args, self.device)
            except Exception:
                traceback.print_exc()
                print(f"[error] offline refinement crashed. Frames are "
                      f"intact; run\n"
                      f"        {sys.argv[0]} --refine-only {self.frames_dir} "
                      f"--output {args.output}")
                sys.exit(1)
            if not args.keep_frames:
                shutil.rmtree(self.frames_dir, ignore_errors=True)

        if n and not args.no_view:
            view_result(args.output)
