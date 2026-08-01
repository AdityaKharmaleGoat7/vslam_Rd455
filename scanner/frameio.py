"""Frame and metadata I/O: the background frame writer, RGB-D frame loading
for the offline pipeline, and the scan metadata JSON.

A frames directory recorded by a capture session contains::

    depth_00000.png ...   16-bit depth (raw units), PNG compression 1
    color_00000.jpg ...   BGR color, JPEG quality 95
    meta.json             intrinsics, depth scale, capture parameters
    poses.npy             (N, 4, 4) tracked camera-to-world poses

which is everything ``--refine-only`` needs to rebuild the point cloud.
"""

import json
import queue
import threading
import time

import cv2
import numpy as np
import open3d as o3d


class FrameWriter:
    """Writes frames to disk on a background thread so PNG/JPEG encoding
    never blocks the tracking loop.

    The queue is bounded with a *block* policy: if the disk falls behind,
    the tracking loop stalls (with a rate-limited warning) rather than
    silently dropping frames — dropped frames would leave holes in the
    recorded sequence that the offline pipeline cannot repair.
    """

    def __init__(self, maxsize=8):
        self.q = queue.Queue(maxsize)
        self._last_warn = 0.0
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def put(self, path, image, params=None):
        """Queue one ``cv2.imwrite(path, image, params)`` call."""
        if self.q.full():
            now = time.time()
            if now - self._last_warn > 2.0:
                print("[warn] frame writer queue full - disk is falling "
                      "behind, tracking loop will stall")
                self._last_warn = now
        self.q.put((str(path), image, params))

    def _run(self):
        while True:
            item = self.q.get()
            try:
                if item is None:
                    return
                path, image, params = item
                cv2.imwrite(path, image, params if params else [])
            except Exception as e:
                print(f"[warn] frame write failed: {e}")
            finally:
                self.q.task_done()

    def drain(self):
        """Discard queued frames and wait for the in-flight write (used on
        reconstruction reset, before the frames directory is cleared)."""
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
            else:
                self.q.task_done()
        self.q.join()

    def flush(self):
        """Block until every queued frame has been written."""
        self.q.join()

    def close(self):
        """Flush, stop, and join the writer thread."""
        self.flush()
        self.q.put(None)
        self.t.join()


def load_rgbd(frames_dir, i, device):
    """Load recorded frame ``i`` as an ``o3d.t.geometry.RGBDImage`` on
    ``device``, or ``None`` if either file is missing."""
    depth_np = cv2.imread(str(frames_dir / f"depth_{i:05d}.png"),
                          cv2.IMREAD_UNCHANGED)
    color_bgr = cv2.imread(str(frames_dir / f"color_{i:05d}.jpg"))
    if depth_np is None or color_bgr is None:
        return None
    depth = o3d.t.geometry.Image(depth_np.astype(np.uint16)).to(device)
    color = o3d.t.geometry.Image(
        np.ascontiguousarray(color_bgr[:, :, ::-1])).to(device)
    return o3d.t.geometry.RGBDImage(color, depth)


def build_meta(intrinsic, depth_scale, args):
    """Collect intrinsics, depth scale, and capture parameters into a dict
    so recorded frames can be reprocessed later without the camera."""
    K = intrinsic.numpy()
    return {
        "width": args.width, "height": args.height, "fps": args.fps,
        "fx": K[0, 0], "fy": K[1, 1], "ppx": K[0, 2], "ppy": K[1, 2],
        "depth_scale": depth_scale,
        "depth_min": args.depth_min, "depth_max": args.depth_max,
        "voxel_size": args.voxel_size, "refine_voxel": args.refine_voxel,
        "block_count": args.block_count,
        "trunc_multiplier": args.trunc_multiplier,
    }


def save_meta(path, intrinsic, depth_scale, args):
    """Write :func:`build_meta` output as JSON to ``path``."""
    with open(path, "w") as f:
        json.dump(build_meta(intrinsic, depth_scale, args), f, indent=2)


def load_meta(path):
    """Read a scan metadata JSON written by :func:`save_meta`."""
    with open(path) as f:
        return json.load(f)
