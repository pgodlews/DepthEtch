"""Read extracted RealSense frames, undistort colour and align depth to it."""
from __future__ import annotations

import collections
import concurrent.futures as cf
import csv
import json
from pathlib import Path

import cv2
import numpy as np

# Librealsense reports this model for D4xx colour streams; a ROS camera_info labels them plumb_bob.
DISTORTION_MODEL = "inverse_brown_conrady"


def brown_conrady(x, y, c):
    """Apply the lens polynomial: ideal normalised coordinates to recorded (distorted) ones."""
    r2 = x * x + y * y
    f = 1 + c[0] * r2 + c[1] * r2 * r2 + c[4] * r2 * r2 * r2
    return (x * f + 2 * c[2] * x * y + c[3] * (r2 + 2 * x * x),
            y * f + 2 * c[3] * x * y + c[2] * (r2 + 2 * y * y))


def undistort_maps(K, c, width, height):
    """For each pixel of the ideal pinhole image, find where to sample the recorded image.

    Measured against librealsense 2.59 for this stream: rs2_project_point_to_pixel equals the
    polynomial applied to the ideal point (within 0.03 px), so no iteration is needed here.
    """
    fx, fy, cx, cy = K[0], K[4], K[2], K[5]
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    xd, yd = brown_conrady((u - cx) / fx, (v - cy) / fy, c)
    return (xd * fx + cx).astype(np.float32), (yd * fy + cy).astype(np.float32)


def quat_matrix(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def transform_to_link(tf, frame):
    """4x4 pose of `frame` in camera_link, following the static transform tree."""
    parent = {x["child"]: x for x in tf}
    T = np.eye(4)
    while frame != "camera_link":
        x = parent[frame]
        A = np.eye(4)
        A[:3, :3] = quat_matrix(x["q"])
        A[:3, 3] = x["t"]
        T = A @ T
        frame = x["parent"]
    return T


def depth_to_color(meta):
    a = transform_to_link(meta["tf"], meta["/camera/depth/camera_info"]["frame"])
    b = transform_to_link(meta["tf"], meta["/camera/color/camera_info"]["frame"])
    T = np.linalg.inv(b) @ a
    return T[:3, :3], T[:3, 3]


def align_depth(depth_mm, Kd, Kc, R, t, width, height):
    """Reproject depth into the ideal colour camera; the nearest surface wins each pixel."""
    v, u = np.nonzero(depth_mm)
    z = depth_mm[v, u].astype(np.float64) * 0.001
    X = np.stack([(u - Kd[2]) / Kd[0] * z, (v - Kd[5]) / Kd[4] * z, z])
    P = R @ X + t[:, None]
    keep = P[2] > 0.05
    P = P[:, keep]
    uc = Kc[0] * P[0] / P[2] + Kc[2]
    vc = Kc[4] * P[1] / P[2] + Kc[5]
    order = np.argsort(-P[2])
    uc, vc, zc = uc[order], vc[order], P[2][order].astype(np.float32)
    out = np.full((height, width), np.inf, np.float32)
    # One depth pixel covers about 1.5 colour pixels, so write a 2x2 footprint to avoid pinholes.
    u0, v0 = np.floor(uc).astype(np.int32), np.floor(vc).astype(np.int32)
    for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
        a, b = u0 + du, v0 + dv
        ok = (a >= 0) & (a < width) & (b >= 0) & (b < height)
        layer = np.full((height, width), np.inf, np.float32)
        layer[b[ok], a[ok]] = zc[ok]  # sorted far to near, so the nearest write is last
        out = np.minimum(out, layer)
    out[~np.isfinite(out)] = np.nan
    return out


def edge_mask(depth, rel_jump=0.04, k=5):
    d = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    kernel = np.ones((k, k), np.uint8)
    high, low = cv2.dilate(d, kernel), cv2.erode(d, kernel)
    holes = cv2.dilate((d <= 0).astype(np.uint8), kernel) > 0
    return ((high - low) > rel_jump * np.maximum(high, 1e-3)) | holes


class Frames:
    def __init__(self, root, limit=None, start=0):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        with open(self.root / "index.csv") as f:
            rows = list(csv.DictReader(f))
        self.rows = rows[start:start + limit if limit else None]
        c, d = self.meta["/camera/color/camera_info"], self.meta["/camera/depth/camera_info"]
        self.width, self.height = c["width"], c["height"]
        self.Kc, self.Kd = c["k"], d["k"]
        self.coeffs = c["d"]
        self.map1, self.map2 = undistort_maps(self.Kc, self.coeffs, self.width, self.height)
        self.R, self.t = depth_to_color(self.meta)

    def load(self, row):
        i = int(row["index"])
        bgr = cv2.imread(str(self.root / "rgb" / f"{i:06d}.png"), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(self.root / "depth" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
        if bgr is None or depth is None:
            raise IOError(f"missing frame {i}")
        rgb = cv2.remap(bgr[..., ::-1].copy(), self.map1, self.map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return i, rgb, align_depth(depth, self.Kd, self.Kc, self.R, self.t, self.width, self.height)

    def __iter__(self):
        with cf.ThreadPoolExecutor(8) as pool:
            queue = collections.deque()
            for row in self.rows:
                queue.append(pool.submit(self.load, row))
                if len(queue) >= 16:
                    yield queue.popleft().result()
            while queue:
                yield queue.popleft().result()
