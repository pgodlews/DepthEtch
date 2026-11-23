"""Whole-clip correction of Depth Pro 2 using a depth camera as anchors.

For each clip it solves one smooth, zero-lag inverse-depth correction across all frames:

    1/z = a(t) * 1/d + b(t)

then adds a smooth local correction. Beyond the camera's trusted range the mapping becomes a pure
scale, so far pixels and sky cannot flip or blow up. The depth camera is a reference, not ground truth.
"""
import concurrent.futures as cf
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .camera import Frames, align_depth, edge_mask

H, W, BLOCK, HOLD = 360, 640, 40, 0.30
GRID_H, GRID_W, CELL = 90, 160, 4
N_TRAIN, N_TEST = 12000, 3000
PRIOR = 0.05
CLIP_RESIDUAL = 0.35


class Points:
    """Anchor or test pixels from all frames, in flat arrays."""

    def __init__(self, parts, frames):
        self.T = frames
        self.idx = np.concatenate([np.full(len(p[0]), t, np.int32) for t, p in enumerate(parts)])
        self.u, self.v, self.c, self.r = [np.concatenate([p[j] for p in parts]) for j in range(4)]
        self.extra = np.concatenate([p[4] for p in parts]) if parts and len(parts[0]) > 4 else None
        self.count = np.bincount(self.idx, minlength=frames).astype(np.float64)
        self.offsets = np.concatenate([[0], np.cumsum(self.count).astype(np.int64)])

    def frame(self, t):
        return slice(int(self.offsets[t]), int(self.offsets[t + 1]))


def pick(mask, n, rng, z, c, extra=None):
    index = np.flatnonzero(mask)
    if index.size > n:
        index = rng.choice(index, n, replace=False)
    v, u = np.unravel_index(index, (H, W))
    out = [u.astype(np.int16), v.astype(np.int16), c.flat[index].astype(np.float32), (1 / z.flat[index]).astype(np.float32)]
    if extra is not None:
        out.append(extra.flat[index].astype(np.float32))
    return out


def nearest_fill(z, train):
    if not train.any():
        return np.full_like(z, np.nan)
    _, labels = cv2.distanceTransformWithLabels((~train).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(int(labels.max()) + 1, np.float32)
    lut[labels[train]] = z[train]
    return lut[labels]


def per_frame_fit(p):
    """Independent robust fit per frame: median scale, and scale + shift in inverse depth."""
    s, a, b = np.full(p.T, np.nan), np.full(p.T, np.nan), np.full(p.T, np.nan)
    for t in range(p.T):
        f = p.frame(t)
        c, r = p.c[f], p.r[f]
        if c.size < 200:
            continue
        s[t] = np.median(c / r)  # depth scale = (1/r) / (1/c)
        A = np.stack([c, np.ones_like(c)], 1)
        w = np.ones_like(c)
        for _ in range(5):
            sw = np.sqrt(w)
            sol = np.linalg.lstsq(A * sw[:, None], r * sw, rcond=None)[0]
            e = np.abs(r - A @ sol)
            w = np.minimum(1, 1.345 * 1.4826 * max(np.median(e), 1e-5) / np.maximum(e, 1e-9))
        a[t], b[t] = sol
    good = np.isfinite(s)
    fallback = np.median(s[good]) if good.any() else 1.0
    s[~good] = fallback
    for x, default in ((a, 1 / fallback), (b, 0.0)):
        bad = ~np.isfinite(x)
        x[bad] = np.median(x[~bad]) if (~bad).any() else default
    return s, a, b


def to_inverse(c, a, b, r_far):
    """a*c + b inside the trusted range; a pure scale beyond it, so depth stays positive and ordered."""
    a = np.maximum(a, 0.05)
    c_far = np.maximum((r_far - b) / a, 1e-3)
    r_at_far = a * c_far + b
    return np.where(c >= c_far, a * c + b, r_at_far * c / c_far)


def smooth_solve(p, tau, a_init, b_init, iterations=4):
    """Robust joint fit of a(t), b(t) with an edge-preserving smoothness penalty across time."""
    T = p.T
    a, b = a_init.copy(), b_init.copy()
    a0, b0 = np.median(a), np.median(b)
    delta_a, delta_b = 0.05 * max(abs(a0), 1e-3), 0.02
    for _ in range(iterations):
        e = p.r - (a[p.idx] * p.c + b[p.idx])
        w = np.ones_like(e, np.float64)
        for t in range(T):
            f = p.frame(t)
            if e[f].size:
                s = 1.4826 * max(np.median(np.abs(e[f])), 1e-4)
                w[f] = np.minimum(1, 1.5 * s / np.maximum(np.abs(e[f]), 1e-9))
        total = np.bincount(p.idx, w, T)
        norm = np.minimum(p.count / 2000, 1) / np.maximum(total, 1e-9)
        s11 = np.bincount(p.idx, w * p.c * p.c, T) * norm
        s12 = np.bincount(p.idx, w * p.c, T) * norm
        s22 = np.bincount(p.idx, w, T) * norm
        y1 = np.bincount(p.idx, w * p.c * p.r, T) * norm
        y2 = np.bincount(p.idx, w * p.r, T) * norm
        la = tau * tau * max(np.median(s11), 1e-3)
        lb = tau * tau * max(np.median(s22), 1e-3)
        wa = np.minimum(1, delta_a / np.maximum(np.abs(np.diff(a)), 1e-9)) if T > 1 else np.zeros(0)
        wb = np.minimum(1, delta_b / np.maximum(np.abs(np.diff(b)), 1e-9)) if T > 1 else np.zeros(0)
        mu_a, mu_b = 1e-3 * max(np.median(s11), 1e-3), 1e-3 * max(np.median(s22), 1e-3)
        da, db = np.zeros(T), np.zeros(T)
        if T > 1:
            da[:-1] += la * wa
            da[1:] += la * wa
            db[:-1] += lb * wb
            db[1:] += lb * wb
        rows, cols, vals = [], [], []
        def add(r, c_, v):
            rows.append(r), cols.append(c_), vals.append(v)
        ia, ib = 2 * np.arange(T), 2 * np.arange(T) + 1
        add(ia, ia, s11 + mu_a + da)
        add(ib, ib, s22 + mu_b + db)
        add(ia, ib, s12)
        add(ib, ia, s12)
        if T > 1:
            add(ia[:-1], ia[1:], -la * wa)
            add(ia[1:], ia[:-1], -la * wa)
            add(ib[:-1], ib[1:], -lb * wb)
            add(ib[1:], ib[:-1], -lb * wb)
        M = sparse.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * T, 2 * T))
        rhs = np.zeros(2 * T)
        rhs[ia], rhs[ib] = y1 + mu_a * a0, y2 + mu_b * b0
        x = spsolve(M, rhs)
        a, b = np.maximum(x[ia], 0.05), x[ib]
    return a, b


def zero_phase(x, alpha):
    if alpha >= 1:
        return x
    y = x.copy()
    for t in range(1, len(y)):
        y[t] = alpha * y[t] + (1 - alpha) * y[t - 1]
    for t in range(len(y) - 2, -1, -1):
        y[t] = alpha * y[t] + (1 - alpha) * y[t + 1]
    return y


def grids(p, e, sigma):
    key = p.idx.astype(np.int64) * (GRID_H * GRID_W) + (p.v.astype(np.int64) // CELL) * GRID_W + p.u.astype(np.int64) // CELL
    size = p.T * GRID_H * GRID_W
    N = np.bincount(key, e, size).astype(np.float32).reshape(p.T, GRID_H, GRID_W)
    D = np.bincount(key, minlength=size).astype(np.float32).reshape(p.T, GRID_H, GRID_W)
    for t in range(p.T):
        cv2.GaussianBlur(N[t], (0, 0), sigma, dst=N[t], borderType=cv2.BORDER_REPLICATE)
        cv2.GaussianBlur(D[t], (0, 0), sigma, dst=D[t], borderType=cv2.BORDER_REPLICATE)
    return N, D


def field_from(p, r_hat, sigma, alpha):
    e = np.clip(np.log(p.r / np.maximum(r_hat, 1e-6)), -CLIP_RESIDUAL, CLIP_RESIDUAL)
    N, D = grids(p, e, sigma)
    N, D = zero_phase(N, alpha), zero_phase(D, alpha)
    return N / (D + PRIOR)


def sample_field(F, idx, u, v):
    x = np.clip((u.astype(np.float32) - 1.5) / CELL, 0, GRID_W - 1.001)
    y = np.clip((v.astype(np.float32) - 1.5) / CELL, 0, GRID_H - 1.001)
    x0, y0 = x.astype(np.int32), y.astype(np.int32)
    fx, fy = x - x0, y - y0
    return (1 - fy) * ((1 - fx) * F[idx, y0, x0] + fx * F[idx, y0, x0 + 1]) + fy * ((1 - fx) * F[idx, y0 + 1, x0] + fx * F[idx, y0 + 1, x0 + 1])


def score(z, x):
    e = np.abs(x - z)
    return dict(mean_rel=float(np.mean(e / z)), median_cm=float(100 * np.median(e)), within_5pct=float(np.mean(e / z < 0.05)))


def global_hat(p, a, b, r_far):
    return to_inverse(p.c, a[p.idx], b[p.idx], r_far)


def load_clip(frames_dir, prediction_dir, seed=0, limit=None):
    """Sample anchors from every frame. Hidden image blocks stay fixed for the whole clip."""
    run = Path(prediction_dir)
    summary = json.loads((run / "summary.json").read_text())
    fx, (zmin, zmax) = summary["f_px"], summary["depth_range_m"]
    frames = Frames(frames_dir, limit)
    if (frames.width, frames.height) != (2 * W, 2 * H):
        raise SystemExit(f"expected {2 * W}x{2 * H} colour, found {frames.width}x{frames.height}")
    cells = np.random.default_rng(1000 + seed).random((H // BLOCK + 1, W // BLOCK + 1)) < HOLD
    hold = np.kron(cells, np.ones((BLOCK, BLOCK), bool))[:H, :W]

    def one(row):
        i = int(row["index"])
        depth = cv2.imread(str(frames.root / "depth" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise IOError(f"missing depth frame {i}")
        z = align_depth(depth, frames.Kd, frames.Kc, frames.R, frames.t, frames.width, frames.height)[::2, ::2]
        c = np.load(run / "pred" / f"{i:06d}.npy").astype(np.float32) * 1280 / fx
        good = np.isfinite(z) & (z >= zmin) & (z <= zmax) & np.isfinite(c) & (c > 1e-4) & ~edge_mask(z)
        train, test = good & ~hold, good & hold
        rng = np.random.default_rng([seed, i])
        return pick(train, N_TRAIN, rng, z, c), pick(test, N_TEST, rng, z, c, nearest_fill(z, train)), pick(good, N_TRAIN, rng, z, c)

    with cf.ThreadPoolExecutor(12) as pool:
        loaded = list(pool.map(one, frames.rows))
    T = len(loaded)
    return dict(frames=frames, fx=fx, zmin=zmin, zmax=zmax, T=T, hold_share=float(hold.mean()),
                train=Points([x[0] for x in loaded], T), test=Points([x[1] for x in loaded], T),
                all=Points([x[2] for x in loaded], T))


def solve(points, tau, sigma, alpha, r_far):
    """The correction for one set of anchors: a(t), b(t) and the local log field."""
    _, a0, b0 = per_frame_fit(points)
    a, b = smooth_solve(points, tau, a0, b0)
    return a, b, field_from(points, global_hat(points, a, b, r_far), sigma, alpha)


def hidden_block_report(clip, tau, sigma, alpha):
    """Score on image blocks the correction never saw, against the raw model and a depth-only fill."""
    p, q = clip["train"], clip["test"]
    r_far = 1 / clip["zmax"]
    a, b, F = solve(p, tau, sigma, alpha, r_far)
    base = to_inverse(q.c, a[q.idx], b[q.idx], r_far)
    z = 1 / q.r
    near = q.extra
    ok = np.isfinite(near) & (near > 0)
    predictions = {"Depth Pro 2 as-is": 1 / q.c, "Depth camera, nearest fill": near,
                   "DepthEtch correction": 1 / (base * np.exp(sample_field(F, q.idx, q.u, q.v)))}
    return {name: score(z[ok], x[ok]) for name, x in predictions.items()}
