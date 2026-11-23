"""The DepthEtch view: colour is metres, etched lines come from the image and Depth Pro 2.

Vivid colour means the depth camera measured that area. Muted colour means the value was filled in
from Depth Pro 2's shape.
"""
import cv2
import numpy as np

from .correction import to_inverse

BIG, SMALL = (960, 540), (320, 180)


def load_correction(path, fpx):
    """The whole-clip correction written by stabilise.py."""
    c = np.load(path)
    if abs(float(c["fx"]) - fpx) > 1e-3:
        raise SystemExit("correction was made with a different focal length")
    return dict(a=c["a"], b=c["b"], field=c["field"].astype(np.float32), r_far=float(c["r_far"]),
                frame={int(x): t for t, x in enumerate(c["frame_index"])})


def stabilised_metric(inv, fpx, correction, frame):
    """Metric depth for one frame. The field is log(reference inverse depth / corrected inverse depth)."""
    t = correction["frame"][frame]
    h, w = inv.shape
    r = to_inverse(inv * 1280 / fpx, correction["a"][t], correction["b"][t], correction["r_far"])
    field = cv2.resize(correction["field"][t], (w, h), interpolation=cv2.INTER_LINEAR)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1 / (r * np.exp(field))


def depth_color(x, zmin, zmax):
    """Log scale, so a room at 1-3 m uses more of the palette. Near is red, far is blue."""
    finite = np.isfinite(x) & (x > 0)
    v = np.log(np.clip(np.where(finite, x, zmax), zmin, zmax))
    n = (v - np.log(zmin)) / (np.log(zmax) - np.log(zmin))
    img = cv2.applyColorMap(((1 - n) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~finite] = 0
    return img


def normalised(x):
    return x / max(float(np.percentile(np.abs(x), 99)), 1e-4)


def depthetch(rgb, metric, measured, zmin, zmax, strength=36.0):
    """Etch image and depth detail into a metric colour map, muting unmeasured areas. Returns BGR."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    gray = cv2.bilateralFilter(gray, 5, 0.06, 2)
    visible = normalised(gray - cv2.GaussianBlur(gray, (0, 0), 5))
    log_depth = np.log(np.clip(metric, zmin * 0.5, zmax * 2)).astype(np.float32)
    shape = normalised(log_depth - cv2.GaussianBlur(log_depth, (0, 0), 4))
    detail = np.clip(0.75 * visible - 0.55 * shape, -1, 1)
    lab = cv2.cvtColor(depth_color(metric, zmin, zmax).astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    soft = cv2.GaussianBlur(measured.astype(np.float32), (0, 0), 2)
    chroma = 0.38 + 0.62 * soft
    lab[..., 0] = np.clip(lab[..., 0] * (0.72 + 0.16 * soft) + strength * detail, 0, 100)
    lab[..., 1] *= chroma
    lab[..., 2] *= chroma
    return (np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), 0, 1) * 255).astype(np.uint8)


def label(img, text, scale=0.5):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(img, (5, 5), (13 + w, 12 + h + base), (0, 0, 0), -1)
    cv2.putText(img, text, (9, 9 + h), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def composite(rgb, reference, raw, metric, zmin, zmax, show_max):
    """The large DepthEtch view with colour, depth camera and raw model panels beside it."""
    valid = np.isfinite(reference) & (reference >= zmin) & (reference <= zmax)
    measured = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    big = cv2.resize(depthetch(rgb, metric, measured, zmin, show_max), BIG, interpolation=cv2.INTER_AREA)
    a = cv2.resize(rgb[..., ::-1], SMALL, interpolation=cv2.INTER_AREA)
    b = cv2.resize(depth_color(reference, zmin, show_max), SMALL, interpolation=cv2.INTER_NEAREST)
    c = cv2.resize(depth_color(raw, zmin, show_max), SMALL, interpolation=cv2.INTER_AREA)
    label(big, "DepthEtch: colour = metres, lines = RGB + Depth Pro 2, muted = filled in", 0.58)
    label(a, "Colour")
    label(b, "RealSense")
    label(c, "Depth Pro 2 H+ raw")
    return np.hstack([big, np.vstack([a, b, c])])
