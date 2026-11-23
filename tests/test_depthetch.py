import cv2
import numpy as np

from depthetch.camera import align_depth, undistort_maps
from depthetch.correction import CELL, GRID_H, GRID_W, Points, field_from, sample_field, smooth_solve, to_inverse
from depthetch.profiles import PROFILES, for_product_id
from depthetch.render import composite, depthetch, stabilised_metric


def synthetic_points(a, b, rng, per_frame=3000):
    parts = []
    for t in range(len(a)):
        c = rng.uniform(0.17, 1.6, per_frame).astype(np.float32)
        r = (a[t] * c + b[t]).astype(np.float32)
        u = rng.integers(0, 640, per_frame).astype(np.int16)
        v = rng.integers(0, 360, per_frame).astype(np.int16)
        parts.append([u, v, c, r])
    return Points(parts, len(a))


def test_inverse_map_is_affine_in_range_and_positive_beyond_it():
    c = np.geomspace(1e-3, 2, 400)
    r = to_inverse(c, 0.8, 0.1, 1 / 6)
    inside = r >= 1 / 6
    assert np.allclose(r[inside], 0.8 * c[inside] + 0.1)
    assert np.all(r > 0)
    assert np.all(np.diff(r) > 0)


def test_solver_recovers_a_slowly_changing_correction():
    frames = 40
    a = 0.8 + 0.05 * np.sin(np.linspace(0, 3, frames))
    b = 0.05 + 0.02 * np.cos(np.linspace(0, 3, frames))
    p = synthetic_points(a, b, np.random.default_rng(1))
    found_a, found_b = smooth_solve(p, 2, np.full(frames, 1.0), np.zeros(frames))
    # Smoothing has neighbours on one side only at the ends of a clip, so they get a looser limit.
    for found, true in ((found_a, a), (found_b, b)):
        error = np.abs(found - true)
        assert np.max(error[5:-5]) < 0.01
        assert np.max(error) < 0.02


def test_local_field_recovers_a_constant_offset():
    a, b = np.full(3, 1.0), np.zeros(3)
    p = synthetic_points(a, b, np.random.default_rng(2), per_frame=60000)
    offset = np.log(1.05)
    field = field_from(p, p.r / np.exp(offset), sigma=4, alpha=1.0)
    assert field.shape == (3, GRID_H, GRID_W)
    sampled = sample_field(field, p.idx, p.u, p.v)
    assert abs(float(np.median(sampled)) - offset) < 0.01
    assert CELL * GRID_W == 640


def test_stabilised_metric_applies_scale_and_field():
    correction = dict(a=np.array([2.0]), b=np.array([0.0]), r_far=0.01, field=np.zeros((1, GRID_H, GRID_W), np.float32), frame={7: 0})
    fpx = 640.0
    inv = np.full((720, 1280), 1.0, np.float32)
    assert np.allclose(stabilised_metric(inv, fpx, correction, 7), 0.25)
    correction["field"][0] = np.log(2)
    assert np.allclose(stabilised_metric(inv, fpx, correction, 7), 0.125)


def test_alignment_keeps_a_flat_wall_at_the_same_distance():
    K = [640.0, 0, 640.0, 0, 640.0, 360.0, 0, 0, 1]
    depth = np.full((720, 1280), 2000, np.uint16)
    z = align_depth(depth, K, K, np.eye(3), np.zeros(3), 1280, 720)
    assert np.nanmax(np.abs(z - 2.0)) < 1e-5
    assert np.isfinite(z).mean() > 0.99


def test_zero_distortion_is_the_identity():
    K = [640.0, 0, 64.0, 0, 640.0, 36.0, 0, 0, 1]
    map1, map2 = undistort_maps(K, [0, 0, 0, 0, 0], 128, 72)
    u, v = np.meshgrid(np.arange(128), np.arange(72))
    assert np.allclose(map1, u) and np.allclose(map2, v)


def test_unmeasured_areas_are_muted():
    rgb = np.full((180, 320, 3), 128, np.uint8)
    metric = np.full((180, 320), 2.0, np.float32)
    measured = np.zeros((180, 320), bool)
    measured[:, :160] = True
    lab = cv2.cvtColor(depthetch(rgb, metric, measured, 0.6, 12), cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    assert chroma[:, :120].mean() > 1.5 * chroma[:, 200:].mean()


def test_composite_size():
    rgb = np.full((720, 1280, 3), 100, np.uint8)
    depth = np.full((720, 1280), 2.0, np.float32)
    assert composite(rgb, depth, depth, depth, 0.6, 6.0, 12).shape == (540, 1280, 3)


def test_profiles_are_complete():
    assert for_product_id("0x0B5C")[0] == "d456"
    assert for_product_id("ffff") == (None, None)
    for profile in PROFILES.values():
        assert profile["min_depth"] < profile["max_depth"]
