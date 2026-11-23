#!/usr/bin/env python3
"""Render a DepthEtch video from extracted frames, Depth Pro 2 predictions and a correction."""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from depthetch.camera import Frames
from depthetch.render import BIG, SMALL, composite, load_correction, stabilised_metric


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frames", help="folder made by extract.py")
    ap.add_argument("prediction", help="folder made by predict.py")
    ap.add_argument("correction", help="correction.npz made by stabilise.py")
    ap.add_argument("output", help="output .mp4")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--show-max", type=float, default=12, help="far end of the colour scale in metres")
    ap.add_argument("--crf", default="19")
    ap.add_argument("--maxrate", default="12M")
    args = ap.parse_args()

    run = Path(args.prediction)
    summary = json.loads((run / "summary.json").read_text())
    fpx, (zmin, zmax) = summary["f_px"], summary["depth_range_m"]
    correction = load_correction(args.correction, fpx)
    frames = Frames(args.frames, args.limit, args.start)
    w, h = BIG[0] + SMALL[0], BIG[1]
    ffmpeg = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(args.fps),
                               "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", args.crf, "-maxrate", args.maxrate, "-bufsize", "16M",
                               "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.output], stdin=subprocess.PIPE)
    try:
        for n, (i, rgb, reference) in enumerate(frames):
            inv_half = np.load(run / "pred" / f"{i:06d}.npy").astype(np.float32)
            inv = cv2.resize(inv_half, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_LINEAR)
            with np.errstate(divide="ignore", invalid="ignore"):
                raw = fpx / (1280 * inv)
                metric = stabilised_metric(inv, fpx, correction, i)
            metric[~np.isfinite(metric) | (metric <= 0)] = args.show_max
            ffmpeg.stdin.write(composite(rgb, reference, raw, metric, zmin, zmax, args.show_max).tobytes())
            if n % 150 == 0:
                print(f"{n}/{len(frames.rows)}", flush=True)
    finally:
        ffmpeg.stdin.close()
        code = ffmpeg.wait()
    if code:
        raise SystemExit(f"ffmpeg failed: {code}")
    print("wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
