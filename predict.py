#!/usr/bin/env python3
"""Run Depth Pro 2 on extracted frames and save half-size inverse depth for stabilise.py and render.py."""
import argparse
import collections
import json
from pathlib import Path

import cv2
import numpy as np

from depthetch.camera import Frames
from depthetch.profiles import PROFILES


def predictions(model, transform, frames, window, overlap, f_px):
    pending = collections.deque()
    stream = model.stream(window=window, overlap=overlap, f_px=f_px, mode="buffered")
    for i, rgb, _ in frames:
        pending.append(i)
        for p in stream.step(transform(rgb)):
            yield pending.popleft(), p["inverse_depth"]
    for p in stream.flush():
        yield pending.popleft(), p["inverse_depth"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frames", help="folder made by extract.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="depthpro2-h", choices=["depthpro2-s", "depthpro2-b", "depthpro2-l", "depthpro2-h"])
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--overlap", type=int, default=4)
    ap.add_argument("--camera", default="d456", choices=sorted(PROFILES), help="sets the trusted depth range")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    import torch
    from depth_pro_2 import create_model_and_transforms

    frames = Frames(args.frames, args.limit)
    out = Path(args.out)
    (out / "pred").mkdir(parents=True, exist_ok=True)
    model, transform = create_model_and_transforms(args.model, device=args.device)
    f_px = float(frames.Kc[0])
    count = 0
    with torch.inference_mode():
        for i, inv in predictions(model, transform, frames, args.window, args.overlap, f_px):
            inv = inv.float().cpu().numpy()
            small = cv2.resize(inv, (inv.shape[1] // 2, inv.shape[0] // 2), interpolation=cv2.INTER_AREA)
            np.save(out / "pred" / f"{i:06d}.npy", small.astype(np.float16))
            count += 1
            if count % 100 == 0:
                print(f"{count}/{len(frames.rows)}", flush=True)
    profile = PROFILES[args.camera]
    summary = dict(model=args.model, mode="buffered", window=args.window, overlap=args.overlap, frames=count, f_px=f_px,
                   camera=args.camera, depth_range_m=[profile["min_depth"], profile["max_depth"]])
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"saved {count} predictions", flush=True)


if __name__ == "__main__":
    main()
