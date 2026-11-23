#!/usr/bin/env python3
"""Solve one smooth whole-clip correction of Depth Pro 2 against the depth camera."""
import argparse
import json
from pathlib import Path

import numpy as np

from depthetch.correction import hidden_block_report, load_clip, solve


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frames", help="folder made by extract.py")
    ap.add_argument("prediction", help="folder made by predict.py")
    ap.add_argument("--out", required=True, help="output correction.npz")
    ap.add_argument("--tau", type=float, default=2, help="temporal smoothness of the global fit")
    ap.add_argument("--sigma", type=float, default=4, help="spatial softness of the local field, in grid cells")
    ap.add_argument("--alpha", type=float, default=1.0, help="newest-frame weight of the local field; 1 means no temporal filter")
    ap.add_argument("--seed", type=int, default=0, help="chooses the hidden test blocks and sampled pixels")
    ap.add_argument("--limit", type=int, help="use only the first N frames")
    ap.add_argument("--report", help="write the hidden-block scores to this JSON file")
    args = ap.parse_args()

    clip = load_clip(args.frames, args.prediction, args.seed, args.limit)
    print(f"{clip['T']} frames, {100 * clip['hold_share']:.0f}% of the image hidden for scoring", flush=True)
    scores = hidden_block_report(clip, args.tau, args.sigma, args.alpha)
    for name, s in scores.items():
        print(f"  {name:28s} {s['mean_rel'] * 100:5.1f}%  {s['median_cm']:5.1f} cm  {s['within_5pct'] * 100:3.0f}% within 5%", flush=True)
    if args.report:
        Path(args.report).write_text(json.dumps(dict(settings=dict(tau=args.tau, sigma=args.sigma, alpha=args.alpha), hidden_blocks=scores), indent=1))

    r_far = 1 / clip["zmax"]
    a, b, field = solve(clip["all"], args.tau, args.sigma, args.alpha, r_far)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, a=a.astype(np.float32), b=b.astype(np.float32), field=field.astype(np.float16),
                        frame_index=np.array([int(r["index"]) for r in clip["frames"].rows], np.int32), r_far=r_far, fx=clip["fx"])
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
