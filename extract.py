#!/usr/bin/env python3
"""Read the MCAP chunks of one RealSense recording in order; save each colour frame with its nearest emitter-on depth frame."""
import collections, concurrent.futures as cf, csv, json, struct, sys, time
from pathlib import Path

import cv2
import numpy as np
from mcap.reader import NonSeekingReader
from mcap_ros2.decoder import DecoderFactory

OUT = Path(sys.argv[1])
INPUTS = sys.argv[2:]
if not INPUTS:
    raise SystemExit("usage: extract.py OUT chunk0.mcap [chunk1.mcap ...]")
MAX_DT = 0.020  # s; emitter-on depth arrives every 33 ms, so the nearest is within ~17 ms
TOPICS = ["/tf_static", "/camera/color/camera_info", "/camera/depth/camera_info", "/camera/depth/metadata",
          "/camera/color/metadata", "/camera/color/image_raw", "/camera/depth/image_rect_raw"]
(OUT / "rgb").mkdir(parents=True, exist_ok=True)
(OUT / "depth").mkdir(parents=True, exist_ok=True)
factory, decoders = DecoderFactory(), {}


def decode(schema, data):
    d = decoders.get(schema.id)
    if d is None:
        d = decoders[schema.id] = factory.decoder_for("cdr", schema)
    return d(data)


def image(b):
    align = lambda o, n: o + ((n - (o - 4) % n) % n)
    sec, nsec = struct.unpack_from("<iI", b, 4)
    o = 12
    n = struct.unpack_from("<I", b, o)[0]
    o = align(o + 4 + n, 4)
    h, w = struct.unpack_from("<II", b, o)
    o += 8
    n = struct.unpack_from("<I", b, o)[0]
    enc = bytes(b[o + 4:o + 3 + n]).decode()
    o = align(o + 4 + n + 1, 4) + 4
    size = struct.unpack_from("<I", b, o)[0]
    o += 4
    return round(sec * 1_000_000 + nsec / 1000), h, w, enc, bytes(b[o:o + size])


def write(i, rgb, depth):
    r = np.frombuffer(rgb[4], np.uint8).reshape(rgb[1], rgb[2], 3)
    d = np.frombuffer(depth[4], np.uint16).reshape(depth[1], depth[2])
    ok1 = cv2.imwrite(str(OUT / "rgb" / f"{i:06d}.png"), r[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 1])
    ok2 = cv2.imwrite(str(OUT / "depth" / f"{i:06d}.png"), d, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not (ok1 and ok2):
        raise IOError(f"failed to write frame {i}")


meta, colors, depths, dmeta, cmeta = {}, collections.OrderedDict(), collections.OrderedDict(), {}, {}
rows, skipped, jobs = [], collections.Counter(), []
pool = cf.ThreadPoolExecutor(8)
t0 = time.time()


def pair(final=False):
    newest = max(depths) if depths else 0
    for ct in list(colors):
        if not final and newest < ct + 80_000:
            break
        rgb = colors.pop(ct)
        # Flashing-emitter recordings report the mode per frame. A camera that omits it has a steady emitter.
        on = [t for t in depths if t in dmeta and dmeta[t].get("frame_emitter_mode", 1) == 1]
        if not on:
            skipped["no_emitter_on_depth"] += 1
            continue
        dt = min(on, key=lambda t: abs(t - ct))
        if abs(dt - ct) > MAX_DT * 1e6:
            skipped["too_far_in_time"] += 1
            continue
        i = len(rows)
        m, c = dmeta[dt], cmeta.get(ct, {})
        rows.append(dict(index=i, color_us=ct, depth_us=dt, dt_ms=round((dt - ct) / 1000, 3),
                         color_frame=c.get("frame_number"), depth_frame=m.get("frame_number"),
                         ir_exposure=m.get("actual_exposure"), ir_gain=m.get("gain_level"),
                         laser=m.get("frame_laser_power"), color_ae=c.get("auto_exposure")))
        jobs.append(pool.submit(write, i, rgb, depths[dt]))
        if len(jobs) >= 32:
            for j in jobs:
                j.result()
            jobs.clear()
        if i % 200 == 0:
            print(f"frame {i}  {time.time() - t0:.0f}s", flush=True)
    while len(depths) > 48:
        t, _ = depths.popitem(last=False)
        dmeta.pop(t, None)


def messages():
    """Every wanted message across all chunks. Schema ids restart in each file, so decoders do too."""
    for path in INPUTS:
        decoders.clear()
        with open(path, "rb") as fh:
            yield from NonSeekingReader(fh).iter_messages(topics=TOPICS, log_time_order=False)


try:
    for schema, ch, msg in messages():
        t = ch.topic
        if t == "/camera/color/image_raw":
            x = image(msg.data)
            colors[x[0]] = x
        elif t == "/camera/depth/image_rect_raw":
            x = image(msg.data)
            depths[x[0]] = x
            depths.move_to_end(x[0])
            pair()
        elif t.endswith("metadata"):
            j = json.loads(decode(schema, msg.data).json_data)
            (dmeta if "depth" in t else cmeta)[round(j["frame_timestamp"] * 1000)] = j
            while len(cmeta) > 64:
                cmeta.pop(next(iter(cmeta)))
        elif t == "/tf_static" and "tf" not in meta:
            meta["tf"] = [dict(parent=x.header.frame_id, child=x.child_frame_id,
                               t=[x.transform.translation.x, x.transform.translation.y, x.transform.translation.z],
                               q=[x.transform.rotation.x, x.transform.rotation.y, x.transform.rotation.z, x.transform.rotation.w])
                          for x in decode(schema, msg.data).transforms]
        elif t.endswith("camera_info") and t not in meta:
            m = decode(schema, msg.data)
            meta[t] = dict(width=m.width, height=m.height, model=m.distortion_model, d=list(m.d), k=list(m.k), frame=m.header.frame_id)
    pair(final=True)
finally:
    for j in jobs:
        j.result()
    pool.shutdown()
    if rows:
        with open(OUT / "index.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    meta["skipped"] = dict(skipped)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
dts = np.abs([r["dt_ms"] for r in rows])
print(f"saved {len(rows)} pairs, skipped {dict(skipped)}, |dt| median {np.median(dts):.2f} ms, max {dts.max():.2f} ms, {time.time() - t0:.0f}s", flush=True)
