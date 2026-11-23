# AGENTS.md

Instructions for people and agents working on DepthEtch, including adding and calibrating a camera.

## Ground rules

- A depth camera is a **reference, not ground truth**. Use a laser distance meter when a claim depends on absolute accuracy.
- Never present filled-in depth as measured depth. DepthEtch mutes areas the camera did not measure.
- Do not update camera firmware, run on-chip calibration, or write calibration to a camera without the owner's explicit approval.
- Keep recordings private by default. Public material must not show people, homes, vehicles, documents, locations, hostnames, personal paths or serial numbers.
- Do not commit weights, ONNX files, recordings or extracted frames.
- Depth Pro 2 weights are for non-commercial research under their licence, and DINOv3 terms also apply. Do not use Apple's name to promote this work.

## What changes per camera

| Part | Per camera? | Notes |
|---|---|---|
| Depth Pro 2 | No | The focal length comes from the recording. |
| DepthEtch and the correction | No | They use the profile's trusted depth range. |
| `depthetch/profiles.py` | **Yes** | One entry per camera model. |
| Frame extraction | Only for non-RealSense cameras | See "Cameras from other vendors". |
| Calibration results | **Yes, per physical unit** | Two units of one model can differ. |

## Add a camera

### 1. Identify it

With the camera on a USB 3 port and no other process using it:

```bash
lsusb
rs-enumerate-devices -s
rs-enumerate-devices -c
```

Record the USB product ID, USB speed, firmware, supported colour and depth modes, whether it has an IMU, and the factory lens model and coefficients. Read the shutter type, stereo baseline, minimum depth and ideal range from the current datasheet. If sources disagree, note it and let the laser check decide.

A USB 2 link usually means a bad cable or a partly inserted plug. Fix that first.

### 2. Add a profile

Add one entry to `PROFILES` in `depthetch/profiles.py`:

```python
"d4xx": dict(
    label="RealSense D4xx",
    usb_products=("0xxx",),      # lowercase USB product ID
    color="1280x720x30",         # WIDTHxHEIGHTxFPS
    depth="848x480x30",          # the model's recommended depth mode
    imu=False,
    min_depth=0.5,               # trusted range, in metres
    max_depth=3.0,
    shutter="rolling",           # or "global"
    range_source="datasheet",    # becomes "laser YYYY-MM-DD" after calibration
),
```

Then confirm detection. This does not start the camera:

```bash
python -m depthetch.profiles list
python -m depthetch.profiles detect
```

### 3. Test a recording

Record at least 30 seconds and extract it with `extract.py`. **Pass means no missing colour or depth frames** and a write rate the disk sustains. If frames drop, lower the depth resolution or frame rate in the profile.

## Calibrate a camera

Do these in order, for each physical unit. Let the camera warm up for about 10 minutes first, and do not publish its serial number.

### 1. Lens

- Use the factory intrinsics and distortion from the camera.
- For RealSense, build undistortion to match `rs2_project_point_to_pixel`. Do not assume the sign or direction of the coefficients: applying them the wrong way produced errors of up to 19 pixels on a D456.
- Confirm that straight edges such as door frames look straight near the image corners after undistortion.
- For a camera without trustworthy factory data, calibrate with a checkerboard from at least 20 varied views. Aim for an RMS reprojection error below 0.5 px.

### 2. Depth-to-colour alignment

Overlay depth edges on the undistorted colour image at a near and a far distance. Edges should agree within about 2 pixels. If they don't, stop and report it rather than compensating in code.

### 3. Depth accuracy with a laser meter

1. Use a flat, matte, well-textured wall. Keep the camera within about 3 degrees of square to it.
2. Measure at least 5 distances, from the datasheet minimum to beyond the ideal maximum. For example: 0.5, 0.75, 1, 1.5, 2, 3 and 4 m.
3. Measure to the camera's **depth origin**, not its front glass. Add the model's offset from the datasheet.
4. Take three depth captures per distance. Fit a plane to the central area and record the signed error, plane-fit RMS, fill rate and tilt.

### 4. Set the trusted range

The trusted range is the longest continuous span where **absolute error is at most 2%, plane RMS is at most 1% of distance, and fill rate is at least 80%**. Update `min_depth`, `max_depth` and `range_source`. Do not silently apply a correction to the camera's depth. If one is needed, propose it with the evidence.

### 5. Check it with DepthEtch

Record separate still and moving clips, run `predict.py` with `--camera <key>`, then `stabilise.py`, and report its hidden-block scores. Tune settings on different recordings from the ones you report. Inspect outliers before excluding them, and state any exclusion. A rolling-shutter camera should also be checked with slow and fast motion.

## Cameras from other vendors

Write an extractor that produces the same folder as `extract.py`:

- `rgb/000000.png`: 1280x720 colour.
- `depth/000000.png`: 16-bit depth in **millimetres**, with 0 for no data.
- `index.csv`: at least an `index` column.
- `meta.json`: colour and depth `camera_info` (`width`, `height`, `k`, `d`, `frame`) and the static transforms between them (`tf`).

Add the vendor ID to `connected()` in `depthetch/profiles.py`, then follow the calibration steps above.

## Definition of done

- The profile is detected, and existing profiles are unchanged.
- A test recording has no missing frames.
- Lens, alignment and laser results are recorded for the physical unit.
- The trusted range comes from laser measurements, or is clearly marked `datasheet`.
- Anything not verified is stated plainly.
