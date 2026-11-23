# DepthEtch

A depth video that people can read at a glance. DepthEtch combines three sources:

- a **depth camera** for real distances
- **Depth Pro 2** for complete shape
- the **colour image** for fine detail

Colour shows metres, from red (near) to blue (far). Etched lines come from the image and the model.
Vivid areas were measured by the depth camera. **Muted areas were filled in**, so estimated depth is
never presented as measured depth.

This is an independent project. It is not affiliated with, endorsed by, or supported by Apple, Meta
or Intel. It contains no model weights and no upstream source code.

> **Depth from this tool is an estimate, not a calibrated measurement.** Do not use it for safety,
> navigation, medical, legal or metrology purposes.

## How it works

Depth Pro 2 produces sharp, complete depth, but its absolute scale drifts with the scene and distance.
A depth camera has reliable scale in its working range, but leaves holes and rough edges. DepthEtch
processes the whole clip offline:

1. Fit one smooth correction across all frames, `1/z = a(t)/d + b(t)`, using past and future frames.
2. Add a smooth local correction for what the global fit misses.
3. Beyond the camera's trusted range, switch to a pure scale so far pixels and sky stay stable.
4. Render metric colour, etch in image and depth detail, and mute everything the camera did not measure.

No model is trained or fine-tuned.

## Results

Two outdoor clips recorded with a RealSense D456, using the Depth Pro 2 H+ model. Each score uses image
blocks hidden from the correction for the entire clip, about 35% of the image. Settings were chosen
on two other clips from the same session.

| Clip | Method | Mean error | Median error | Within 5% |
|---|---|---:|---:|---:|
| 2,499 frames | Depth Pro 2 as-is | 15.4% | 35.8 cm | 24% |
| | Depth camera, nearest fill | 5.9% | 4.0 cm | 72% |
| | **DepthEtch correction** | **3.4%** | **4.0 cm** | **86%** |
| 928 frames | Depth Pro 2 as-is | 20.2% | 43.3 cm | 11% |
| | Depth camera, nearest fill | 4.5% | 2.4 cm | 81% |
| | **DepthEtch correction** | **2.4%** | **2.3 cm** | **93%** |

**These are errors against the RealSense camera, which is a reference and not ground truth.** They have
not been checked with a laser distance meter, and two clips from one session cannot show how the method
behaves in general.

## Setup

Tested on Linux with Python 3.12, an NVIDIA GPU, and Depth Pro 2 at upstream commit `aa389cb`.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`extract.py` also needs `mcap` and `mcap-ros2-support`. `predict.py` also needs PyTorch and the
Depth Pro 2 package from its upstream repository. On first use, upstream's loader downloads the weights.
Read upstream's `LICENSE_MODEL` first: downloading or using the weights means accepting it.

## Usage

Record colour, depth, camera info, metadata and static transforms from the RealSense ROS 2 driver to
MCAP. Colour must be 1280x720. Then:

```bash
python extract.py work/frames recording/*.mcap
python predict.py work/frames --out work/prediction --camera d456
python stabilise.py work/frames work/prediction --out work/correction.npz
python render.py work/frames work/prediction work/correction.npz work/depthetch.mp4
```

`stabilise.py` prints its accuracy on the hidden blocks. To use another camera, follow
[AGENTS.md](AGENTS.md).

## Limitations

- Processing is offline and needs the whole clip. H+ is far from real time.
- Only RealSense D456 footage has been evaluated. A D415 profile exists, but its range comes from the
  datasheet and it has not been evaluated in a real scene.
- Colour must be 1280x720 and the correction assumes that size.
- Areas the camera never measures rely on Depth Pro 2 and the fitted correction.
- CI uses synthetic data only. It does not run the camera, GPU or model.

## Licence and notices

**This project's own code and documentation** are under the MIT License in [`LICENSE`](LICENSE). That
licence does not cover Depth Pro 2 code or weights, DINOv3, TIMM, or anything derived from the weights.
Their own terms still apply.

**Depth Pro 2 code** is under Apple's licence in upstream's `LICENSE`. It is not copied here.

**Depth Pro 2 weights** are under the Apple Machine Learning Research Model License Agreement
(`LICENSE_MODEL` upstream), which limits use to non-commercial research and excludes commercial
exploitation, product development and use in commercial products or services. Those terms stay in force
for anyone using these scripts.

**Third-party terms.** Depth Pro 2 builds on DINOv3 (Meta) and TIMM, whose licence terms also apply.
Upstream's `ACKNOWLEDGEMENTS` file lists them in full.

**Names.** Apple, Depth Pro, Intel, RealSense and other names belong to their owners and are used only
to identify the software and hardware involved. Nothing here implies endorsement.

If you use the model in research, cite the Depth Pro 2 paper as described in the upstream README.
