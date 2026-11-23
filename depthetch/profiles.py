"""Per-camera settings. To add a camera, add one entry to PROFILES and follow AGENTS.md.

    python -m depthetch.profiles list      known cameras
    python -m depthetch.profiles detect    key of the connected camera (Linux)
"""
import sys
from pathlib import Path

# min_depth and max_depth are the range where this camera is trusted as a depth reference.
# Replace datasheet values with laser-checked values when a unit has been calibrated.
PROFILES = {
    "d456": dict(
        label="RealSense D455/D456",
        usb_products=("0b5c",),
        color="1280x720x30",
        depth="848x480x30",
        imu=True,
        min_depth=0.6,
        max_depth=6.0,
        shutter="global",
        range_source="datasheet",
    ),
    "d415": dict(
        label="RealSense D415",
        usb_products=("0ad3",),
        color="1280x720x30",
        depth="848x480x30",  # 1280x720x30 dropped depth frames in testing
        imu=False,
        min_depth=0.5,
        max_depth=3.0,
        shutter="rolling",
        range_source="datasheet",
    ),
}


def for_product_id(product_id):
    """The (key, profile) for a USB product id, or (None, None) if the camera is not known."""
    wanted = str(product_id or "").lower().removeprefix("0x")
    for key, profile in PROFILES.items():
        if wanted in profile["usb_products"]:
            return key, profile
    return None, None


def connected():
    """Known Intel cameras on USB, as (key, speed) pairs. Reads sysfs, so it never opens a camera."""
    found = []
    for device in sorted(Path("/sys/bus/usb/devices").glob("*")):
        try:
            if (device / "idVendor").read_text().strip() != "8086":
                continue
            key, _ = for_product_id((device / "idProduct").read_text().strip())
            if key:
                found.append((key, (device / "speed").read_text().strip() + "M"))
        except OSError:
            continue
    return found


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "list":
        for key, p in PROFILES.items():
            print(f"{key}: {p['label']}, colour {p['color']}, depth {p['depth']}, IMU {'yes' if p['imu'] else 'no'}, "
                  f"trusted {p['min_depth']:g}-{p['max_depth']:g} m ({p['range_source']}), {p['shutter']} shutter")
    elif command == "detect":
        found = connected()
        if len(found) != 1:
            raise SystemExit(f"expected one known camera, found {len(found)}")
        print(found[0][0])
    else:
        print(__doc__.strip())
        raise SystemExit(2)


if __name__ == "__main__":
    main()
