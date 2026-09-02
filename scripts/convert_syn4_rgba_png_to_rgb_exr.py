#!/usr/bin/env python3
"""Convert Synthetic4Relight train_NNN/rgba.png into train/NNN_rgb.exr.

The Blender PNG is 16-bit RGBA. RGB is normalized directly to float32 [0, 1]
without tone mapping or an additional transfer-function conversion. Alpha stays
in the separately stored mask and is not multiplied into RGB.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", type=Path)
    args = parser.parse_args()

    scene = args.scene.resolve()
    output_dir = scene / "train"
    sources = sorted(scene.glob("train_[0-9][0-9][0-9]/rgba.png"))
    if not sources:
        raise RuntimeError(f"No train_NNN/rgba.png files found below {scene}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for expected, source in enumerate(sources):
        index = int(source.parent.name.removeprefix("train_"))
        if index != expected:
            raise RuntimeError(f"Non-contiguous view mapping: expected {expected:03d}, got {index:03d}")

        rgba = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.dtype != np.uint16 or rgba.ndim != 3 or rgba.shape[2] != 4:
            raise RuntimeError(f"Expected 16-bit RGBA PNG: {source}")

        # OpenCV uses BGRA/BGR consistently, so channel order remains correct
        # when the float EXR is written through OpenCV.
        rgb = rgba[:, :, :3].astype(np.float32) / np.float32(65535.0)
        destination = output_dir / f"{index:03d}_rgb.exr"
        temporary = output_dir / f".{index:03d}_rgb.tmp.exr"
        if not cv2.imwrite(str(temporary), rgb):
            raise RuntimeError(f"Could not write {temporary}")
        os.replace(temporary, destination)

        check = cv2.imread(str(destination), cv2.IMREAD_UNCHANGED)
        max_error = float(np.max(np.abs(check - rgb)))
        if max_error != 0.0:
            raise RuntimeError(f"EXR verification failed for {destination}: max error {max_error}")

    print(f"Converted and verified {len(sources)} views: {scene}/train_NNN/rgba.png -> {output_dir}/*_rgb.exr")


if __name__ == "__main__":
    main()
