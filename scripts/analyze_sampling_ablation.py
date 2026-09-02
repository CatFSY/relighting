#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    variants = [
        "baseline_ds128_ls256_iid",
        "stratified_ds64_ls64",
        "stratified_ds32_ls32",
    ]
    images = {
        name: np.asarray(Image.open(
            args.root / name / "preview_frames" / "frame_00000.png"
        ).convert("RGB"),
                         dtype=np.float32) / 255.0
        for name in variants
    }
    reference = images[variants[0]][28:]
    metrics = {}
    for name in variants:
        candidate = images[name][28:]
        error = candidate - reference
        mse = float(np.mean(np.square(error)))
        metrics[name] = {
            "mae": float(np.mean(np.abs(error))),
            "rmse": math.sqrt(mse),
            "psnr_db_vs_baseline": (
                float("inf") if mse == 0 else -10.0 * math.log10(mse)
            ),
            "pixels_any_channel_abs_error_gt_0p05": int(
                np.any(np.abs(error) > 0.05, axis=-1).sum()
            ),
        }

    report = {"reference": variants[0], "label_rows_excluded": 28,
              "metrics": metrics}
    (args.root / "sampling_comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=True) + "\n")

    panels = []
    for name in variants:
        panel = Image.fromarray((images[name] * 255).astype(np.uint8))
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 30, 340, 56), fill="white")
        draw.text((8, 37), name, fill="black")
        panels.append(panel)
    width = sum(panel.width for panel in panels)
    montage = Image.new("RGB", (width, panels[0].height), "white")
    x = 0
    for panel in panels:
        montage.paste(panel, (x, 0))
        x += panel.width
    montage.save(args.root / "sampling_comparison.png")


if __name__ == "__main__":
    main()
