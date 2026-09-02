#!/usr/bin/env python3
"""Render one IRGS checkpoint with a finite inverse-square point light."""

import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import render_ir  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from scene.light import PointLight  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402


def main():
    parser = ArgumentParser(description=__doc__)
    model_args = ModelParams(parser, sentinel=True)
    pipeline_args = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=20000)
    parser.add_argument("--view-id", type=int, default=-1)
    parser.add_argument("--light-position", type=float, nargs=3, required=True,
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--light-intensity", type=float, default=10.0)
    parser.add_argument("--light-color", type=float, nargs=3,
                        default=(1.0, 1.0, 1.0))
    parser.add_argument("--background", choices=("black", "white"),
                        default="white")
    parser.add_argument("--output", required=True)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    if args.light_intensity < 0 or any(x < 0 for x in args.light_color):
        parser.error("light intensity and color must be non-negative")

    safe_state(args.quiet)
    dataset = model_args.extract(args)
    pipe = pipeline_args.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    gaussians.build_bvh()
    cameras = list(scene.getTrainCameras()) + list(scene.getTestCameras())
    cameras.sort(key=lambda camera: str(camera.image_name))
    if not cameras:
        raise RuntimeError("No cameras found")
    view_index = len(cameras) // 2 if args.view_id < 0 else args.view_id % len(cameras)
    camera = cameras[view_index]

    gaussians.env_map = None
    gaussians.direct_light = PointLight(
        position=args.light_position,
        intensity=args.light_intensity * np.asarray(args.light_color),
        device="cuda",
    ).cuda()
    pipe.wo_indirect_relight = True
    background_value = 1.0 if args.background == "white" else 0.0
    background = torch.full(
        (3,), background_value, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        package = render_ir(
            camera, gaussians, pipe, background, training=False, relight=True,
            base_color_scale=torch.ones(3, dtype=torch.float32, device="cuda"),
        )
    torch.cuda.synchronize()

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    to_pil_image(package["render"].clamp(0, 1).cpu()).save(output / "render.png")
    to_pil_image(package["visibility"].clamp(0, 1).cpu()).save(
        output / "visibility.png")
    report = {
        "model_path": str(Path(dataset.model_path).resolve()),
        "iteration": int(scene.loaded_iter),
        "camera": str(camera.image_name),
        "camera_index": view_index,
        "resolution": [int(camera.image_width), int(camera.image_height)],
        "light": {
            "type": "finite_point",
            "position_scene_units": list(args.light_position),
            "intensity_linear_rgb": (
                args.light_intensity * np.asarray(args.light_color)).tolist(),
            "distance_attenuation": "inverse_square",
            "shadow_max_distance": "finite segment to emitter",
        },
        "background": args.background,
        "diffuse_samples_saved_but_unused_for_delta_light": int(
            pipe.diffuse_sample_num),
        "analytic_samples_per_pixel": 1,
        "output": str(output / "render.png"),
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
