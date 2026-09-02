#!/usr/bin/env python3
"""Render one trained IRGS camera with identical settings on black/white backgrounds."""

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image

IRGS_ROOT = Path(__file__).resolve().parents[1]
if str(IRGS_ROOT) not in sys.path:
    sys.path.insert(0, str(IRGS_ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render_ir
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


def font(size):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    return ImageFont.truetype(path, size) if Path(path).is_file() else None


def labeled_grid(items, title):
    images = [(label, to_pil_image(tensor.detach().cpu().clamp(0, 1)))
              for label, tensor in items]
    width, height = images[0][1].size
    header, label_height = 52, 34
    canvas = Image.new("RGB", (width * len(images), header + label_height + height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((canvas.width // 2, header // 2), title, fill="black", font=font(22), anchor="mm")
    for index, (label, image) in enumerate(images):
        x = index * width
        draw.text((x + width // 2, header + label_height // 2), label,
                  fill="black", font=font(17), anchor="mm")
        canvas.paste(image, (x, header + label_height))
    return canvas


def main():
    parser = ArgumentParser(description=__doc__)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=20000)
    parser.add_argument("--overview_slot", type=int, choices=range(4), default=0,
                        help="Row 0..3 from train.py's four-view overview selection")
    parser.add_argument("--output", required=True)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    safe_state(args.quiet)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    # train.py creates Scene with shuffle=True, so reproduce the overview row ordering.
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=True)
    gaussians.build_bvh()

    cameras = scene.getTrainCameras()
    indices = np.linspace(0, len(cameras) - 1, min(4, len(cameras)), dtype=int)
    camera_index = int(indices[min(args.overview_slot, len(indices) - 1)])
    camera = cameras[camera_index]

    common = {
        "viewpoint_camera": camera,
        "pc": gaussians,
        "pipe": pipe,
        "training": False,
        "relight": False,
    }
    with torch.no_grad():
        black_package = render_ir(
            **common, bg_color=torch.zeros(3, dtype=torch.float32, device="cuda"))
        white_package = render_ir(
            **common, bg_color=torch.ones(3, dtype=torch.float32, device="cuda"))

    black = black_package["render"].clamp(0, 1)
    white = white_package["render"].clamp(0, 1)
    alpha_black = black_package["rend_alpha"].clamp(0, 1)
    alpha_white = white_package["rend_alpha"].clamp(0, 1)
    alpha_rgb = alpha_black.expand(3, -1, -1)
    gt = camera.original_image.cuda().clamp(0, 1)

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    to_pil_image(black.cpu()).save(output / "render_black.png")
    to_pil_image(white.cpu()).save(output / "render_white.png")
    to_pil_image(alpha_rgb.cpu()).save(output / "render_alpha.png")
    to_pil_image(gt.cpu()).save(output / "gt_input.png")
    labeled_grid(
        [("GT input", gt), ("PBR / black BG", black),
         ("PBR / white BG", white), ("render alpha", alpha_rgb)],
        f"{camera.image_name} | IRGS iteration {scene.loaded_iter} | same camera/checkpoint",
    ).save(output / "comparison.png")

    expected_delta = (1.0 - alpha_black).expand_as(black)
    actual_delta = white - black
    sam_mask = (camera.mask.float().cuda() if camera.mask is not None
                else torch.ones_like(alpha_black))
    alpha_binary = alpha_black > 0.5
    sam_binary = sam_mask > 0.5
    intersection = (alpha_binary & sam_binary).sum().item()
    union = (alpha_binary | sam_binary).sum().item()
    report = {
        "model": str(Path(dataset.model_path).resolve()),
        "iteration": int(scene.loaded_iter),
        "overview_slot": int(args.overview_slot),
        "camera_list_index_after_training_shuffle": camera_index,
        "camera": camera.image_name,
        "resolution": [int(camera.image_width), int(camera.image_height)],
        "only_changed_parameter": "bg_color: [0,0,0] vs [1,1,1]",
        "alpha_max_abs_difference_between_runs": float(
            (alpha_white - alpha_black).abs().max().item()),
        "white_minus_black_vs_one_minus_alpha_mae": float(
            (actual_delta - expected_delta).abs().mean().item()),
        "sam_vs_alpha_threshold_0p5_iou": float(intersection / union) if union else 1.0,
        "sam_foreground_pixels": int(sam_binary.sum().item()),
        "alpha_gt_0p5_pixels": int(alpha_binary.sum().item()),
    }
    with (output / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
