#!/usr/bin/env python3
"""Render one common view for all loss ablations and make a labeled sheet."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid, save_image

from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel, render_ir
from scene import Scene
from utils.general_utils import safe_state
from utils.graphics_utils import rgb_to_srgb


def load_font(size: int, bold: bool = False):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def tensor_to_pil(x: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    x = x.detach().float().cpu().clamp(0.0, 1.0)
    if x.ndim == 2:
        x = x[None]
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    x = x.permute(1, 2, 0).numpy()
    return Image.fromarray((x * 255.0 + 0.5).astype(np.uint8), "RGB").resize(
        size, Image.Resampling.LANCZOS
    )


def masked_white(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    return x * mask + (1.0 - mask)


def psnr_from_log(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore")
    values = re.findall(r"\[ITER 20000\] Evaluating test set: PSNR ([0-9.eE+-]+)", text)
    return float(values[-1]) if values else None


@torch.no_grad()
def render_model(model_path: Path, dataset_path: Path, view_name: str, iteration: int):
    parser = argparse.ArgumentParser(add_help=False)
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    args = parser.parse_args(["-s", str(dataset_path), "-m", str(model_path)])
    # The ablation folders contain cfg_args, but this lightweight renderer
    # receives a common dataset path and should keep the original resolution.
    args.resolution = -1
    args.data_device = "cuda"
    args.sh_degree = 3
    args.mask_dir = str(dataset_path / "masks" / "paper_cup")
    args.pred_material_dir = str(dataset_path)
    dataset = model_params.extract(args)
    # PipelineParams.extract is available through the same parser namespace.
    pipe = pipeline_params.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    gaussians.build_bvh()
    gaussians.env_map.update_pdf()
    gaussians.env_map.set_transform(torch.tensor(
        [[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=torch.float32, device="cuda"
    ))
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device="cuda"
    )
    views = scene.getTrainCameras()
    view = next((v for v in views if v.image_name == view_name), None)
    if view is None:
        raise RuntimeError(f"view {view_name!r} not found in {model_path}")
    pkg = render_ir(view, gaussians, pipe, background)
    mask = view.mask.float().cuda() if view.mask is not None else torch.ones_like(pkg["rend_alpha"])
    gt_rgb = masked_white(view.original_image.cuda(), mask)
    pred_rgb = masked_white(pkg["render"], mask)
    gt_albedo = masked_white(rgb_to_srgb(view.gt_albedo.cuda()), mask)
    pred_albedo = masked_white(pkg["base_color"], mask)
    gt_rough = masked_white(view.gt_orm[1:2].cuda(), mask)
    pred_rough = masked_white(pkg["roughness"], mask)
    mask3 = mask.repeat(3, 1, 1)

    def mae(a, b):
        return float(((a - b).abs() * mask3).sum().item() / mask3.sum().clamp_min(1).item())

    metrics = {
        "rgb_mae": mae(gt_rgb, pred_rgb),
        "albedo_mae": mae(gt_albedo, pred_albedo),
        "roughness_mae": mae(gt_rough, pred_rough),
    }
    result = {
        "gt": [gt_rgb.cpu(), gt_albedo.cpu(), gt_rough.cpu()],
        "render": [pred_rgb.cpu(), pred_albedo.cpu(), pred_rough.cpu()],
        "metrics": metrics,
    }
    del scene, gaussians, pkg
    torch.cuda.empty_cache()
    return result


def save_sheet(rows, output: Path, title: str, thumb: tuple[int, int], label_w: int):
    tw, th = thumb
    header_h, row_h = 70, th + 36
    width = label_w + 3 * tw
    canvas = Image.new("RGB", (width, header_h + len(rows) * row_h), "#eeeeee")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(25, True)
    row_font = load_font(15)
    headers = [title, "RGB", "Albedo", "Roughness"]
    xs = [0, label_w, label_w + tw, label_w + 2 * tw]
    ws = [label_w, tw, tw, tw]
    for x, w, text in zip(xs, ws, headers):
        draw.rectangle((x, 0, x + w - 1, header_h - 1), fill="#263238")
        draw.text((x + 10, 20), text, fill="white", font=title_font)
    for i, row in enumerate(rows):
        y = header_h + i * row_h
        fill = "#ffffff" if i % 2 == 0 else "#e4e8eb"
        draw.rectangle((0, y, width, y + row_h - 1), fill=fill)
        draw.multiline_text((8, y + 8), row["label"], fill="#111111", font=row_font, spacing=3)
        for col, image in enumerate(row["images"]):
            x = label_w + col * tw
            pil = tensor_to_pil(image, (tw - 8, th - 8))
            canvas.paste(pil, (x + 4, y + 4))
            draw.rectangle((x, y, x + tw - 1, y + th - 1), outline="#9aa3a8")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95, subsampling=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view", default="frame_00000")
    parser.add_argument("--iteration", type=int, default=20000)
    parser.add_argument("--page-size", type=int, default=12)
    args = parser.parse_args()
    safe_state(True)

    names = sorted(
        p.name for p in args.models_root.iterdir()
        if p.is_dir() and (p / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud1.map").exists()
    )
    if not names:
        raise SystemExit("no completed models found")

    rows = []
    gt = None
    for index, name in enumerate(names, 1):
        print(f"[{index}/{len(names)}] rendering {name}", flush=True)
        result = render_model(args.models_root / name, args.dataset, args.view, args.iteration)
        if gt is None:
            gt = result["gt"]
        m = result["metrics"]
        rows.append({
            "label": f"{name}\nRGB MAE {m['rgb_mae']:.4f} | A {m['albedo_mae']:.4f} | R {m['roughness_mae']:.4f}",
            "images": result["render"],
            "name": name,
            "metrics": m,
        })

    args.output.mkdir(parents=True, exist_ok=True)
    all_rows = [{"label": f"GT\n{args.view}", "images": gt}] + rows
    save_sheet(all_rows, args.output / f"gt_vs_{len(rows)}_ablations_{args.view}_all.jpg",
               "GT / Experiment", (320, 178), 420)
    for start in range(0, len(rows), args.page_size):
        page_rows = [{"label": f"GT\n{args.view}", "images": gt}] + rows[start:start + args.page_size]
        end = min(start + args.page_size, len(rows))
        save_sheet(page_rows, args.output / f"gt_vs_ablations_{start:02d}_{end - 1:02d}_{args.view}.jpg",
                   "GT / Experiment", (320, 178), 420)
    # Keep image tensors out of the auxiliary JSON; they are only needed while
    # composing the sheets and are not JSON serializable.
    metrics_rows = [
        {"name": row["name"], "metrics": row["metrics"]}
        for row in rows
    ]
    (args.output / "metrics.json").write_text(json.dumps({
        "view": args.view, "iteration": args.iteration, "experiments": metrics_rows,
    }, indent=2), encoding="utf-8")
    print(f"saved {len(rows)} experiments to {args.output}")


if __name__ == "__main__":
    main()
