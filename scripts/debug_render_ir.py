#!/usr/bin/env python
"""Debug: 复现 4b 第一次 render_ir, 打印 render_alpha/opacity/scaling/bbox 统计

用法 (irgs env):
  python scripts/debug_render_ir.py <cfg_args路径> <refgs_checkpoint.pth>
"""
import pickle
import sys

import torch

sys.path.insert(0, "/amax/home/fengshuangyu/relighting/IRGS")
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render_ir
from train import set_gaussian_para


def main():
    cfg_path, ckpt_path = sys.argv[1], sys.argv[2]
    from argparse import Namespace
    with open(cfg_path) as f:
        args = eval(f.read(), {"Namespace": Namespace})

    import argparse
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    set_gaussian_para(gaussians, opt)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    model_args, _ = torch.load(ckpt_path, weights_only=False)
    gaussians.restore_from_refgs(model_args, opt)
    gaussians.build_bvh()

    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    cam = scene.getTrainCameras()[0]

    print(f"camera 0: center={cam.camera_center.cpu().tolist()}")
    print(f"xyz: n={gaussians.get_xyz.shape[0]} "
          f"min={gaussians.get_xyz.min(0).values.cpu().tolist()} "
          f"max={gaussians.get_xyz.max(0).values.cpu().tolist()}")
    print(f"scaling: shape={tuple(gaussians.get_scaling.shape)} "
          f"min={gaussians.get_scaling.min().item():.4f} max={gaussians.get_scaling.max().item():.4f} "
          f"mean={gaussians.get_scaling.mean().item():.4f}")
    print(f"rotation: shape={tuple(gaussians.get_rotation.shape)}")
    opac = gaussians.get_opacity
    print(f"opacity: min={opac.min().item():.4f} max={opac.max().item():.4f} "
          f"mean={opac.mean().item():.4f} frac>0.5={(opac > 0.5).float().mean().item():.4f}")
    cov = gaussians.get_covariance(1.0)
    print(f"covariance: shape={tuple(cov.shape)}")

    # 直接调 rasterizer (render_ir 同款参数), 分离 rasterizer 与 rendering_equation
    import math
    from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    tanfovx = math.tan(cam.FoVx * 0.5)
    tanfovy = math.tan(cam.FoVy * 0.5)
    rs = GaussianRasterizationSettings(
        image_height=int(cam.image_height), image_width=int(cam.image_width),
        tanfovx=tanfovx, tanfovy=tanfovy, bg=torch.zeros_like(bg),
        scale_modifier=1.0, viewmatrix=cam.world_view_transform,
        projmatrix=cam.full_proj_transform, sh_degree=gaussians.active_sh_degree,
        campos=cam.camera_center, prefiltered=False, debug=False)
    rasterizer = GaussianRasterizer(raster_settings=rs)
    means2D = torch.zeros_like(gaussians.get_xyz, requires_grad=True, device="cuda")
    features = torch.cat([gaussians.get_base_color, gaussians.get_rough, gaussians.get_metallic], dim=-1)
    contrib, rendered_image, rendered_features, radii, allmap = rasterizer(
        means3D=gaussians.get_xyz, means2D=means2D, shs=gaussians.get_features,
        colors_precomp=None, features=features, opacities=gaussians.get_opacity,
        scales=gaussians.get_scaling, rotations=gaussians.get_rotation,
        cov3D_precomp=None)
    ra = allmap[1:2]
    print(f"[rasterizer] radii>0: {(radii > 0).sum().item()}/{radii.numel()} "
          f"max_radii={radii.max().item():.1f}")
    print(f"[rasterizer] alpha: sum={ra.sum().item():.0f} max={ra.max().item():.4f} "
          f"mean={ra.mean().item():.6f}")
    print(f"[rasterizer] rendered_image mean={rendered_image.mean().item():.4f}")

    try:
        pkg = render_ir(cam, gaussians, pipe, bg, opt=opt, iteration=1, training=True)
    except RuntimeError as e:
        print(f"RENDER CRASH: {e}")
        return
    ra = pkg["rend_alpha"]
    print(f"rend_alpha: shape={tuple(ra.shape)} sum={ra.sum().item():.0f} "
          f"max={ra.max().item():.4f} mean={ra.mean().item():.6f}")
    print(f"render: mean={pkg['render'].mean().item():.4f}")
    if "mask" in pkg:
        print(f"mask: sum={pkg['mask'].sum().item()}")


if __name__ == "__main__":
    main()
