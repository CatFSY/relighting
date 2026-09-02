#!/usr/bin/env python
"""Debug: 4a 路径 (RefGaussianModel + render_surfel) 渲染同一 checkpoint, 对比 radii/alpha"""
import sys
sys.path.insert(0, "/amax/home/fengshuangyu/relighting/IRGS")

import torch
from scene import Scene
from scene.ref_gaussian_model import RefGaussianModel
from gaussian_renderer import render_surfel
from arguments.refgs import ModelParams, PipelineParams, OptimizationParams
from train_refgaussian import set_gaussian_para


def main():
    import argparse
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args()
    args.__dict__.update(dict(
        sh_degree=3, source_path="/amax/home/fengshuangyu/relighting/IRGS/dataset/so101_links/moving_jaw_so101_v1_link",
        model_path="/amax/home/fengshuangyu/relighting/IRGS/outputs/so101_links/moving_jaw_so101_v1_link/refgs_smoke",
        images="images", data_device="cuda", eval=False, white_background=False,
        mask_dir="/amax/home/fengshuangyu/relighting/IRGS/dataset/so101_links/moving_jaw_so101_v1_link/masks",
        pred_material_dir="", envmap_resolution=128,
        envmap_min_roughness=0.05, envmap_max_roughness=1.0,
        volume_render_until_iter=0, init_until_iter=0, srgb=False, indirect=False,
    ))
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    gaussians = RefGaussianModel(dataset.sh_degree)
    # set_gaussian_para 需要 enlarge_scale/rough_msk_thr 等字段
    for k, v in [("enlarge_scale", 1.0), ("rough_msk_thr", 0.0),
                 ("init_roughness_value", 0.6), ("init_metallic_value", 0.2),
                 ("metallic_msk_thr", 0.0)]:
        setattr(gaussians, k, v)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    model_args, _ = torch.load(
        "/amax/home/fengshuangyu/relighting/IRGS/outputs/so101_links/moving_jaw_so101_v1_link/refgs_smoke/chkpnt3000.pth",
        weights_only=False)
    gaussians.restore(model_args, opt)
    gaussians.build_bvh()

    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    cam = scene.getTrainCameras()[0]
    s = gaussians.get_scaling
    print(f"refgs scaling: shape={tuple(s.shape)} min={s.min().item():.5f} max={s.max().item():.5f}")
    pkg = render_surfel(cam, gaussians, pipe, bg, srgb=False, opt=opt)
    ra = pkg["rend_alpha"]
    print(f"[render_surfel] alpha sum={ra.sum().item():.0f} max={ra.max().item():.4f} "
          f"mean={ra.mean().item():.6f} radii>0={(pkg['radii'] > 0).sum().item()}/20000")
    print(f"[render_surfel] render mean={pkg['render'].mean().item():.4f}")


if __name__ == "__main__":
    main()
