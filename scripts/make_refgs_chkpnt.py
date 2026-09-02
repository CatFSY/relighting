#!/usr/bin/env python
"""从 4a 的 point_cloud/iteration_XXXX/point_cloud.ply 构造 chkpntXXXX.pth,
让 4b 无需重跑 4a (4a 中途被杀无 chkpnt, 但 save_iterations 存了 ply 全状态).

用法 (irgs env): python scripts/make_refgs_chkpnt.py <link1> <link2> ...
"""
import os
import sys

import torch

sys.path.insert(0, "/amax/home/fengshuangyu/relighting/IRGS")
from scene.ref_gaussian_model import RefGaussianModel

OUT_ROOT = "/amax/home/fengshuangyu/relighting/IRGS/outputs/so101_links"


def make(link, iteration=40000):
    mdir = os.path.join(OUT_ROOT, link, "refgs_full")
    ply = os.path.join(mdir, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    assert os.path.exists(ply), f"missing: {ply}"
    gaussians = RefGaussianModel(3)
    gaussians.load_ply(ply)
    for attr in ("_xyz", "_metallic", "_roughness", "_base_color", "_features_dc",
                 "_features_rest", "_indirect_dc", "_indirect_rest",
                 "_scaling", "_rotation", "_opacity"):
        setattr(gaussians, attr, getattr(gaussians, attr).cuda())
    N = gaussians._xyz.shape[0]
    args19 = (
        gaussians.active_sh_degree,
        gaussians._xyz,
        gaussians._metallic,
        gaussians._roughness,
        gaussians._base_color,
        gaussians._features_dc,
        gaussians._features_rest,
        gaussians._indirect_dc,
        gaussians._indirect_rest,
        gaussians._scaling,
        gaussians._rotation,
        gaussians._opacity,
        torch.zeros(N, dtype=torch.float32, device="cuda"),  # max_radii2D
        torch.zeros((N, 3), dtype=torch.float32, device="cuda"),  # xyz_gradient_accum
        torch.zeros(N, dtype=torch.float32, device="cuda"),  # denom
        None, None, None,  # opt_dict, env_1_dict, env_2_dict (4b 重建)
        1.0,  # spatial_lr_scale (仅影响 xyz lr 比例, 场景 ~0.8m 尺度)
    )
    out = os.path.join(mdir, f"chkpnt{iteration}.pth")
    torch.save((args19, iteration), out)
    print(f"[OK] {link}: {N} pts -> {out}")


if __name__ == "__main__":
    for link in sys.argv[1:]:
        make(link)
