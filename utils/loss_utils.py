#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from kornia.filters import spatial_gradient
from .image_utils import psnr
import numpy as np
import trimesh
import math
from utils.graphics_utils import rgb_to_srgb, srgb_to_rgb

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def first_order_edge_aware_loss(data, img):
    return (spatial_gradient(data[None], order=1)[0].abs() * torch.exp(-spatial_gradient(img[None], order=1)[0].abs())).sum(1).mean()

def masked_l1_loss(pred, gt, mask):
    """Mean |pred - gt| over masked pixels and all channels.

    pred/gt: (C, H, W), mask: (1, H, W) float. Normalized by the masked
    area (not the full frame) so the weight is independent of object size.
    """
    diff = torch.abs(pred - gt) * mask
    normalizer = (mask.sum() * pred.shape[0]).clamp_min(1.0)
    return diff.sum() / normalizer


def colmap_opengl_gt_normal_to_world(viewpoint_camera):
    """Convert DiffusionRenderer OpenGL camera normals to COLMAP world space.

    The shared loader has already rotated its decoded normal into world space
    without the OpenGL-to-OpenCV axis conversion. Recover camera space, apply
    (x, y, z) -> (x, -y, -z), and rotate it back to world space. This helper is
    used only by the opt-in Stage1 GT-normal loss.
    """
    gt_world_uncorrected = viewpoint_camera.gt_normal.cuda()
    rotation_c2w = torch.as_tensor(
        viewpoint_camera.R,
        dtype=gt_world_uncorrected.dtype,
        device=gt_world_uncorrected.device,
    )
    gt_camera_opengl = gt_world_uncorrected.permute(1, 2, 0) @ rotation_c2w
    axis_flip = torch.tensor(
        [1.0, -1.0, -1.0],
        dtype=gt_world_uncorrected.dtype,
        device=gt_world_uncorrected.device,
    )
    gt_camera_colmap = gt_camera_opengl * axis_flip
    gt_world = gt_camera_colmap @ rotation_c2w.T
    return F.normalize(gt_world.permute(2, 0, 1), dim=0, eps=1e-6)

def tv_loss(depth):
    # return spatial_gradient(data[None], order=2)[0, :, [0, 2]].abs().sum(1).mean()
    h_tv = torch.square(depth[..., 1:, :] - depth[..., :-1, :]).mean()
    w_tv = torch.square(depth[..., :, 1:] - depth[..., :, :-1]).mean()
    return h_tv + w_tv

def calculate_loss(viewpoint_camera, pc, render_pkg, opt, iteration):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    
    rendered_image = render_pkg["render"]
    rendered_opacity = render_pkg["rend_alpha"]
    rendered_depth = render_pkg["surf_depth"]
    rendered_normal = render_pkg["rend_normal"]
    visibility_filter = render_pkg["visibility_filter"]
    rend_dist = render_pkg["rend_dist"]
    gt_image = viewpoint_camera.original_image.cuda()

    Ll1 = l1_loss(rendered_image, gt_image)
    ssim_val = ssim(rendered_image, gt_image)
    loss0 = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)
    loss = torch.zeros_like(loss0)
    tb_dict["loss_l1"] = Ll1.item()
    tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
    tb_dict["ssim"] = ssim_val.item()
    tb_dict["loss0"] = loss0.item()
    loss += loss0

    if opt.lambda_normal_render_depth > 0 and iteration > opt.normal_loss_start:
        surf_normal = render_pkg['surf_normal']
        normal_ref = rendered_normal
        if getattr(opt, 'one_way_normal_to_depth', False):
            # 单向：只让 normal 影响 depth（detach rend_normal），
            # 不让毛躁 depth 反向污染 normal
            normal_ref = rendered_normal.detach()
        loss_normal_render_depth = (1 - (normal_ref * surf_normal).sum(dim=0))[None]
        loss_normal_render_depth = loss_normal_render_depth.mean()
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth
        loss = loss + opt.lambda_normal_render_depth * loss_normal_render_depth
    else:
        tb_dict["loss_normal_render_depth"] = torch.zeros_like(loss)

    # Optional Stage1 pseudo-GT normal supervision. It is enabled only by the
    # explicit --normal_supervision switch; lambda_normal_gt controls its weight.
    # Reuse normal_loss_start so geometry can settle before supervision begins.
    if (getattr(opt, "normal_supervision", False) and
            opt.lambda_normal_gt > 0 and iteration > opt.normal_loss_start and
            getattr(viewpoint_camera, "gt_normal", None) is not None):
        render_alpha = render_pkg["rend_alpha"]
        pred_normal = rendered_normal / render_alpha.clamp_min(1e-6)
        pred_normal = F.normalize(pred_normal, dim=0, eps=1e-6)
        gt_normal = colmap_opengl_gt_normal_to_world(viewpoint_camera)
        image_mask = (viewpoint_camera.mask.float().cuda()
                      if viewpoint_camera.mask is not None
                      else (render_alpha.detach() > 0).float())
        normal_error = (1.0 - (pred_normal * gt_normal).sum(dim=0))[None]
        loss_normal_gt = masked_l1_loss(
            normal_error, torch.zeros_like(normal_error), image_mask)
        loss = loss + opt.lambda_normal_gt * loss_normal_gt
        tb_dict["loss_normal_gt"] = opt.lambda_normal_gt * loss_normal_gt
    else:
        tb_dict["loss_normal_gt"] = torch.zeros_like(loss)

    if opt.lambda_dist > 0 and iteration > opt.dist_loss_start:
        dist_loss = opt.lambda_dist * rend_dist.mean()
        tb_dict["loss_dist"] = dist_loss
        loss += dist_loss
    else:
        tb_dict["loss_dist"] = torch.zeros_like(loss)

    # ---- Opacity 正则（anti-floater）：L = λ·mean(o·(1-o))，把 opacity 压向 {0,1} ----
    if getattr(opt, "lambda_opacity_reg", 0) > 0:
        rendered_opacity = render_pkg["rend_alpha"]
        loss_opacity_reg = (rendered_opacity * (1 - rendered_opacity)).mean()
        tb_dict["loss_opacity_reg"] = (opt.lambda_opacity_reg * loss_opacity_reg).item()
        loss = loss + opt.lambda_opacity_reg * loss_opacity_reg
    else:
        tb_dict["loss_opacity_reg"] = 0.0

    if opt.lambda_normal_smooth > 0 and iteration > opt.normal_smooth_from_iter and iteration < opt.normal_smooth_until_iter:
        loss_normal_smooth = first_order_edge_aware_loss(rendered_normal, gt_image)
        tb_dict["loss_normal_smooth"] = loss_normal_smooth.item()
        lambda_normal_smooth = opt.lambda_normal_smooth
        loss = loss + lambda_normal_smooth * loss_normal_smooth
    else:
        tb_dict["loss_normal_smooth"] = torch.zeros_like(loss)
    
    if opt.lambda_depth_smooth > 0 and iteration > 3000:
        loss_depth_smooth = first_order_edge_aware_loss(rendered_depth, gt_image)
        tb_dict["loss_depth_smooth"] = loss_depth_smooth.item()
        lambda_depth_smooth = opt.lambda_depth_smooth
        loss = loss + lambda_depth_smooth * loss_depth_smooth
    else:
        tb_dict["loss_depth_smooth"] = torch.zeros_like(loss)
    
    if viewpoint_camera.mask is not None and opt.lambda_mask_entropy > 0:
        rendered_opacity = render_pkg["rend_alpha"]
        image_mask = viewpoint_camera.mask.float()
        o = rendered_opacity.clamp(1e-6, 1 - 1e-6)
        loss_mask_entropy = -(image_mask * torch.log(o) + (1-image_mask) * torch.log(1 - o)).mean()
        tb_dict["loss_mask_entropy"] = loss_mask_entropy.item()
        loss = loss + opt.lambda_mask_entropy * loss_mask_entropy
    else:
        tb_dict["loss_mask_entropy"] = torch.zeros_like(loss)
        
    tb_dict["loss"] = loss.item()
    
    return loss, tb_dict

def calculate_loss2(viewpoint_camera, pc, render_pkg, opt, iteration):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }

    # 伪真值监督状态日志：前几次迭代 + 每 500 iter 打印一次，
    # 确认 GT 材质（gt_albedo/gt_orm/gt_normal）确实加载并参与 loss
    _sup = []
    
    rendered_normal = render_pkg["rend_normal"]
    gt_image = viewpoint_camera.original_image.cuda()

    if opt.train_ray:
        mask = render_pkg["mask"]
        ray_rgb_gt = gt_image.permute(1, 2, 0)[mask]
        ray_rgb = render_pkg["ray_rgb"]
        Ll1 = F.l1_loss(ray_rgb, ray_rgb_gt)
    else:
        rendered_image = render_pkg["render"]
        Ll1 = F.l1_loss(rendered_image, gt_image) + opt.lambda_dssim * (1.0 - ssim(rendered_image, gt_image))
    tb_dict["loss_l1"] = Ll1.item()
    loss = Ll1
    
    rendered_image_sh = render_pkg["render_sh"]
    loss_sh = (1.0 - opt.lambda_dssim) * l1_loss(rendered_image_sh, gt_image) + opt.lambda_dssim * (1.0 - ssim(rendered_image_sh, gt_image))
    tb_dict["loss_sh"] = loss_sh.item()
    loss += loss_sh

    if opt.lambda_normal_render_depth > 0 and iteration > opt.normal_loss_start:
        surf_normal = render_pkg['surf_normal']
        normal_ref = rendered_normal
        if getattr(opt, 'one_way_normal_to_depth', False):
            # 单向：只让 normal 影响 depth（detach rend_normal），
            # 不让毛躁 depth 反向污染 normal
            normal_ref = rendered_normal.detach()
        loss_normal_render_depth = (1 - (normal_ref * surf_normal).sum(dim=0))[None]
        loss_normal_render_depth = loss_normal_render_depth.mean()
        tb_dict["loss_normal_render_depth"] = (opt.lambda_normal_render_depth * loss_normal_render_depth).item()
        loss = loss + opt.lambda_normal_render_depth * loss_normal_render_depth
    else:
        tb_dict["loss_normal_render_depth"] = 0.0

    if opt.lambda_normal_gt > 0 and getattr(viewpoint_camera, "gt_normal", None) is not None:
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
        else:
            image_mask = torch.ones_like(render_pkg["rend_alpha"])
        render_alpha = render_pkg["rend_alpha"]
        pred_normal = render_pkg["rend_normal"] / render_alpha.clamp_min(1e-6)
        pred_normal = pred_normal / pred_normal.norm(dim=0, keepdim=True).clamp_min(1e-6)
        gt_normal = viewpoint_camera.gt_normal.cuda()
        # cosine distance in [-1, 1]; 0 means identical, 2 means opposite
        loss_normal_gt = (1.0 - (pred_normal * gt_normal).sum(dim=0))[None]
        loss_normal_gt = masked_l1_loss(loss_normal_gt, torch.zeros_like(loss_normal_gt), image_mask)
        tb_dict["loss_normal_gt"] = (opt.lambda_normal_gt * loss_normal_gt).item()
        loss = loss + opt.lambda_normal_gt * loss_normal_gt
        _sup.append(f"normal λ={opt.lambda_normal_gt} gt✓ {loss_normal_gt.item():.4f}")
    else:
        tb_dict["loss_normal_gt"] = 0.0
        if opt.lambda_normal_gt == 0:
            _sup.append("normal 未启用(λ=0)")
        else:
            _sup.append("normal GT缺失!")
            print(f"[监督] WARNING: λ_normal_gt={opt.lambda_normal_gt} 但 gt_normal 未加载 "
                  f"（{viewpoint_camera.image_name}）", flush=True)

    if opt.lambda_dist > 0 and iteration > opt.dist_loss_start:
        rend_dist = render_pkg["rend_dist"]
        dist_loss = opt.lambda_dist * rend_dist.mean()
        tb_dict["loss_dist"] = dist_loss.item()
        loss += dist_loss
    else:
        tb_dict["loss_dist"] = 0.0

    # ---- Opacity 正则（anti-floater）：L = λ·mean(o·(1-o))，把 opacity 压向 {0,1} ----
    if getattr(opt, "lambda_opacity_reg", 0) > 0:
        rendered_opacity = render_pkg["rend_alpha"]
        loss_opacity_reg = (rendered_opacity * (1 - rendered_opacity)).mean()
        tb_dict["loss_opacity_reg"] = (opt.lambda_opacity_reg * loss_opacity_reg).item()
        loss = loss + opt.lambda_opacity_reg * loss_opacity_reg
    else:
        tb_dict["loss_opacity_reg"] = 0.0

    if opt.lambda_depth_smooth > 0 and iteration > 3000:
        rendered_depth = render_pkg["surf_depth"]
        loss_depth_smooth = first_order_edge_aware_loss(rendered_depth, gt_image)
        tb_dict["loss_depth_smooth"] = (opt.lambda_depth_smooth * loss_depth_smooth).item()
        loss = loss + opt.lambda_depth_smooth * loss_depth_smooth
    else:
        tb_dict["loss_depth_smooth"] = 0.0
        
    if viewpoint_camera.mask is not None and opt.lambda_mask_entropy > 0:
        rendered_opacity = render_pkg["rend_alpha"]
        image_mask = viewpoint_camera.mask.float()
        o = rendered_opacity.clamp(1e-6, 1 - 1e-6)
        loss_mask_entropy = -(image_mask * torch.log(o) + (1-image_mask) * torch.log(1 - o)).mean()
        tb_dict["loss_mask_entropy"] = (opt.lambda_mask_entropy * loss_mask_entropy).item()
        loss = loss + opt.lambda_mask_entropy * loss_mask_entropy
    else:
        tb_dict["loss_mask_entropy"] = 0.0
    
    if opt.lambda_base_color_smooth > 0:
        rendered_base_color = render_pkg["base_color_linear"]
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
            loss_base_color_smooth = first_order_edge_aware_loss(rendered_base_color * image_mask, gt_image)
        else:
            loss_base_color_smooth = first_order_edge_aware_loss(rendered_base_color, gt_image)
        tb_dict["loss_base_color_smooth"] = (opt.lambda_base_color_smooth * loss_base_color_smooth).item()
        loss = loss + opt.lambda_base_color_smooth * loss_base_color_smooth
    else:
        tb_dict["loss_base_color_smooth"] = 0.0
    
    if opt.lambda_metallic_smooth > 0:
        rendered_metallic = render_pkg["metallic"]
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
            loss_metallic_smooth = first_order_edge_aware_loss(rendered_metallic * image_mask, gt_image)
        else:
            loss_metallic_smooth = first_order_edge_aware_loss(rendered_metallic, gt_image)
        tb_dict["loss_metallic_smooth"] = (opt.lambda_metallic_smooth * loss_metallic_smooth).item()
        loss = loss + opt.lambda_metallic_smooth * loss_metallic_smooth
    else:
        tb_dict["loss_metallic_smooth"] = 0.0
    
    if opt.lambda_roughness_smooth > 0:
        rendered_roughness = render_pkg["roughness"]
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
            loss_roughness_smooth = first_order_edge_aware_loss(rendered_roughness * image_mask, gt_image)
        else:
            loss_roughness_smooth = first_order_edge_aware_loss(rendered_roughness, gt_image)
        tb_dict["loss_roughness_smooth"] = (opt.lambda_roughness_smooth * loss_roughness_smooth).item()
        loss = loss + opt.lambda_roughness_smooth * loss_roughness_smooth
    else:
        tb_dict["loss_roughness_smooth"] = 0.0
    
    if opt.lambda_normal_smooth > 0:
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
            loss_normal_smooth = first_order_edge_aware_loss(rendered_normal * image_mask, gt_image)
        else:
            loss_normal_smooth = first_order_edge_aware_loss(rendered_normal, gt_image)
        tb_dict["loss_normal_smooth"] = (opt.lambda_normal_smooth * loss_normal_smooth).item()
        loss = loss + opt.lambda_normal_smooth * loss_normal_smooth
    else:
        tb_dict["loss_normal_smooth"] = 0.0
    
    if opt.lambda_light > 0:
        light_direct = render_pkg["ray_light_direct"]
        mean_light = light_direct.mean(-1, keepdim=True).expand_as(light_direct)
        loss_light = F.l1_loss(light_direct, mean_light)
        tb_dict["loss_light"] = (opt.lambda_light * loss_light).item()
        loss = loss + opt.lambda_light * loss_light
    else:
        tb_dict["loss_light"] = 0.0

    if opt.lambda_light_smooth > 0:
        env = render_pkg["env_only"]
        loss_light_smooth = tv_loss(env)
        tb_dict["loss_light_smooth"] = (opt.lambda_light_smooth * loss_light_smooth).item()
        loss = loss + opt.lambda_light_smooth * loss_light_smooth
    else:
        tb_dict["loss_light_smooth"] = 0.0

    # ---- GT material supervision (arb_render_data) ----
    # Predicted maps are already multiplied by render_alpha, GT only supervises
    # inside the object mask. GT albedo is stored RAW LINEAR (converted from
    # the sRGB PNG at load time), orm is natively linear:
    #   albedo: opt.albedo_loss_space == "linear" -> raw linear L1
    #           (render_pkg["base_color_linear"] vs GT as-is)
    #           opt.albedo_loss_space == "srgb"  -> sRGB L1
    #           (render_pkg["base_color"] vs rgb_to_srgb(GT))
    #   roughness/metallic: linear [0,1] (orm G/B channels)
    if getattr(opt, "lambda_albedo", 0.0) > 0 and getattr(viewpoint_camera, "gt_albedo", None) is not None:
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
        else:
            image_mask = torch.ones_like(render_pkg["rend_alpha"])
        if getattr(opt, "albedo_loss_space", "srgb") == "linear":
            pred_albedo = render_pkg["base_color_linear"]
            gt_albedo = viewpoint_camera.gt_albedo.cuda()
        else:
            pred_albedo = render_pkg["base_color"]
            gt_albedo = rgb_to_srgb(viewpoint_camera.gt_albedo.cuda())
        loss_albedo = masked_l1_loss(pred_albedo, gt_albedo, image_mask)
        tb_dict["loss_albedo"] = (opt.lambda_albedo * loss_albedo).item()
        loss = loss + opt.lambda_albedo * loss_albedo
        _sup.append(f"albedo λ={opt.lambda_albedo} gt✓ {loss_albedo.item():.4f}")
    else:
        tb_dict["loss_albedo"] = 0.0
        if opt.lambda_albedo == 0:
            _sup.append("albedo 未启用(λ=0)")
        else:
            _sup.append("albedo GT缺失!")
            print(f"[监督] WARNING: λ_albedo={opt.lambda_albedo} 但 gt_albedo 未加载 "
                  f"（{viewpoint_camera.image_name}）", flush=True)

    if getattr(opt, "lambda_roughness", 0.0) > 0 and getattr(viewpoint_camera, "gt_orm", None) is not None:
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
        else:
            image_mask = torch.ones_like(render_pkg["rend_alpha"])
        gt_roughness = viewpoint_camera.gt_orm[1:2].cuda()
        loss_roughness = masked_l1_loss(render_pkg["roughness"], gt_roughness, image_mask)
        tb_dict["loss_roughness"] = (opt.lambda_roughness * loss_roughness).item()
        loss = loss + opt.lambda_roughness * loss_roughness
        _sup.append(f"rough λ={opt.lambda_roughness} gt✓ {loss_roughness.item():.4f}")
    else:
        tb_dict["loss_roughness"] = 0.0
        if opt.lambda_roughness == 0:
            _sup.append("rough 未启用(λ=0)")
        else:
            _sup.append("rough GT缺失!")
            print(f"[监督] WARNING: λ_roughness={opt.lambda_roughness} 但 gt_orm 未加载 "
                  f"（{viewpoint_camera.image_name}）", flush=True)

    if getattr(opt, "lambda_metallic", 0.0) > 0 and getattr(viewpoint_camera, "gt_orm", None) is not None:
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
        else:
            image_mask = torch.ones_like(render_pkg["rend_alpha"])
        gt_metallic = viewpoint_camera.gt_orm[2:3].cuda()
        loss_metallic = masked_l1_loss(render_pkg["metallic"], gt_metallic, image_mask)
        tb_dict["loss_metallic"] = (opt.lambda_metallic * loss_metallic).item()
        loss = loss + opt.lambda_metallic * loss_metallic
        _sup.append(f"metal λ={opt.lambda_metallic} gt✓ {loss_metallic.item():.4f}")
    else:
        tb_dict["loss_metallic"] = 0.0
        if opt.lambda_metallic == 0:
            _sup.append("metal 未启用(λ=0)")
        else:
            _sup.append("metal GT缺失!")
            print(f"[监督] WARNING: λ_metallic={opt.lambda_metallic} 但 gt_orm 未加载 "
                  f"（{viewpoint_camera.image_name}）", flush=True)

    tb_dict["loss"] = loss.item()

    # 前几次迭代 + 每 500 iter 打印监督状态（确认伪真值参与计算）
    if _sup and (iteration <= 3 or iteration % 500 == 0):
        print(f"[监督] iter {iteration}: " + " | ".join(_sup), flush=True)

    return loss, tb_dict
