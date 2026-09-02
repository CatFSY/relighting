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

import os
import cv2
import json
import torch
from datetime import datetime
from random import randint
from utils.loss_utils import calculate_loss2
from gaussian_renderer import render_ir
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import numpy as np
from tqdm import tqdm
from utils.image_utils import psnr, psnr_ray
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from utils.image_utils import visualize_depth
from utils.graphics_utils import rgb_to_srgb
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, checkpoint_refgs, model_path, debug_from=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger()

    lr_scale = opt.lr_scale
    # 兼容旧行为：未显式开启新开关时，仍由 lr_scale 控制全部旧参数。
    # 新实验中 allow_geometry_update=True 时，仅开启 xyz/scale/rotation。
    geometry_lr_scale = opt.geometry_lr_scale if opt.allow_geometry_update else lr_scale
    opacity_lr_scale = opt.opacity_lr_scale if opt.allow_opacity_update else lr_scale
    opt.position_lr_init *= geometry_lr_scale
    opt.opacity_lr *= opacity_lr_scale
    opt.scaling_lr *= geometry_lr_scale
    opt.rotation_lr *= geometry_lr_scale
    
    gaussians = GaussianModel(dataset.sh_degree)
    set_gaussian_para(gaussians, opt)
    
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)
    elif checkpoint_refgs:
        (model_params, _) = torch.load(checkpoint_refgs, weights_only=False)
        gaussians.restore_from_refgs(model_params, opt)
        
    gaussians.build_bvh(static=False)
    
    if scene.light_rotate:
        transform = torch.tensor([
            [0, -1, 0], 
            [0, 0, 1], 
            [-1, 0, 0]
        ], dtype=torch.float32, device="cuda")
        gaussians.env_map.set_transform(transform)
        
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_psnr_for_log = 0.0
    psnr_test = 0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    iteration = first_iter
    
    while iteration < opt.iterations + 1:
        iter_start.record()

        # gaussians.update_learning_rate(iteration)
        
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render_ir(viewpoint_cam, gaussians, pipe, background, opt=opt, iteration=iteration, training=True)

        gt_image = viewpoint_cam.original_image.cuda()
        
        total_loss, tb_dict = calculate_loss2(viewpoint_cam, gaussians, render_pkg, opt, iteration)
        dist_loss, normal_loss, loss = tb_dict["loss_dist"], tb_dict["loss_normal_render_depth"], tb_dict["loss"]

        total_loss.backward()
            
        iter_end.record()

        with torch.no_grad():
            
            # Densification
            is_densify = False
            # if iteration < opt.densify_until_iter:
            #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
            #                                                          radii[visibility_filter])
            #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
            #     if iteration % opt.densification_interval == 0:
            #         is_densify = True
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_opacity_threshold, scene.cameras_extent,
            #                                     size_threshold)

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            
            if geometry_lr_scale > 0:
                if is_densify:
                    gaussians.build_bvh(static=False)
                else:
                    gaussians.update_bvh()

            if iteration % 500 == 0 or iteration == first_iter + 1:
                save_training_vis(scene, gaussians, background, render_ir, pipe, opt, iteration, num_views=4)

            ema_loss_for_log = 0.4 * loss + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss + 0.6 * ema_normal_for_log
            if opt.train_ray:
                mask = render_pkg["mask"]
                ray_rgb_gt = viewpoint_cam.original_image.cuda().permute(1, 2, 0)[mask]
                ray_rgb = render_pkg["ray_rgb"]
                # psnr() 逐行(每 ray)求 mse，黑像素 mse 下溢成 0 -> inf 污染 EMA；
                # ray 模式用全局 mse 的 psnr_ray 才有可读性
                ema_psnr_for_log = 0.4 * psnr_ray(ray_rgb, ray_rgb_gt).double().item() + 0.6 * ema_psnr_for_log
            else:
                image = render_pkg["render"]
                ema_psnr_for_log = 0.4 * psnr(image, gt_image).mean().double().item() + 0.6 * ema_psnr_for_log
            
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Distort": f"{ema_dist_for_log:.{5}f}",
                    "Normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{gaussians.get_xyz.shape[0]}",
                    "PSNR-train": f"{ema_psnr_for_log:.{4}f}",
                    "PSNR-test": f"{psnr_test:.{4}f}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)

            # tensorboard: scalars every iteration (like vanilla 3DGS),
            # images/envmap every opt.tensorboard_vis_interval
            if tb_writer:
                for key, value in tb_dict.items():
                    if key == "num_points":
                        continue  # logged as scene/total_points instead
                    if torch.is_tensor(value):
                        value = value.item()
                    tb_writer.add_scalar("train_loss/{}".format(key), value, iteration)
                tb_writer.add_scalar("train_loss/psnr_train_ema", ema_psnr_for_log, iteration)
            if tb_writer and iteration % opt.tensorboard_vis_interval == 0:
                tensorboard_report(tb_writer, iteration, scene, pipe, background, dataset)

            if iteration == opt.iterations:
                progress_bar.close()

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                save_path = model_path + f"/chkpnt{iteration}.pth"
                torch.save((gaussians.capture(), iteration), save_path)
                
            if iteration in testing_iterations:
                psnr_test = evaluate_psnr(scene, render_ir, {"pipe": pipe, "bg_color": background, "opt": opt}, iteration)
                if tb_writer:
                    tb_writer.add_scalar("test/psnr", psnr_test, iteration)
        iteration += 1

def set_gaussian_para(gaussians, opt):
    gaussians.init_base_color_value = opt.init_base_color_value
    gaussians.init_metallic_value = opt.init_metallic_value
    gaussians.init_roughness_value = opt.init_roughness_value
    gaussians.base_color_min = opt.base_color_min

def save_training_vis(scene, gaussians, background, render_fn, pipe, opt, iteration, num_views=4):
    with torch.no_grad():
        train_cams = scene.getTrainCameras()
        num_views = min(num_views, len(train_cams))
        indices = np.linspace(0, len(train_cams) - 1, num_views, dtype=int)
        selected_cams = [train_cams[i] for i in indices]

        # ---- 1) 第一个视角保持原来的详细 19 通道可视化 ----
        viewpoint_cam = selected_cams[0]
        render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background, opt=opt)

        error_map = torch.abs(viewpoint_cam.original_image.cuda() - render_pkg["render"])

        visualization_list = [
            viewpoint_cam.original_image.cuda(),
            render_pkg["render"],
            render_pkg["diffuse"],
            render_pkg["specular"],
            render_pkg["render_sh"],
            render_pkg["base_color_linear"],
            render_pkg["base_color"],
            render_pkg["roughness"].repeat(3, 1, 1),
            render_pkg["metallic"].repeat(3, 1, 1),
            render_pkg["visibility"].repeat(3, 1, 1),
            render_pkg["light_indirect"],
            render_pkg["light_direct"],
            render_pkg["light"],
            render_pkg["rend_alpha"].repeat(3, 1, 1),
            visualize_depth(render_pkg["surf_depth"]),
            render_pkg["rend_normal"] * 0.5 + 0.5,
            render_pkg["surf_normal"] * 0.5 + 0.5,
            error_map,
            render_pkg["render_env"],
        ]

        grid = torch.stack(visualization_list, dim=0)
        grid = make_grid(grid, nrow=4)
        scale = grid.shape[-2] / 1600
        grid = F.interpolate(grid[None], (int(grid.shape[-2] / scale), int(grid.shape[-1] / scale)))[0]
        save_image(grid, os.path.join(args.visualize_path, f"{iteration:06d}.png"))

        # ---- 2) 新增：多个训练视角的概览图（每行一个视角，6 列） ----
        overview_tiles = []
        for viewpoint_cam in selected_cams:
            render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background, opt=opt)
            mask = viewpoint_cam.mask.float().cuda() if viewpoint_cam.mask is not None \
                else torch.ones_like(render_pkg["rend_alpha"])

            def on_white(t):
                return torch.where(mask > 0.5, t, torch.ones_like(t))

            overview_tiles.extend([
                viewpoint_cam.original_image.cuda().clamp(0, 1),
                render_pkg["render"].clamp(0, 1),
                on_white(render_pkg["base_color"].clamp(0, 1)),
                on_white(render_pkg["roughness"].repeat(3, 1, 1).clamp(0, 1)),
                on_white(render_pkg["metallic"].repeat(3, 1, 1).clamp(0, 1)),
                on_white((render_pkg["rend_normal"] * 0.5 + 0.5).clamp(0, 1)),
            ])

        overview_grid = make_grid(torch.stack(overview_tiles, dim=0), nrow=6)
        save_image(overview_grid, os.path.join(args.visualize_path, f"{iteration:06d}_views.png"))

        env_dict = gaussians.render_env_map()

        grid = [
            rgb_to_srgb(env_dict["env1"].permute(2, 0, 1)),
            rgb_to_srgb(env_dict["env2"].permute(2, 0, 1)),
        ]
        grid = make_grid(grid, nrow=1, padding=10)
        save_image(grid, os.path.join(args.visualize_path, f"{iteration:06d}_env.png"))

      
NORM_CONDITION_OUTSIDE = False
def prepare_output_and_logger():    
    # Create Tensorboard writer under a per-run date-stamped subfolder so
    # results from different runs stay separated:
    #   model_path/YYYY-MM-DD_HH-MM-SS/events.out.tfevents.*
    tb_writer = None
    if TENSORBOARD_FOUND:
        run_dir = os.path.join(args.model_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        tb_writer = SummaryWriter(run_dir)
        print("Tensorboard run folder: {}".format(run_dir))
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def load_gt_normal_vis(src, view_id, transform_matrices):
    """GT normal for visualization: (n*0.5+0.5)-encoded camera-space (OpenGL)
    EXR -> world space -> [0,1] display encoding, as a (3,H,W) float tensor.
    Returns None if the EXR is missing/unreadable."""
    p = os.path.join(src, f"normal_{view_id:02d}.exr")
    if not os.path.exists(p):
        return None
    n = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if n is None:  # e.g. OPENCV_IO_ENABLE_OPENEXR not set
        return None
    n = cv2.cvtColor(n[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32) * 2.0 - 1.0
    R_c2w = np.array(transform_matrices[view_id], dtype=np.float32)[:3, :3]
    n = n @ R_c2w.T
    n = np.ascontiguousarray(n.transpose(2, 0, 1))
    return torch.from_numpy(n) * 0.5 + 0.5


@torch.no_grad()
def build_vis_grid(viewpoint_cam, pkg, gt_normal=None):
    """2x5 tensorboard grid: row 1 = GT, row 2 = prediction.
    Columns: color/render, albedo, roughness, metallic, normal."""
    mask = viewpoint_cam.mask.float().cuda() if viewpoint_cam.mask is not None \
        else torch.ones_like(pkg["rend_alpha"])

    def on_white(t):
        return torch.where(mask > 0.5, t, torch.ones_like(t))

    pred_row = [pkg["render"].clamp(0, 1),                        # render is sRGB
                on_white(pkg["base_color"].clamp(0, 1)),        # albedo in sRGB for visualization
                on_white(pkg["roughness"][:1].expand(3, -1, -1).clamp(0, 1)),
                on_white(pkg["metallic"][:1].expand(3, -1, -1).clamp(0, 1)),
                on_white((pkg["rend_normal"] * 0.5 + 0.5).clamp(0, 1))]

    # GT row; white placeholder when a GT map is unavailable
    gt_row = [torch.clamp(viewpoint_cam.original_image.cuda(), 0, 1)]  # GT color is sRGB
    if getattr(viewpoint_cam, "gt_albedo", None) is not None:
        gt_row.append(on_white(rgb_to_srgb(viewpoint_cam.gt_albedo.cuda()).clamp(0, 1)))  # GT albedo -> sRGB
    else:
        gt_row.append(torch.ones_like(pred_row[0]))
    if getattr(viewpoint_cam, "gt_orm", None) is not None:
        gt_row.append(on_white(viewpoint_cam.gt_orm[1:2].cuda().expand(3, -1, -1).clamp(0, 1)))
        gt_row.append(on_white(viewpoint_cam.gt_orm[2:3].cuda().expand(3, -1, -1).clamp(0, 1)))
    else:
        gt_row += [torch.ones_like(pred_row[0])] * 2
    if gt_normal is not None:
        gt_row.append(on_white(gt_normal.cuda().clamp(0, 1)))
    else:
        gt_row.append(torch.ones_like(pred_row[0]))

    return make_grid(torch.stack(gt_row + pred_row, dim=0), nrow=5, padding=4)


@torch.no_grad()
def tensorboard_report(tb_writer, iteration, scene, pipe, background, dataset):
    """Image logs following vanilla 3DGS training_report: GT-vs-pred material
    grids for a few train/test views, the learned envmap, gaussian stats."""
    train_cams = scene.getTrainCameras()
    step = max(1, len(train_cams) // 16)
    configs = [{"name": "train", "cameras": train_cams[::step][:16]},
               {"name": "test", "cameras": scene.getTestCameras()[:6]}]
    for config in configs:
        for viewpoint in config["cameras"]:
            pkg = render_ir(viewpoint, scene.gaussians, pipe, background)
            # GT normal is now pre-loaded in Camera; convert from [-1,1] world-space
            # to [0,1] visualization encoding expected by build_vis_grid.
            gt_normal = getattr(viewpoint, "gt_normal", None)
            if gt_normal is not None:
                gt_normal = ((gt_normal + 1.0) / 2.0).clamp(0, 1)
            grid = build_vis_grid(viewpoint, pkg, gt_normal)
            tb_writer.add_image(f"{config['name']}_view_{viewpoint.image_name}/gt_vs_pred",
                                grid, global_step=iteration)

    # learned environment map (same visualization as dump_arb_materials.py)
    env_dict = scene.gaussians.render_env_map()
    env_grid = make_grid([
        rgb_to_srgb(env_dict["env1"].permute(2, 0, 1)).clamp(0, 1),
        rgb_to_srgb(env_dict["env2"].permute(2, 0, 1)).clamp(0, 1),
    ], nrow=1, padding=10)
    tb_writer.add_image("envmap/env1_env2", env_grid, global_step=iteration)

    tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
    tb_writer.add_scalar("scene/total_points", scene.gaussians.get_xyz.shape[0], iteration)
    if scene.gaussians.optimizer is not None:
        env_lr = next((g["lr"] for g in scene.gaussians.optimizer.param_groups if g["name"] == "env"), None)
        if env_lr is not None:
            tb_writer.add_scalar("scene/envmap_lr", env_lr, iteration)
    torch.cuda.empty_cache()

@torch.no_grad()
def evaluate_psnr(scene, renderFunc, renderkwargs, iteration):    
    eval_path = os.path.join(scene.model_path, "eval", "ours_{}".format(iteration))
    os.makedirs(eval_path, exist_ok=True)
    psnr_test = 0.0
    if len(scene.getTestCameras()):
        for idx, viewpoint in enumerate(tqdm(scene.getTestCameras())):
            render_pkg = renderFunc(viewpoint, scene.gaussians, **renderkwargs)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            psnr_test += psnr(image, gt_image).mean().double()
            # save_image(image, os.path.join(eval_path, '{0:05d}'.format(idx) + ".png"))
            # save_image(torch.clamp(render_pkg["diffuse"], 0.0, 1.0), os.path.join(eval_path, '{0:05d}_diffuse'.format(idx) + ".png"))
            # save_image(torch.clamp(render_pkg["specular"], 0.0, 1.0), os.path.join(eval_path, '{0:05d}_specular'.format(idx) + ".png"))
        psnr_test /= len(scene.getTestCameras())
        print("\n[ITER {}] Evaluating test set: PSNR {}".format(iteration, psnr_test))
        with open(os.path.join(eval_path, "psnr.txt"), 'w') as psnr_f:
            psnr_f.write(str(psnr_test))
    torch.cuda.empty_cache()
    return psnr_test

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000,60000,70000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("-c", "--start_checkpoint", type=str, default = None)
    parser.add_argument("--start_checkpoint_refgs", type=str, default = None)
    parser.add_argument('--gui', action='store_true', default=False, help="use gui")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)
    args.save_iterations = args.save_iterations + [i for i in range(5000, args.iterations+1, 5000)]
    args.checkpoint_iterations = args.checkpoint_iterations + [i for i in range(5000, args.iterations+1, 5000)]
    
    # Set up output folder
    os.makedirs(args.model_path, exist_ok = True)
    full_cmd = f"python {' '.join(sys.argv)}"
    print("Command: " + full_cmd)
    
    with open(os.path.join(args.model_path, "cmd.txt"), 'w') as cmd_f:
        cmd_f.write(full_cmd)
    
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    
    print("Output folder: {}".format(args.model_path))
    args.visualize_path = os.path.join(args.model_path, "visualize")
    os.makedirs(args.visualize_path, exist_ok=True)
    print("Visualization folder: {}".format(args.visualize_path))
    

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint_refgs, args.model_path)

    # All done
    print("\nTraining complete.")
