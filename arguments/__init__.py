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

from argparse import ArgumentParser, Namespace
import sys
import os
from . import refgs

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                elif t == list: # #
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, nargs="+")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                elif t == list: # #
                    group.add_argument("--" + key, default=value, nargs="+")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        # Rendering Settings
        self.sh_degree = 3
        self._resolution = -1
        self._white_background = False
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']
        self.batch_size = 2**16
        
        # Paths
        self._source_path = ""
        self._model_path = ""
        self._images = "images"

        # Device Settings
        self.data_device = "cuda"
        self.eval = False

        # arb_render_data: which lighting condition (color_{idx}_*) to train on
        self.arb_env_idx = 0
        # 预测材质目录：设置后，arb 数据加载器改从该目录读取 DiffusionRenderer
        # 输出的 basecolor/roughness/metallic/normal 作为材质监督，替代 source_path
        # 下的 GT 材质图（albedo_*.png / orm_*.png / normal_*.exr）
        self.pred_material_dir = ""
        # 物体 mask 目录（COLMAP 数据集）：SAM3 分割输出的 RGBA mask（alpha=物体），
        # 文件名与图像 stem 一致；设置后图像按 mask 合成到背景，预测材质也用该 mask 裁背景
        self.mask_dir = ""

        # EnvLight Settings
        self.envmap_resolution = 8
        self.relight = False
        self.envmap_init_value = 1.5
        self.envmap_activation = 'exp'

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        group = super().extract(args)
        group.source_path = os.path.abspath(group.source_path)
        return group


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        # Processing Settings
        self.convert_SHs_python = False
        self.compute_cov3D_python = False

        # Debugging
        self.depth_ratio = 0.0
        self.debug = False
        # Default relighting estimator: 32 cosine-weighted diffuse samples
        # plus 32 stratified environment-importance samples per visible pixel.
        self.light_sample_num = 32
        self.diffuse_sample_num = 32
        self.specular_sample_num = 0
        self.light_t_min = 0.10
        self.diffuse_sampling_mode = "cosine"
        self.light_sampling_mode = "stratified_shared"
        # At 64 samples/pixel this accommodates up to 262,144 visible pixels
        # in one outer shading chunk, while allocation still follows the
        # actual ray count rather than this upper bound.
        self.render_ray_budget = 2**24
        # Pack large FG LUT queries into 2-D nvdiffrast grids. This avoids the
        # CUDA launch limit of the historical [1, N, 1, 2] layout.
        self.fg_lut_query_layout = "tiled"
        self.fg_lut_tile_width = 2048
        
        self.wo_indirect = False
        self.wo_indirect_relight = False
        self.detach_indirect = False
        # 是否在 BRDF 中真正使用 metallic：True=标准 metallic-roughness 流程；
        # False=与原 IRGS 一致，metallic 仅用于可视化和 GT loss，不参与渲染。
        self.use_metallic_brdf = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # Learning Rate Settings
        self.iterations = 60_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.features_lr = 0.0075 
        self.indirect_lr = 0.0075 
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.lr_scale = 0.0
        # Stage2 默认冻结几何；显式开启后只恢复 xyz/scale/rotation 更新。
        # opacity 仍由 lr_scale 单独控制，避免几何实验同时改变 opacity。
        self.allow_geometry_update = False
        self.geometry_lr_scale = 1.0
        self.allow_opacity_update = False
        self.opacity_lr_scale = 1.0

        self.base_color_lr = 0.0075 
        self.metallic_lr =  0.005 
        self.roughness_lr =  0.005 
        self.normal_lr = 0.006
        self.envmap_cubemap_lr = 0.1
        
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.0
        # Opacity 正则（anti-floater）：L = λ·mean(o·(1-o))，把 opacity 压向 {0,1}
        self.lambda_opacity_reg = 0.0
        self.lambda_normal_render_depth = 0.05
        # one_way_normal_to_depth: loss_normal_render_depth 只让 normal 影响
        # depth(detach rend_normal)，不让毛躁 depth 反向污染 normal
        self.one_way_normal_to_depth = False
        self.lambda_normal_smooth = 0.01
        self.lambda_normal_gt = 0.0
        self.lambda_depth_smooth = 0.0
        self.lambda_mask_entropy = 0.01
        
        self.lambda_base_color_smooth = 0.0
        self.lambda_roughness_smooth = 0.0
        self.lambda_metallic_smooth = 0.0
        self.lambda_light = 0.0
        self.lambda_light_smooth = 0.0

        # GT material supervision (arb_render_data albedo_*.png / orm_*.png)
        self.lambda_albedo = 0.0
        self.lambda_roughness = 0.0
        self.lambda_metallic = 0.0
        # albedo loss domain: "srgb" compares in the dataset's native sRGB
        # space; "linear" converts GT albedo to raw linear and supervises
        # the linear base_color instead
        self.albedo_loss_space = "srgb"
        # tensorboard: image/envmap visualization interval in iterations
        # (loss/psnr scalars are logged every iteration)
        self.tensorboard_vis_interval = 1000

        self.init_roughness_value = 0.7
        self.init_base_color_value = 0.3
        # lower bound of the base_color sigmoid activation. Vanilla IRGS uses
        # 0.03 to avoid albedo collapse; set 0.0 for GT-supervised dark objects
        self.base_color_min = 0.03
        self.init_metallic_value = 0.2

        self.percent_dense = 0.01
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 25000 
        self.densify_grad_threshold = 0.0002
        self.prune_opacity_threshold = 0.005

        self.normal_loss_start = 1000
        self.dist_loss_start = 1000
        
        self.train_ray = False
        self.trace_num_rays = 2**18
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
