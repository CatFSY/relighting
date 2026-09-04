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
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.point_utils import depth_to_normal, depths_to_points
from utils.graphics_utils import (rotation_between_z, fibonacci_sphere_sampling,
                                  cosine_hemisphere_fibonacci_sampling,
                                  rgb_to_srgb, srgb_to_rgb)
from utils.refl_utils import  get_specular_color_surfel, get_full_color_volume, get_full_color_volume_indirect, get_specular_color_surfel2
from .ref_gaussian import render_initial, render_surfel, render_volume, render_surfel2
import numpy as np
from utils.system_utils import Timing
import trimesh
import nvdiffrast.torch as dr
import kornia
from torchvision.utils import save_image


def _profile_cuda_start(profile):
    """Record a CUDA-stage start event when detailed profiling is enabled."""
    if profile is None:
        return None
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def _profile_cuda_stop(profile, stage, start_event):
    """Accumulate one CUDA stage, synchronizing only in opt-in profile mode."""
    if profile is None or start_event is None:
        return
    end_event = torch.cuda.Event(enable_timing=True)
    end_event.record()
    end_event.synchronize()
    elapsed_ms = float(start_event.elapsed_time(end_event))
    profile["stages_ms"][stage] = profile["stages_ms"].get(stage, 0.0) + elapsed_ms
    profile["stage_calls"][stage] = profile["stage_calls"].get(stage, 0) + 1


def _new_timing_profile(enabled):
    if not enabled:
        return None
    return {"stages_ms": {}, "stage_calls": {}, "metadata": {}}


def prepare_ir_raster_context(pc):
    """Activate camera-independent raster inputs once for multi-pass inference."""
    return {
        "means3D": pc.get_xyz,
        "opacity": pc.get_opacity,
        "base_color": pc.get_base_color,
        "roughness": pc.get_rough,
        "metallic": pc.get_metallic,
        "scales": pc.get_scaling,
        "rotations": pc.get_rotation,
        "shs": pc.get_features,
    }


def _query_fg_lut(fg_lut, fg_uv, pipe):
    """Query the split-sum GGX LUT using one 2-D UV grid.

    Packing millions of independent queries as [1, N, 1, 2] can exceed a
    CUDA grid-dimension limit in nvdiffrast. The tiled layout represents the
    same flattened UV sequence as [1, H, W, 2] and restores its original
    shape after the lookup.
    """
    original_shape = fg_uv.shape
    uv_flat = fg_uv.reshape(-1, 2)
    if uv_flat.shape[0] == 0:
        return fg_uv.new_empty(original_shape[:-1] + (2,))

    layout = getattr(pipe, "fg_lut_query_layout", "tiled")
    tile_width = int(getattr(pipe, "fg_lut_tile_width", 2048))
    if tile_width <= 0:
        raise ValueError("FG LUT tile width must be positive")

    valid_count = uv_flat.shape[0]
    if layout == "flat":
        uv_grid = uv_flat.reshape(1, -1, 1, 2).contiguous()
    elif layout == "tiled":
        width = min(tile_width, valid_count)
        height = math.ceil(valid_count / width)
        padded_count = height * width
        if padded_count != valid_count:
            padding = uv_flat[-1:].expand(padded_count - valid_count, -1)
            uv_flat = torch.cat((uv_flat, padding), dim=0)
        uv_grid = uv_flat.reshape(1, height, width, 2).contiguous()
    else:
        raise ValueError(
            f"Unknown fg_lut_query_layout={layout!r}; expected 'flat' or 'tiled'")

    queried = dr.texture(
        fg_lut,
        uv_grid,
        filter_mode="linear",
        boundary_mode="clamp",
    ).reshape(-1, 2)
    return queried[:valid_count].reshape(*original_shape)


def compute_2dgs_normal_and_regularizations(allmap, viewpoint_camera, pipe):
    # 2DGS normal and regularizations
    # additional regularizations
    render_alpha = allmap[1:2]
    
    # get normal map
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # Channel 5 is E[z^2] before alpha normalization; channel 7 is the true
    # median-contributor depth emitted by our rasterizer extension.
    render_depth_second_moment = torch.nan_to_num(
        allmap[5:6] / render_alpha.clamp_min(1e-8), 0, 0)
    render_depth_median = torch.nan_to_num(allmap[7:8], 0, 0)
    
    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]
    
    # pseudo surface attributes
    surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate pseudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * render_alpha.detach()
    
    render_var = render_depth_second_moment - render_depth_expected.square()
    return {
        'render_alpha': render_alpha,
        'render_normal': render_normal,
        'render_depth_median': render_depth_median,
        'render_depth_expected': render_depth_expected,
        'render_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
        'render_var': render_var,
    }



def render_ir(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor,
              scaling_modifier=1.0, override_color=None, opt=None, iteration=-1,
              training=False, relight=False, base_color_scale=None,
              material_only=False, uniform_base_color=None,
              uniform_roughness=None, uniform_metallic=None,
              override_normal_map=None, force_visibility_one=False,
              visibility_origin_mode="incident",
              visibility_origin_epsilon=None, profile_timing=False,
              raster_opacity_override=None, ambient_light=None,
              raster_context=None, shading_mask_override=None,
              trace_context=None, minimal_output=False):
    timing_profile = _new_timing_profile(profile_timing)
    if raster_context is None:
        raster_context = prepare_ir_raster_context(pc)
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(
        raster_context["means3D"], requires_grad=training) + 0
    try:
        if training:
            screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    imH = int(viewpoint_camera.image_height)
    imW = int(viewpoint_camera.image_width)

    raster_settings = GaussianRasterizationSettings(
        image_height=imH,
        image_width=imW,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg = torch.zeros_like(bg_color),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = raster_context["means3D"]
    means2D = screenspace_points
    opacity = raster_context["opacity"]
    # Inference-only camera visibility override.  This intentionally affects
    # only Gaussian rasterization: ray-traced lighting and shadows continue to
    # use the complete scene through ``pc``.  It is useful for rendering
    # separate table/object G-buffers without mutating the trained model.
    if raster_opacity_override is not None:
        if raster_opacity_override.shape != opacity.shape:
            raise ValueError(
                "raster_opacity_override must have shape "
                f"{tuple(opacity.shape)}, got "
                f"{tuple(raster_opacity_override.shape)}")
        opacity = raster_opacity_override.to(
            device=opacity.device, dtype=opacity.dtype)
    
    base_color = raster_context["base_color"]
    roughness = raster_context["roughness"]
    metallic = raster_context["metallic"]
    
    scales = raster_context["scales"]
    rotations = raster_context["rotations"]
    cov3D_precomp = None
    
    shs = raster_context["shs"]
    colors_precomp = None
    
    features = torch.cat([base_color, roughness, metallic], dim=-1)

    timing_event = _profile_cuda_start(timing_profile)
    contrib, rendered_image, rendered_features, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        features = features,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
    )
    _profile_cuda_stop(timing_profile, "rasterization", timing_event)
    
    # 2DGS normal and regularizations
    timing_event = _profile_cuda_start(timing_profile)
    # additional regularizations
    render_alpha = allmap[1:2]
    
    # get normal map
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    # Newer rasterizers append true median depth at channel 7.  The CUDA
    # extension installed in older IRGS environments exposes only the original
    # seven G-buffer channels, so fall back to expected depth in that case.
    if allmap.shape[0] > 7:
        render_depth_median = torch.nan_to_num(allmap[7:8], 0, 0)
    else:
        render_depth_median = render_depth_expected
    
    # get depth distortion map
    render_dist = allmap[6:7]
    
    # pseudo surface attributes
    surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    points = surf_depth.permute(1, 2, 0) * viewpoint_camera.rays_d_hw_unnormalized + viewpoint_camera.camera_center
    
    surf_normal = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    surf_normal[1:-1, 1:-1, :] = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    
    surf_normal = surf_normal.permute(2,0,1)
    
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * render_alpha.detach()
    
    # Use normal map computed in 2DGS pipeline to perform reflection query
    normal_map = render_normal.permute(1,2,0)
    normal_map = normal_map / render_alpha.permute(1,2,0).clamp_min(1e-6)  
    normal_map = F.normalize(normal_map, dim=-1)
    if override_normal_map is not None:
        if override_normal_map.shape != render_normal.shape:
            raise ValueError(
                f"override_normal_map must have shape {tuple(render_normal.shape)}, "
                f"got {tuple(override_normal_map.shape)}")
        normal_map = F.normalize(
            override_normal_map.to(
                device=render_normal.device, dtype=render_normal.dtype
            ).permute(1, 2, 0), dim=-1)

    rendered_base_color, rendered_roughness, rendered_metallic = rendered_features.split([3, 1, 1], dim=0)
    if base_color_scale is not None:
        rendered_base_color = rendered_base_color * base_color_scale[:, None, None]

    # Optional inference-time material ablation.  Values are linear PBR
    # parameters.  Defaults are None, so existing training/rendering behavior
    # is unchanged.  Keep the tensors here and pass them into the ray-traced
    # indirect branch as well, otherwise only the camera-visible surface would
    # be uniform while bounce surfaces would retain their learned materials.
    uniform_base_color_tensor = None
    uniform_roughness_tensor = None
    uniform_metallic_tensor = None
    if uniform_base_color is not None:
        uniform_base_color_tensor = torch.as_tensor(
            uniform_base_color, dtype=rendered_base_color.dtype,
            device=rendered_base_color.device).flatten()
        if uniform_base_color_tensor.numel() == 1:
            uniform_base_color_tensor = uniform_base_color_tensor.repeat(3)
        if uniform_base_color_tensor.numel() != 3:
            raise ValueError("uniform_base_color must contain 1 or 3 values")
        rendered_base_color = torch.ones_like(rendered_base_color) * \
            uniform_base_color_tensor[:, None, None]
    if uniform_roughness is not None:
        uniform_roughness_tensor = torch.as_tensor(
            uniform_roughness, dtype=rendered_roughness.dtype,
            device=rendered_roughness.device).reshape(1)
        rendered_roughness = torch.ones_like(rendered_roughness) * \
            uniform_roughness_tensor.view(1, 1, 1)
    if uniform_metallic is not None:
        uniform_metallic_tensor = torch.as_tensor(
            uniform_metallic, dtype=rendered_metallic.dtype,
            device=rendered_metallic.device).reshape(1)
        rendered_metallic = torch.ones_like(rendered_metallic) * \
            uniform_metallic_tensor.view(1, 1, 1)
    _profile_cuda_stop(
        timing_profile, "gbuffer_3d_reconstruction", timing_event)

    if material_only:
        results = {
            "roughness": rendered_roughness * render_alpha,
            "metallic": rendered_metallic * render_alpha,
            "base_color": rgb_to_srgb(rendered_base_color) * render_alpha,
            "base_color_linear": rendered_base_color * render_alpha,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            ## normal, accum alpha, dist, depth map
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
        }
        return results
    
    timing_event = _profile_cuda_start(timing_profile)
    if training:
        if opt.train_ray:
            mask_alpha = render_alpha[0] > 0
            mask_sum = mask_alpha.sum()
            
            num_pixels = opt.trace_num_rays // (pipe.diffuse_sample_num + pipe.light_sample_num)
            if num_pixels > mask_sum:
                ray_ids = torch.arange(mask_sum, device='cuda')
            else:
                ray_ids = torch.multinomial(torch.ones(mask_sum, device=mask_sum.device), num_pixels, replacement=False)

            mask_=mask_alpha[mask_alpha]
            mask_[ray_ids]=False
            mask = torch.zeros_like(mask_alpha)
            mask[mask_alpha]=~mask_
        else:
            mask = render_alpha[0] > 0
    else:
        mask = render_alpha[0] > 0
    if shading_mask_override is not None:
        if shading_mask_override.shape != mask.shape:
            raise ValueError(
                "shading_mask_override must have shape "
                f"{tuple(mask.shape)}, got {tuple(shading_mask_override.shape)}")
        mask = mask & shading_mask_override.to(device=mask.device, dtype=torch.bool)
        
    rays_d = viewpoint_camera.rays_d_hw
    w_o = -rays_d
    use_metallic_brdf = getattr(pipe, 'use_metallic_brdf', False)
    # LS importance sampling needs a PDF over the current environment. During
    # training the environment parameters change every optimizer step, so the
    # PDF must be refreshed here rather than only once in an inference script.
    # update_pdf() is intentionally no-grad: it defines the sampling proposal,
    # while gradients still flow through the queried environment radiance.
    if (getattr(pc, "direct_light", None) is None
            and pipe.light_sample_num > 0
            and pc.get_envmap is not None):
        pc.get_envmap.update_pdf()
    _profile_cuda_stop(timing_profile, "prepare_shading_inputs", timing_event)
    if training:
        render_results = rendering_equation_chunk(
            rendered_base_color.permute(1, 2, 0)[mask],
            rendered_roughness.permute(1, 2, 0)[mask],
            rendered_metallic.permute(1, 2, 0)[mask],
            normal_map[mask], points[mask], w_o[mask], pc,
            pipe=pipe, training=training, camera_center=viewpoint_camera.camera_center,
            use_metallic_brdf=use_metallic_brdf,
            uniform_base_color=uniform_base_color_tensor,
            uniform_roughness=uniform_roughness_tensor,
            uniform_metallic=uniform_metallic_tensor,
            force_visibility_one=force_visibility_one,
            visibility_origin_mode=visibility_origin_mode,
            visibility_origin_epsilon=visibility_origin_epsilon,
            timing_profile=timing_profile,
            trace_context=trace_context)
    else:
        render_results = rendering_equation_chunk(
            rendered_base_color.permute(1, 2, 0)[mask],
            rendered_roughness.permute(1, 2, 0)[mask],
            rendered_metallic.permute(1, 2, 0)[mask],
            normal_map[mask], points[mask], w_o[mask], pc,
            pipe=pipe, training=training, relight=relight, camera_center=viewpoint_camera.camera_center,
            use_metallic_brdf=use_metallic_brdf,
            uniform_base_color=uniform_base_color_tensor,
            uniform_roughness=uniform_roughness_tensor,
            uniform_metallic=uniform_metallic_tensor,
            force_visibility_one=force_visibility_one,
            visibility_origin_mode=visibility_origin_mode,
            visibility_origin_epsilon=visibility_origin_epsilon,
            timing_profile=timing_profile,
            trace_context=trace_context)
        
    timing_event = _profile_cuda_start(timing_profile)
    diffuse = render_results['diffuse']
    specular = render_results['specular']
    light_direct = render_results['light_direct']
    
    rendered_diffuse = torch.zeros_like(rendered_image).permute(1, 2, 0)
    rendered_diffuse[mask] = diffuse
    rendered_diffuse = rendered_diffuse.permute(2, 0, 1)

    # Optional constant environment fill in linear radiance space. This is
    # intentionally independent of direct-light visibility, so point-light
    # shadows retain their shape without collapsing to pure black.
    if ambient_light is not None:
        ambient = torch.as_tensor(
            ambient_light, dtype=rendered_base_color.dtype,
            device=rendered_base_color.device).flatten()
        if ambient.numel() == 1:
            ambient = ambient.repeat(3)
        if ambient.numel() != 3:
            raise ValueError("ambient_light must contain 1 or 3 values")
        rendered_diffuse = (
            rendered_diffuse
            + rendered_base_color * ambient[:, None, None]
        )
        
    rendered_specular = torch.zeros_like(rendered_image).permute(1, 2, 0)
    rendered_specular[mask] = specular
    rendered_specular = rendered_specular.permute(2, 0, 1)
    rendered_full = rgb_to_srgb(rendered_diffuse + rendered_specular)
    final_image = rendered_full * render_alpha + bg_color[:, None, None] * (1 - render_alpha)

    if minimal_output:
        results = {
            "render": final_image,
            "mask": mask,
            "rend_alpha": render_alpha,
            "surf_depth": surf_depth,
        }
        _profile_cuda_stop(timing_profile, "composition", timing_event)
        if timing_profile is not None:
            results["timing_profile"] = timing_profile
        return results
        
    final_image_sh = rgb_to_srgb(rendered_image) + bg_color[:, None, None] * (1 - render_alpha)
    
    if pc.get_envmap is None:
        # Finite analytic lights do not define an infinitely distant visible
        # background.  Keep the diagnostic env buffer black; final compositing
        # still uses bg_color below.
        direct_lights = torch.zeros_like(rendered_image)
    else:
        direct_lights = rgb_to_srgb(pc.get_envmap(rays_d, mode='pure_env').permute(2,0,1))
    env_only = direct_lights
    
    results = {
        "render": final_image,
        "env_only": env_only,
        "render_sh": final_image_sh,
        "diffuse": rgb_to_srgb(rendered_diffuse),
        "specular": rgb_to_srgb(rendered_specular),
        "mask": mask,
        "roughness": rendered_roughness * render_alpha,
        "metallic": rendered_metallic * render_alpha,
        "base_color": rgb_to_srgb(rendered_base_color) * render_alpha,
        "base_color_linear": rendered_base_color * render_alpha,
        "viewspace_points": means2D,
        "visibility_filter" : radii > 0,
        "radii": radii,
        ## normal, accum alpha, dist, depth map
        'rend_alpha': render_alpha,
        'rend_normal': render_normal,
        'rend_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
        "ray_light_direct": light_direct,
    }
    
    if opt is not None and training and opt.train_ray:
        alpha = render_alpha.permute(1,2,0)[mask]
        full = diffuse + specular
        full = rgb_to_srgb(full)
        ray_rgb = full * alpha + bg_color[None, :] * (1 - alpha)
        results.update({
            "ray_rgb": ray_rgb,
        })
        
    if not training:
        
        visibility = render_results['visibility']
        light = render_results['light']
        light_indirect = render_results['light_indirect']
        
        rendered_visibility = torch.zeros_like(rendered_image[:1]).permute(1, 2, 0)
        rendered_visibility[mask] = visibility
        rendered_visibility = rendered_visibility.permute(2, 0, 1) * render_alpha
        
        rendered_light = torch.zeros_like(rendered_image).permute(1, 2, 0)
        rendered_light[mask] = light
        rendered_light = rendered_light.permute(2, 0, 1) * render_alpha
        
        rendered_light_indirect = torch.zeros_like(rendered_image).permute(1, 2, 0)
        rendered_light_indirect[mask] = light_indirect
        rendered_light_indirect = rendered_light_indirect.permute(2, 0, 1) * render_alpha
        
        rendered_light_direct = torch.zeros_like(rendered_image).permute(1, 2, 0)
        rendered_light_direct[mask] = light_direct
        rendered_light_direct = rendered_light_direct.permute(2, 0, 1) * render_alpha
        
        final_image_env = rendered_full * render_alpha + direct_lights * (1 - render_alpha)
        
        results.update({
            "render_env": final_image_env,
            "light_direct": rgb_to_srgb(rendered_light_direct),
            "visibility": rendered_visibility,
            "light": rgb_to_srgb(rendered_light),
            "light_indirect": rgb_to_srgb(rendered_light_indirect),
        })

    _profile_cuda_stop(timing_profile, "composition", timing_event)
    if timing_profile is not None:
        results["timing_profile"] = timing_profile

    return results

def rendering_equation_chunk(base_color, roughness, metallic, normal, position, w_o, pc, pipe, training=False, f0=0.02, relight=False, chunk_size=None, camera_center=None, image_sh=None, timing_profile=None, **kwargs):
    # An interpolated camera can occasionally have no visible Gaussians.  In
    # that case `mask` produces empty material/geometry tensors.  Passing the
    # resulting empty UV tensor to nvdiffrast raises
    # "uv must have shape [>0, >0, >0, 2]".  Return correctly shaped empty
    # shading buffers instead; the caller will composite a pure background /
    # environment frame because the visibility mask is empty.
    if base_color.shape[0] == 0:
        empty_rgb = base_color.new_zeros((0, 3))
        results = {
            "diffuse": empty_rgb,
            "specular": empty_rgb.clone(),
            "light_direct": empty_rgb.clone(),
        }
        if not training:
            results.update({
                "visibility": base_color.new_zeros((0, 1)),
                "light": empty_rgb.clone(),
                "light_indirect": empty_rgb.clone(),
            })
        return results

    if chunk_size is None:
        chunk_size = getattr(pipe, "render_ray_budget", 2**24)
    analytic_light = getattr(pc, "direct_light", None)
    if analytic_light is None:
        samples_per_point = pipe.diffuse_sample_num + pipe.light_sample_num
    else:
        samples_per_point = analytic_light.sample_count(
            getattr(pipe, "analytic_light_sample_num", 64)
        )
    chunk_size = max(chunk_size // max(samples_per_point, 1), 1)
    if timing_profile is not None:
        timing_profile["metadata"].update({
            "foreground_pixels": int(base_color.shape[0]),
            "samples_per_foreground_pixel": int(samples_per_point),
            "pixels_per_shading_chunk": int(chunk_size),
            "shading_chunks": int(math.ceil(base_color.shape[0] / chunk_size)),
        })
    # All shading chunks query the same Gaussian model.  Prepare activated
    # opacity, tangent frames, normals and optional hit materials once rather
    # than rebuilding them for every ray-budget chunk.
    force_full_relight_trace = bool(
        relight and getattr(pipe, "force_full_relight_trace", False))
    if analytic_light is not None or (relight and pipe.wo_indirect_relight):
        trace_mode = "visibility"
    elif force_full_relight_trace:
        trace_mode = "full"
    elif relight:
        trace_mode = "material"
    else:
        trace_mode = "full"
    trace_features = None
    if force_full_relight_trace:
        trace_fields = [pc.get_base_color, pc.get_rough]
        if kwargs.get("use_metallic_brdf", False):
            trace_fields.append(pc.get_metallic)
        trace_features = torch.cat(trace_fields, dim=1)
    if kwargs.get("trace_context") is None:
        kwargs["trace_context"] = pc.prepare_trace_context(
            camera_center=camera_center,
            trace_mode=trace_mode,
            use_metallic=kwargs.get("use_metallic_brdf", False),
            features=trace_features,
        )
    kwargs["trace_mode_override"] = (
        "full" if force_full_relight_trace else None)
    if base_color.shape[0] <= chunk_size:
        return rendering_equation(base_color, roughness, metallic, normal, position, w_o, pc, pipe, training, f0, relight=relight, camera_center=camera_center, timing_profile=timing_profile, **kwargs)
    else:
        results = []
        for i in range(0, base_color.shape[0], chunk_size):
            results.append(rendering_equation(base_color[i:i+chunk_size], roughness[i:i+chunk_size], metallic[i:i+chunk_size], normal[i:i+chunk_size], position[i:i+chunk_size], w_o[i:i+chunk_size], pc, pipe, training, f0, relight=relight, camera_center=camera_center, timing_profile=timing_profile, **kwargs))
        return {k: torch.cat([r[k] for r in results], 0) for k in results[0]}
    
def sample_incident_rays(normals, is_training=False, sample_num=24,
                         sampling_mode="uniform"):
    sampler = (
        fibonacci_sphere_sampling
        if sampling_mode == "uniform"
        else cosine_hemisphere_fibonacci_sampling
        if sampling_mode == "cosine"
        else None
    )
    if sampler is None:
        raise ValueError(
            f"Unknown diffuse sampling mode {sampling_mode!r}; expected "
            "'uniform' or 'cosine'")
    if is_training:
        incident_dirs, incident_areas = sampler(
            normals, sample_num, random_rotate=True)
    else:
        incident_dirs, incident_areas = sampler(
            normals, sample_num, random_rotate=False)

    return incident_dirs, incident_areas  # [N, S, 3], [N, S, 1]


def rendering_equation_analytic_light(
        base_color, roughness, metallic, normals, position, viewdirs, pc, pipe,
        training=False, f0=0.04, camera_center=None,
        use_metallic_brdf=False, force_visibility_one=False,
        visibility_origin_mode="incident", visibility_origin_epsilon=None,
        trace_context=None):
    """Evaluate a finite point or rectangle emitter.

    Unlike an environment map, a finite emitter has a position, inverse-square
    transport and a finite shadow segment.  ``sample_incident`` returns an MC
    weight that already contains the point-light attenuation or the rectangle
    area/Jacobian term, so only BRDF, surface cosine and visibility remain here.
    """
    light = pc.direct_light
    sample_num = light.sample_count(
        getattr(pipe, "analytic_light_sample_num", 64)
    )
    samples = light.sample_incident(position, sample_num, training=training)
    incident_dirs = samples["directions"]
    global_incident_lights = samples["radiance_weight"]

    origin_epsilon = (
        pipe.light_t_min if visibility_origin_epsilon is None
        else visibility_origin_epsilon
    )
    if visibility_origin_mode == "incident":
        trace_origins = position[:, None, :] + incident_dirs * origin_epsilon
    elif visibility_origin_mode == "normal":
        trace_origins = position[:, None, :] + normals[:, None, :] * origin_epsilon
        trace_origins = trace_origins.expand_as(incident_dirs)
    else:
        raise ValueError(
            f"Unknown visibility_origin_mode={visibility_origin_mode!r}; "
            "expected 'incident' or 'normal'"
        )

    if force_visibility_one:
        incident_visibility = torch.ones_like(incident_dirs[..., :1])
    else:
        # Aim the shadow ray at the sampled emitter point after applying the
        # origin offset.  max_distance prevents geometry behind the emitter
        # from casting a false shadow.
        trace_vectors = samples["light_points"] - trace_origins
        trace_distances = torch.linalg.vector_norm(
            trace_vectors, dim=-1, keepdim=True
        ).clamp_min(1e-6)
        trace_directions = trace_vectors / trace_distances
        trace_outputs = pc.trace(
            trace_origins,
            trace_directions,
            camera_center=camera_center,
            max_distance=trace_distances,
            trace_mode="visibility",
            trace_context=trace_context,
        )
        incident_visibility = 1.0 - trace_outputs["alpha"][..., None]

    incident_lights = incident_visibility * global_incident_lights
    n_d_i = (normals[:, None] * incident_dirs).sum(
        dim=-1, keepdim=True
    ).clamp_min(0.0)

    if use_metallic_brdf:
        metallic = metallic[:, None]
        fresnel = 0.04 * (1.0 - metallic) + base_color[:, None] * metallic
        f_d = base_color[:, None] * (1.0 - metallic) / np.pi
        f_s = GGX_specular(
            normals, viewdirs, incident_dirs, roughness, fresnel=fresnel
        )
    else:
        f_d = base_color[:, None] / np.pi
        f_s = GGX_specular(
            normals, viewdirs, incident_dirs, roughness, fresnel=0.04
        )

    transport = incident_lights * n_d_i
    diffuse = (f_d * transport).mean(dim=-2)
    specular = (f_s * transport).mean(dim=-2)

    results = {
        "diffuse": diffuse,
        "specular": specular,
        "light_direct": global_incident_lights.mean(dim=1),
    }
    if not training:
        results.update({
            "visibility": incident_visibility.mean(dim=1),
            "light": incident_lights.mean(dim=1),
            # Analytic emitters currently implement direct illumination and
            # finite-distance shadows.  No fake env-map bounce is added.
            "light_indirect": torch.zeros_like(diffuse),
        })
    return results

def rendering_equation(base_color, roughness, metallic, normals, position,
                       viewdirs, pc, pipe, training=False, f0=0.04,
                       relight=False, camera_center=None,
                       use_metallic_brdf=False, uniform_base_color=None,
                       uniform_roughness=None, uniform_metallic=None,
                       force_visibility_one=False,
                       visibility_origin_mode="incident",
                       visibility_origin_epsilon=None, timing_profile=None,
                       trace_context=None,
                       trace_mode_override=None,
                       **kwargs):
    if getattr(pc, "direct_light", None) is not None:
        return rendering_equation_analytic_light(
            base_color, roughness, metallic, normals, position, viewdirs,
            pc, pipe, training=training, f0=f0,
            camera_center=camera_center,
            use_metallic_brdf=use_metallic_brdf,
            force_visibility_one=force_visibility_one,
            visibility_origin_mode=visibility_origin_mode,
            visibility_origin_epsilon=visibility_origin_epsilon,
            trace_context=trace_context,
        )

    B = base_color.shape[0]
    envmap = pc.get_envmap
    diffuse_sampling_mode = getattr(pipe, "diffuse_sampling_mode", "cosine")
    light_sampling_mode = getattr(
        pipe, "light_sampling_mode", "stratified_shared")

    timing_event = _profile_cuda_start(timing_profile)
    if pipe.diffuse_sample_num > 0 and pipe.light_sample_num == 0:
        incident_dirs, incident_areas = sample_incident_rays(
            normals, training, pipe.diffuse_sample_num,
            sampling_mode=diffuse_sampling_mode)
    elif pipe.diffuse_sample_num > 0 and pipe.light_sample_num > 0:
        p_diffuse = pipe.diffuse_sample_num / (pipe.diffuse_sample_num + pipe.light_sample_num)
        p_light = pipe.light_sample_num / (pipe.diffuse_sample_num + pipe.light_sample_num)
    
        diffuse_directions, diffuse_areas = sample_incident_rays(
            normals, training, pipe.diffuse_sample_num,
            sampling_mode=diffuse_sampling_mode)
        diffuse_pdfs = 1 / diffuse_areas
        
        light_directions, light_pdfs = pc.get_envmap.sample_light_directions(
            B, pipe.light_sample_num, training, strategy=light_sampling_mode)
    
        if diffuse_sampling_mode == "cosine":
            diffuse_pdfs_light = (
                normals[:, None] * light_directions
            ).sum(-1, keepdim=True).clamp_min(0.0) / np.pi
        else:
            diffuse_pdfs_light = 1 / (2 * np.pi)
        light_pdfs_diffuse = pc.get_envmap.light_pdf(diffuse_directions)
        
        diffuse_pdfs = diffuse_pdfs * p_diffuse + light_pdfs_diffuse * p_light
        light_pdfs = diffuse_pdfs_light * p_diffuse + light_pdfs * p_light
        
        incident_dirs = torch.cat([diffuse_directions, light_directions], dim=1)
        incident_pdfs = torch.cat([diffuse_pdfs, light_pdfs], dim=1)
        incident_areas = 1 / incident_pdfs.clamp_min(1e-6)
    elif pipe.diffuse_sample_num == 0 and pipe.light_sample_num > 0:
        # Pure environment-importance sampling.  This is useful for
        # ablations; no MIS mixture is needed when the light proposal is the
        # only sampling distribution.
        incident_dirs, incident_pdfs = pc.get_envmap.sample_light_directions(
            B, pipe.light_sample_num, training, strategy=light_sampling_mode)
        incident_areas = 1 / incident_pdfs.clamp_min(1e-6)
    else:
        raise NotImplementedError
    _profile_cuda_stop(timing_profile, "ds_ls_sampling", timing_event)

    timing_event = _profile_cuda_start(timing_profile)
    global_incident_lights = envmap(incident_dirs, mode='pure_env')
    _profile_cuda_stop(timing_profile, "environment_query", timing_event)

    timing_event = _profile_cuda_start(timing_profile)
    origin_epsilon = (pipe.light_t_min if visibility_origin_epsilon is None
                      else visibility_origin_epsilon)
    if visibility_origin_mode == "incident":
        trace_origins = position.unsqueeze(1) + incident_dirs * origin_epsilon
    elif visibility_origin_mode == "normal":
        trace_origins = position.unsqueeze(1) + \
            normals.unsqueeze(1) * origin_epsilon
        trace_origins = trace_origins.expand_as(incident_dirs)
    else:
        raise ValueError(
            f"Unknown visibility_origin_mode={visibility_origin_mode!r}; "
            "expected 'incident' or 'normal'")
    _profile_cuda_stop(timing_profile, "trace_origin_setup", timing_event)
    
    if relight:
        if pipe.wo_indirect_relight:
            # Direct-only relighting still traces alpha for visibility/shadows,
            # but avoids interpolating hit materials and shading the secondary
            # surface.  The previous implementation computed all indirect
            # terms and only then replaced them with zeros.
            timing_event = _profile_cuda_start(timing_profile)
            trace_outputs = pc.trace(
                trace_origins, incident_dirs, camera_center=camera_center,
                timing_profile=timing_profile, trace_mode="visibility",
                trace_context=trace_context)
            _profile_cuda_stop(timing_profile, "bvh_trace", timing_event)
        else:
            timing_event = _profile_cuda_start(timing_profile)
            trace_outputs = pc.trace(
                trace_origins, incident_dirs,
                camera_center=camera_center,
                timing_profile=timing_profile,
                trace_mode=trace_mode_override or "material",
                trace_context=trace_context,
                use_metallic=use_metallic_brdf)
            _profile_cuda_stop(timing_profile, "bvh_trace", timing_event)
        timing_event = _profile_cuda_start(timing_profile)
        trace_alpha = trace_outputs['alpha'][..., None]
        incident_visibility = 1 - trace_alpha
        if pipe.wo_indirect_relight:
            local_incident_lights = torch.zeros_like(global_incident_lights)
        else:
            trace_feature = trace_outputs['feature'] / trace_alpha.clamp_min(1e-6)
            trace_normal = F.normalize(trace_outputs['normal'], dim=-1)
            if use_metallic_brdf:
                trace_base_color, trace_roughness, trace_metallic = trace_feature.split([3, 1, 1], dim=-1)
            else:
                trace_base_color, trace_roughness = trace_feature.split([3, 1], dim=-1)
            if uniform_base_color is not None:
                trace_base_color = torch.ones_like(trace_base_color) * \
                    uniform_base_color.view(1, 1, 3)
            if uniform_roughness is not None:
                trace_roughness = torch.ones_like(trace_roughness) * \
                    uniform_roughness.view(1, 1, 1)
            if use_metallic_brdf and uniform_metallic is not None:
                trace_metallic = torch.ones_like(trace_metallic) * \
                    uniform_metallic.view(1, 1, 1)
            if use_metallic_brdf:
                trace_diffuse = trace_base_color * (1.0 - trace_metallic) * envmap(trace_normal, mode='diffuse')
            else:
                trace_diffuse = trace_base_color * envmap(trace_normal, mode='diffuse')
            trace_wi = -incident_dirs
            trace_NdotV = (trace_normal * trace_wi).sum(-1, keepdim=True)
            trace_reflected = F.normalize(trace_NdotV * trace_normal * 2 - trace_wi, dim=-1)
            fg_uv = torch.cat([trace_NdotV, trace_roughness], -1).clamp(0, 1)
            # nvdiffrast 0.4.0 cannot launch its texture kernel for the very
            # long 1-D UV tensor produced by large shading chunks.  Keep the
            # outer shading/BVH chunk independent from this implementation
            # limit by querying the LUT in bounded, mathematically identical
            # sub-chunks.
            fg = _query_fg_lut(pc.FG_LUT, fg_uv, pipe)
            if use_metallic_brdf:
                trace_f0 = 0.04 * (1.0 - trace_metallic) + trace_base_color * trace_metallic
            else:
                trace_f0 = f0
            trace_specular = envmap(trace_reflected, roughness=trace_roughness, mode='specular') * (trace_f0 * fg[..., 0:1] + fg[..., 1:2])
            local_incident_lights = (trace_diffuse + trace_specular) * trace_alpha
        incident_lights = incident_visibility * global_incident_lights + local_incident_lights
        _profile_cuda_stop(timing_profile, "indirect_light", timing_event)
    else:
        timing_event = _profile_cuda_start(timing_profile)
        trace_outputs = pc.trace(
            trace_origins, incident_dirs, camera_center=camera_center,
            timing_profile=timing_profile, trace_mode="full",
            trace_context=trace_context)
        _profile_cuda_stop(timing_profile, "bvh_trace", timing_event)
        timing_event = _profile_cuda_start(timing_profile)
        incident_visibility = 1 - trace_outputs['alpha'][..., None]
        local_incident_lights = trace_outputs['color']
        if pipe.wo_indirect:
            local_incident_lights = torch.zeros_like(local_incident_lights)
        if pipe.detach_indirect:
            incident_visibility = incident_visibility.detach()
            local_incident_lights = local_incident_lights.detach()
        _profile_cuda_stop(timing_profile, "indirect_light", timing_event)

    timing_event = _profile_cuda_start(timing_profile)
    if force_visibility_one:
        incident_visibility = torch.ones_like(incident_visibility)
    incident_lights = incident_visibility * global_incident_lights + local_incident_lights
    
    n_d_i = (normals[:, None] * incident_dirs).sum(-1, keepdim=True).clamp(min=0)

    if use_metallic_brdf:
        # 标准 metallic-roughness 工作流
        # metallic: [B, 1] -> [B, 1, 1]
        metallic = metallic[:, None]
        f0 = 0.04 * (1.0 - metallic) + base_color[:, None] * metallic
        f_d = base_color[:, None] * (1.0 - metallic) / np.pi
        f_s = GGX_specular(normals, viewdirs, incident_dirs, roughness, fresnel=f0)
    else:
        # 原 IRGS 行为：metallic 不参与渲染
        f_d = base_color[:, None] / np.pi
        f_s = GGX_specular(normals, viewdirs, incident_dirs, roughness, fresnel=0.04)

    transport = incident_lights * incident_areas * n_d_i  # （num_pts, num_sample, 3)
    diffuse = ((f_d) * transport).mean(dim=-2)
    specular = ((f_s) * transport).mean(dim=-2)
    _profile_cuda_stop(timing_profile, "brdf_integration", timing_event)

    if training:
        results = {
            "diffuse": diffuse,
            "specular": specular,
            "light_direct": global_incident_lights.mean(dim=1),
        }
    else:
        results = {
            "diffuse": diffuse,
            "specular": specular,
            "visibility": incident_visibility.mean(dim=1),
            "light": incident_lights.mean(dim=1),
            "light_indirect": local_incident_lights.mean(dim=1),
            "light_direct": global_incident_lights.mean(dim=1),
        }
    return results

def GGX_specular(
        normal,
        pts2c,
        pts2l,
        roughness,
        fresnel
):
    L = F.normalize(pts2l, dim=-1)  # [nrays, nlights, 3]
    V = F.normalize(pts2c, dim=-1)  # [nrays, 3]
    H = F.normalize((L + V[:, None, :]) / 2.0, dim=-1)  # [nrays, nlights, 3]
    N = F.normalize(normal, dim=-1)  # [nrays, 3]

    NoV = torch.sum(V * N, dim=-1, keepdim=True)  # [nrays, 1]
    N = N * NoV.sign()  # [nrays, 3]

    NoL = torch.sum(N[:, None, :] * L, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1] TODO check broadcast
    NoV = torch.sum(N * V, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, 1]
    NoH = torch.sum(N[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]
    VoH = torch.sum(V[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]

    alpha = roughness * roughness  # [nrays, 3]
    alpha2 = alpha * alpha  # [nrays, 3]
    k = (alpha + 2 * roughness + 1.0) / 8.0
    FMi = ((-5.55473) * VoH - 6.98316) * VoH
    frac0 = fresnel + (1 - fresnel) * torch.pow(2.0, FMi)  # [nrays, nlights, 3]
    
    frac = frac0 * alpha2[:, None, :]  # [nrays, 1]
    nom0 = NoH * NoH * (alpha2[:, None, :] - 1) + 1

    nom1 = NoV * (1 - k) + k
    nom2 = NoL * (1 - k[:, None, :]) + k[:, None, :]
    nom = (4 * np.pi * nom0 * nom0 * nom1[:, None, :] * nom2).clamp_(1e-6, 4 * np.pi)
    spec = frac / nom
    return spec
