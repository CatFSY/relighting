import torch
from surfel_tracer import _C


def _timing_start(profile):
    if profile is None:
        return None
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def _timing_stop(profile, stage, start_event):
    if profile is None or start_event is None:
        return
    end_event = torch.cuda.Event(enable_timing=True)
    end_event.record()
    end_event.synchronize()
    elapsed_ms = float(start_event.elapsed_time(end_event))
    times = profile.setdefault("bvh_substages_ms", {})
    calls = profile.setdefault("bvh_substage_calls", {})
    times[stage] = times.get(stage, 0.0) + elapsed_ms
    calls[stage] = calls.get(stage, 0) + 1


class _GaussianTrace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bvh, rays_o, rays_d, gs_idxs, means3D, opacity, ru, rv, normals, features, shs, alpha_min, transmittance_min, deg, back_culling):    
        color = torch.zeros_like(rays_o)
        normal = torch.zeros_like(rays_o)
        feature = torch.zeros(*rays_o.shape[:-1], features.shape[-1], device=rays_o.device, dtype=rays_o.dtype)
        depth = torch.zeros_like(rays_o[:, 0])
        alpha = torch.zeros_like(rays_o[:, 0])
        diagnostics = torch.empty(
            0, dtype=torch.int32, device=rays_o.device)
        bvh.trace_forward(
            rays_o, rays_d, gs_idxs, means3D, opacity, ru, rv, normals, features, shs, 
            color, normal, feature, depth, alpha, 
            alpha_min, transmittance_min, deg, back_culling, 0, diagnostics
        )
        
        ctx.alpha_min = alpha_min
        ctx.transmittance_min = transmittance_min
        ctx.deg = deg
        ctx.bvh = bvh
        ctx.back_culling = back_culling
        ctx.save_for_backward(rays_o, rays_d, gs_idxs, means3D, opacity, ru, rv, normals, features, shs, color, normal, feature, depth, alpha)
        return color, normal, feature, depth, alpha

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_normal, grad_out_feature, grad_out_depth, grad_out_alpha):
        rays_o, rays_d, gs_idxs, means3D, opacity, ru, rv, normals, features, shs, color, normal, feature, depth, alpha = ctx.saved_tensors
        grad_rays_o = torch.zeros_like(rays_o)
        grad_rays_d = torch.zeros_like(rays_d)
        grad_means3D = torch.zeros_like(means3D)
        grad_opacity = torch.zeros_like(opacity)
        grad_ru = torch.zeros_like(ru)
        grad_rv = torch.zeros_like(rv)
        grad_normals = torch.zeros_like(normals)
        grad_features = torch.zeros_like(features)
        grad_shs = torch.zeros_like(shs)
        
        ctx.bvh.trace_backward(
            rays_o, rays_d, gs_idxs, means3D, opacity, ru, rv, normals, features, shs, 
            color, normal, feature, depth, alpha, 
            grad_rays_o, grad_rays_d, grad_means3D, grad_opacity, grad_ru, grad_rv, grad_normals, grad_features, grad_shs,
            grad_out_color, grad_out_normal, grad_out_feature, grad_out_depth, grad_out_alpha,
            ctx.alpha_min, ctx.transmittance_min, ctx.deg, ctx.back_culling
        )
        
        grads = (
            None,
            grad_rays_o,
            grad_rays_d,
            None,
            grad_means3D,
            grad_opacity,
            grad_ru,
            grad_rv,
            grad_normals,
            grad_features,
            grad_shs,
            None,
            None,
            None,
            None,
        )

        return grads


class GaussianTracer():
    def __init__(self, transmittance_min=0.001):
        self.impl = _C.create_gaussiantracer()
        self.transmittance_min = transmittance_min
        
    def build_bvh(self, vertices_b, faces_b, gs_idxs, static=True):
        self.faces_b = faces_b
        self.gs_idxs = gs_idxs.int()
        self.impl.build_bvh(vertices_b[faces_b], static)

    def update_bvh(self, vertices_b, faces_b, gs_idxs):
        assert (self.faces_b == faces_b).all(), "Update bvh must keep the triangle id not change~"
        self.gs_idxs = gs_idxs.int()
        self.impl.update_bvh(vertices_b[faces_b])

    def build_component_ias(self, components):
        """Build one GAS per generic component and one top-level IAS.

        Each component is a dict containing ``vertices``, ``faces``,
        ``gs_idxs``, ``static`` and ``transform``.  Names and scene semantics
        deliberately do not enter the CUDA extension; callers classify any
        asset as static geometry, deforming geometry, or a rigid instance.
        """
        if not components:
            raise ValueError("component IAS requires at least one component")
        triangles = []
        mappings = []
        offsets = [0]
        static_flags = []
        transforms = []
        self.component_faces = []
        self.component_gs_idxs = []
        for index, component in enumerate(components):
            vertices = component["vertices"].contiguous()
            faces = component["faces"].contiguous()
            gs_idxs = component["gs_idxs"].int().contiguous()
            if vertices.is_cuda is False or faces.is_cuda is False:
                raise ValueError(f"component {index} geometry must be CUDA")
            if faces.ndim != 2 or tuple(faces.shape[1:]) != (3,):
                raise ValueError(f"component {index} faces must be [F, 3]")
            if gs_idxs.numel() != faces.shape[0]:
                raise ValueError(
                    f"component {index} needs one GS index per proxy face")
            component_triangles = vertices[faces].contiguous()
            if component_triangles.shape[0] == 0:
                raise ValueError(f"component {index} contains no proxy faces")
            triangles.append(component_triangles)
            mappings.append(gs_idxs)
            offsets.append(offsets[-1] + component_triangles.shape[0])
            static_flags.append(bool(component.get("static", False)))
            transform = torch.as_tensor(
                component.get("transform", torch.eye(4)[:3]),
                dtype=torch.float32, device="cpu")
            if transform.shape == (4, 4):
                transform = transform[:3]
            if transform.shape != (3, 4):
                raise ValueError(
                    f"component {index} transform must be [3, 4] or [4, 4]")
            transforms.append(transform.contiguous())
            self.component_faces.append(faces)
            self.component_gs_idxs.append(gs_idxs)

        self.gs_idxs = torch.cat(mappings, dim=0).int().contiguous()
        self.impl.build_ias(
            torch.cat(triangles, dim=0).contiguous(),
            torch.tensor(offsets, dtype=torch.int64),
            torch.tensor(static_flags, dtype=torch.bool),
            torch.stack(transforms, dim=0).contiguous(),
        )

    def update_component_gas(self, component_index, vertices_b, faces_b,
                             gs_idxs):
        faces_b = faces_b.contiguous()
        gs_idxs = gs_idxs.int().contiguous()
        if component_index < 0 or component_index >= len(self.component_faces):
            raise IndexError("component GAS index out of range")
        if not torch.equal(self.component_faces[component_index], faces_b):
            raise ValueError("component GAS update changed proxy topology")
        if not torch.equal(self.component_gs_idxs[component_index], gs_idxs):
            raise ValueError("component GAS update changed primitive-to-GS mapping")
        self.impl.update_ias_gas(
            int(component_index), vertices_b[faces_b].contiguous())

    def update_component_transforms(self, transforms):
        transforms = torch.as_tensor(
            transforms, dtype=torch.float32, device="cpu")
        if transforms.ndim != 3 or tuple(transforms.shape[1:]) != (3, 4):
            raise ValueError("IAS transforms must have shape [N, 3, 4]")
        if transforms.shape[0] != len(self.component_faces):
            raise ValueError("IAS transform count must remain unchanged")
        self.impl.update_ias_transforms(transforms.contiguous())

    def trace(self, rays_o, rays_d, means3D, opacity, ru, rv, normals,
              features, shs, alpha_min, deg=3, back_culling=False,
              timing_profile=None, trace_mode="full"):
        mode_ids = {"full": 0, "material": 1, "visibility": 2}
        if trace_mode not in mode_ids:
            raise ValueError(
                f"Unknown trace_mode={trace_mode!r}; expected one of "
                f"{tuple(mode_ids)}")
        mode_id = mode_ids[trace_mode]
        if trace_mode == "material" and features.shape[-1] not in (4, 5):
            raise ValueError(
                "material trace requires 4 (RGB+roughness) or 5 "
                f"(RGB+roughness+metallic) features, got {features.shape[-1]}")
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        means3D = means3D.contiguous()
        opacity = opacity.contiguous()
        ru = ru.contiguous()
        rv = rv.contiguous()
        normals = normals.contiguous()
        if features is not None:
            features = features.contiguous()
        else:
            features = torch.zeros_like(means3D[:, :0])
        shs = shs.contiguous()

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)

        B = rays_o.shape[0]
        mask = torch.zeros(B, dtype=torch.bool, device='cuda')
        timing_event = _timing_start(timing_profile)
        self.impl.intersection_test(rays_o, rays_d, self.gs_idxs, means3D, opacity, ru, rv, normals, mask)
        _timing_stop(
            timing_profile, "intersection_test", timing_event)

        timing_event = _timing_start(timing_profile)
        color = (torch.zeros(B, 3, dtype=torch.float32, device='cuda')
                 if trace_mode == "full" else
                 torch.empty(0, dtype=torch.float32, device='cuda'))
        normal = (torch.zeros(B, 3, dtype=torch.float32, device='cuda')
                  if trace_mode != "visibility" else
                  torch.empty(0, dtype=torch.float32, device='cuda'))
        feature = (torch.zeros(B, features.shape[-1], dtype=torch.float32, device='cuda')
                   if trace_mode != "visibility" else
                   torch.empty(0, dtype=torch.float32, device='cuda'))
        depth = (torch.zeros(B, dtype=torch.float32, device='cuda')
                 if trace_mode == "full" else
                 torch.empty(0, dtype=torch.float32, device='cuda'))
        alpha = torch.zeros(B, dtype=torch.float32, device='cuda')
        _timing_stop(
            timing_profile, "output_allocation", timing_event)
        
        timing_event = _timing_start(timing_profile)
        rays_o_ = rays_o[mask]
        rays_d_ = rays_d[mask]
        _timing_stop(
            timing_profile, "ray_compaction", timing_event)
        if timing_profile is not None:
            metadata = timing_profile.setdefault("metadata", {})
            metadata["bvh_input_rays"] = metadata.get("bvh_input_rays", 0) + int(B)
            metadata["bvh_candidate_rays"] = (
                metadata.get("bvh_candidate_rays", 0) + int(rays_o_.shape[0]))
        if not rays_o_.shape[0] == 0:
            timing_event = _timing_start(timing_profile)
            if trace_mode == "full":
                trace_results = _GaussianTrace.apply(
                    self.impl, rays_o_, rays_d_, self.gs_idxs, means3D, opacity,
                    ru, rv, normals, features, shs, alpha_min,
                    self.transmittance_min, deg, back_culling)
            else:
                candidate_color = torch.empty(
                    0, dtype=torch.float32, device='cuda')
                candidate_depth = torch.empty(
                    0, dtype=torch.float32, device='cuda')
                if trace_mode == "material":
                    candidate_normal = torch.zeros_like(rays_o_)
                    candidate_feature = torch.zeros(
                        rays_o_.shape[0], features.shape[-1],
                        dtype=rays_o_.dtype, device=rays_o_.device)
                else:
                    candidate_normal = torch.empty(
                        0, dtype=torch.float32, device='cuda')
                    candidate_feature = torch.empty(
                        0, dtype=torch.float32, device='cuda')
                candidate_alpha = torch.zeros(
                    rays_o_.shape[0], dtype=rays_o_.dtype,
                    device=rays_o_.device)
                diagnostics = (
                    torch.empty(
                        rays_o_.shape[0], 4, dtype=torch.int32,
                        device=rays_o_.device)
                    if timing_profile is not None else
                    torch.empty(0, dtype=torch.int32, device=rays_o_.device)
                )
                self.impl.trace_forward(
                    rays_o_, rays_d_, self.gs_idxs, means3D, opacity,
                    ru, rv, normals, features, shs,
                    candidate_color, candidate_normal, candidate_feature,
                    candidate_depth, candidate_alpha, alpha_min,
                    self.transmittance_min, deg, back_culling, mode_id,
                    diagnostics)
                if diagnostics.numel() > 0:
                    diagnostic_totals = diagnostics.sum(dim=0).tolist()
                    metadata = timing_profile.setdefault("metadata", {})
                    for key, value in zip((
                        "bvh_trace_batches", "bvh_buffered_hits",
                        "bvh_alpha_hits", "bvh_full_hit_batches",
                    ), diagnostic_totals):
                        metadata[key] = metadata.get(key, 0) + int(value)
                trace_results = (
                    candidate_color, candidate_normal, candidate_feature,
                    candidate_depth, candidate_alpha)
            _timing_stop(
                timing_profile, "full_gaussian_trace", timing_event)

            timing_event = _timing_start(timing_profile)
            if trace_mode == "full":
                color[mask], normal[mask], feature[mask], depth[mask], alpha[mask] = \
                    trace_results
            elif trace_mode == "material":
                normal[mask], feature[mask], alpha[mask] = (
                    trace_results[1], trace_results[2], trace_results[4])
            else:
                alpha[mask] = trace_results[4]
            _timing_stop(
                timing_profile, "result_scatter", timing_event)

        timing_event = _timing_start(timing_profile)
        if trace_mode == "full":
            color = color.view(*prefix, 3)
            depth = depth.view(*prefix)
        if trace_mode != "visibility":
            normal = normal.view(*prefix, 3)
            feature = feature.view(*prefix, features.shape[-1])
        alpha = alpha.view(*prefix)
        _timing_stop(
            timing_profile, "result_reshape", timing_event)
        
        return color, normal, feature, depth, alpha
