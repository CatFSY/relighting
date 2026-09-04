#!/usr/bin/env python
"""Render the Isaac SO101 pick trajectory as one dynamic IRGS scene.

All final geometry is represented by 2D Gaussians.  Mesh files are not loaded
at render time; their audited bounds are used only to reject table/cup floaters
and to reproduce the canonical transforms recorded in execution_contract.json.

Scene convention:
  * Isaac world is Z-up and measured in metres.
  * IRGS scene coordinates use the same axes/origin, enlarged by SCENE_SCALE=4.
  * Robot link PLYs are already enlarged by four in their URDF link frames.
  * Table/cup video reconstructions are converted to Isaac metres first and
    then enlarged by four with the rest of the scene.

The table and 80 mm cup use the exact source-to-V32 transforms archived in
dataset/guiji1/provenance/complete_coordinate_transforms.json.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from PIL import Image, ImageDraw
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from trimesh.transformations import euler_matrix, rotation_matrix


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from arguments import PipelineParams  # noqa: E402
from gaussian_renderer import prepare_ir_raster_context, render_ir  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.light import EnvLight, PointLight, RectAreaLight  # noqa: E402
from utils.graphics_utils import getProjectionMatrix  # noqa: E402


SCENE_SCALE = 4.0
URDF_RELATIVE_PATH = Path("SO-ARM100/Simulation/SO101/so101_new_calib.urdf")
TRACE_ROOT = ROOT / "dataset/guiji1"
TRACE = TRACE_ROOT / "trajectories/formal_task_lift_trace.csv"
CONTRACT = TRACE_ROOT / "configs/execution_contract.json"
TABLE_ALIGNMENT = TRACE_ROOT / "results/newbase_geometry_mapping.json"
TRANSFORM_DOSSIER = TRACE_ROOT / "provenance/complete_coordinate_transforms.json"
TABLE_PLY = ROOT / "outputs/input_video_frames_newbase/irgs/point_cloud/iteration_20000/point_cloud.ply"
CUP_PLY = ROOT / "outputs/input_video_frames_beizi/irgs/point_cloud/iteration_20000/point_cloud.ply"
ROBOT_ROOT = Path(os.environ.get(
    "SO101_IRGS_ROBOT_ROOT", str(ROOT / "outputs/so101_links")))
DEFAULT_OUT = ROOT / "outputs/guiji1_irgs"

LINKS = [
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "lower_arm_link",
    "wrist_link",
    "gripper_link",
    "moving_jaw_so101_v1_link",
]
JOINT_COLUMN = {
    "shoulder_pan": "Rotation",
    "shoulder_lift": "Pitch",
    "elbow_flex": "Elbow",
    "wrist_flex": "Wrist_Pitch",
    "wrist_roll": "Wrist_Roll",
    "gripper": "Jaw",
}

# These caches are used only when a handoff worker keeps this module alive
# across jobs. Direct one-shot invocations retain their previous behaviour.
_PLY_CACHE = {}
_AFFINE_SURFEL_CACHE = {}
_URDF_CACHE = {}
_PERSISTENT_MODEL_CACHE = {}
_ACTIVE_VIDEO_WRITERS = set()
_PLY_CACHE_HITS = 0
_PLY_CACHE_MISSES = 0


def resolve_urdf_path(explicit_path=None) -> Path:
    """Resolve the SO101 URDF in source checkouts and portable handoff bundles."""
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    if os.environ.get("SO101_IRGS_URDF"):
        candidates.append(Path(os.environ["SO101_IRGS_URDF"]).expanduser())
    candidates.extend((REPO_ROOT / URDF_RELATIVE_PATH, ROOT / URDF_RELATIVE_PATH))
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate
    attempted = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "找不到 SO101 URDF。请传 --urdf 或设置 SO101_IRGS_URDF。"
        f"\n已检查:\n  {attempted}"
    )


def resolve_trajectory_csv(trace_root: Path, explicit_path=None) -> Path:
    """Prefer the formal trace and support early-stop ``.partial.csv`` traces."""
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"轨迹 CSV 不存在: {path}")
        return path
    candidates = [
        trace_root / "trajectories/formal_task_lift_trace.csv",
        trace_root / "trajectories/formal_task_lift_trace.partial.csv",
        trace_root / "trajectory.csv",
    ]
    existing = [path.resolve() for path in candidates if path.is_file()]
    if not existing:
        raise FileNotFoundError(
            f"{trace_root} 中没有 formal_task_lift_trace.csv、"
            "formal_task_lift_trace.partial.csv 或 trajectory.csv"
        )
    return existing[0]


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, scalar-first; supports q2 shape (..., 4)."""
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    q_xyzw = Rotation.from_matrix(np.asarray(matrix)).as_quat()
    return np.concatenate([q_xyzw[..., 3:4], q_xyzw[..., :3]], axis=-1)


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return Rotation.from_quat(np.concatenate([q[..., 1:], q[..., :1]], axis=-1)).as_matrix()


def load_irgs_ply(path: Path, bounds=None, margin=0.0):
    global _PLY_CACHE_HITS, _PLY_CACHE_MISSES
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    bounds_key = None if bounds is None else tuple(
        np.asarray(bounds, dtype=np.float64).reshape(-1).tolist())
    cache_key = (
        str(path), stat.st_size, stat.st_mtime_ns, bounds_key, float(margin))
    cached = _PLY_CACHE.get(cache_key)
    if cached is not None:
        _PLY_CACHE_HITS += 1
        return cached

    _PLY_CACHE_MISSES += 1
    v = PlyData.read(str(path))["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float64)
    keep = np.ones(len(xyz), dtype=bool)
    if bounds is not None:
        lo, hi = np.asarray(bounds, dtype=np.float64)
        keep = ((xyz >= lo - margin) & (xyz <= hi + margin)).all(axis=1)
    v = v[keep]
    xyz = xyz[keep]

    scale_names = sorted(
        [n for n in v.dtype.names if n.startswith("scale_")],
        key=lambda n: int(n.rsplit("_", 1)[1]),
    )
    rest_names = sorted(
        [n for n in v.dtype.names if n.startswith("f_rest_")],
        key=lambda n: int(n.rsplit("_", 1)[1]),
    )
    result = {
        "xyz": xyz,
        "base_color": np.stack([v[f"base_color_{i}"] for i in range(3)], -1),
        "roughness": np.asarray(v["roughness"])[:, None],
        "metallic": np.asarray(v["metallic"])[:, None],
        "opacity": np.asarray(v["opacity"])[:, None],
        "scale": np.stack([v[n] for n in scale_names], -1),
        "rot": np.stack([v[f"rot_{i}"] for i in range(4)], -1),
        "f_dc": np.stack([v[f"f_dc_{i}"] for i in range(3)], -1)[:, None, :],
        # GaussianModel.save_ply() stores SH rest coefficients after
        # transposing [N, 15, 3] to [N, 3, 15] and flattening.  Undo that
        # layout exactly so loading and re-saving is lossless.
        "f_rest": np.stack([v[n] for n in rest_names], -1)
        .reshape(len(v), 3, 15)
        .transpose(0, 2, 1),
    }
    loaded = (result, int((~keep).sum()))
    _PLY_CACHE[cache_key] = loaded
    return loaded


def resolve_ply_path(path: Path) -> Path:
    """Accept either a PLY file or an IRGS output directory."""
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return path
    if path.is_dir():
        # An IRGS output directory commonly contains both the stage-1
        # ``refgs`` model and the stage-2 ``irgs`` model.  They have the same
        # vertex schema but very different material fields; choosing the
        # numerically largest iteration across both trees can silently pick
        # ``refgs/iteration_50000`` when the caller intended the trained IRGS
        # ``irgs/iteration_20000`` model.  Prefer IRGS explicitly, then a
        # direct point_cloud directory, and only use the recursive fallback
        # when neither conventional layout exists.
        candidate_groups = [
            list(path.glob("irgs/point_cloud/iteration_*/point_cloud.ply")),
            list(path.glob("point_cloud/iteration_*/point_cloud.ply")),
            list(path.glob("refgs/point_cloud/iteration_*/point_cloud.ply")),
            list(path.rglob("point_cloud.ply")),
        ]
        candidates = next((group for group in candidate_groups if group), [])
        if candidates:
            def iteration_number(candidate):
                try:
                    return int(candidate.parent.name.split("_")[-1])
                except ValueError:
                    return -1
            return max(candidates, key=iteration_number)
    raise FileNotFoundError(f"找不到 PLY 或 IRGS 输出目录: {path}")


def contract_output_path(contract, role, trace_root):
    """Read an optional IRGS output reference from a trajectory contract."""
    keys = {
        "table": ("table_irgs_output", "table_output", "base_irgs_output"),
        "object": ("object_irgs_output", "object_output", "cup_irgs_output"),
    }[role]
    for key in keys:
        value = contract.get(key)
        if value:
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = trace_root / candidate
            return resolve_ply_path(candidate)
    return None


def subset(data, mask):
    return {key: value[mask] for key, value in data.items()}


def rigid_transform(data, rotation, translation, uniform_scale=1.0):
    result = {key: value.copy() for key, value in data.items()}
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    result["xyz"] = uniform_scale * (data["xyz"] @ rotation.T) + translation
    result["scale"] = data["scale"] + math.log(uniform_scale)
    q = matrix_to_quat_wxyz(rotation)
    result["rot"] = quat_mul(q, data["rot"])
    result["rot"] /= np.linalg.norm(result["rot"], axis=-1, keepdims=True)
    return result


def rigid_geometry(data, rotation, translation, uniform_scale=1.0):
    """Rigid/uniform transform of only the three frame-varying attributes."""
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    q = matrix_to_quat_wxyz(rotation)
    transformed_rotation = quat_mul(q, data["rot"])
    transformed_rotation /= np.linalg.norm(transformed_rotation, axis=-1, keepdims=True)
    return {
        "xyz": uniform_scale * (data["xyz"] @ rotation.T) + translation,
        "scale": data["scale"] + math.log(uniform_scale),
        "rot": transformed_rotation,
    }


def affine_transform_surfels(data, linear, translation, cache_key=None):
    """Apply a general affine map and refactor each rank-2 Gaussian frame."""
    if cache_key is not None and cache_key in _AFFINE_SURFEL_CACHE:
        return dict(_AFFINE_SURFEL_CACHE[cache_key])
    result = {key: value.copy() for key, value in data.items()}
    linear = np.asarray(linear, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    result["xyz"] = data["xyz"] @ linear.T + translation

    rotations = quat_wxyz_to_matrix(data["rot"])
    tangent = rotations[:, :, :2] * np.exp(data["scale"])[:, None, :]
    tangent = np.einsum("ij,njk->nik", linear, tangent)
    u, singular, _ = np.linalg.svd(tangent, full_matrices=False)

    old_normal = rotations[:, :, 2]
    new_normal = old_normal @ np.linalg.inv(linear)
    new_normal /= np.linalg.norm(new_normal, axis=-1, keepdims=True)
    cross = np.cross(u[:, :, 0], u[:, :, 1])
    flip = np.sum(cross * new_normal, axis=-1) < 0
    u[flip, :, 1] *= -1
    cross = np.cross(u[:, :, 0], u[:, :, 1])
    new_rotation = np.stack([u[:, :, 0], u[:, :, 1], cross], axis=-1)
    result["rot"] = matrix_to_quat_wxyz(new_rotation)
    result["scale"] = np.log(np.maximum(singular, 1e-12))
    if cache_key is not None:
        _AFFINE_SURFEL_CACHE[cache_key] = dict(result)
    return result


def concatenate(parts):
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}


def concatenate_geometry(parts):
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in ("xyz", "scale", "rot")}


def parse_urdf(path):
    identity = path_identity(path)
    if identity in _URDF_CACHE:
        return _URDF_CACHE[identity]
    joints = {}
    for element in ET.parse(path).getroot().findall("joint"):
        origin = element.find("origin")
        axis = element.find("axis")
        xyz = np.array([float(x) for x in origin.attrib.get("xyz", "0 0 0").split()])
        rpy = np.array([float(x) for x in origin.attrib.get("rpy", "0 0 0").split()])
        axis_xyz = np.array([float(x) for x in axis.attrib.get("xyz", "0 0 1").split()]) if axis is not None else np.array([0.0, 0.0, 1.0])
        joints[element.attrib["name"]] = {
            "parent": element.find("parent").attrib["link"],
            "child": element.find("child").attrib["link"],
            "xyz": xyz,
            "rpy": rpy,
            "axis": axis_xyz,
            "type": element.attrib["type"],
        }
    _URDF_CACHE[identity] = joints
    return joints


def fk(joints, angles):
    transforms = {"base_link": np.eye(4)}
    while True:
        added = 0
        for name, joint in joints.items():
            if joint["child"] in transforms or joint["parent"] not in transforms:
                continue
            origin = euler_matrix(*joint["rpy"], axes="sxyz")
            origin[:3, 3] = joint["xyz"]
            motion = np.eye(4)
            if joint["type"] in ("revolute", "continuous"):
                motion = rotation_matrix(angles.get(name, 0.0), joint["axis"])
            transforms[joint["child"]] = transforms[joint["parent"]] @ origin @ motion
            added += 1
        if not added:
            return transforms


def pose_matrix(position, quaternion_wxyz):
    result = np.eye(4)
    result[:3, :3] = quat_wxyz_to_matrix(np.asarray(quaternion_wxyz))
    result[:3, 3] = np.asarray(position)
    return result


def robot_world_from_base(contract):
    root = contract["robot_root_transform"]
    world_from_active_root = pose_matrix(root["position_world_m"], root["orientation_wxyz"])
    active_root_from_base = np.asarray(root["active_usd_root_from_urdf_base_link"], dtype=np.float64)
    return world_from_active_root @ active_root_from_base


class SceneCamera:
    def __init__(self, c2w, width, height, fov_y, znear=0.01, zfar=100.0):
        aspect = width / height
        fov_x = 2 * math.atan(aspect * math.tan(fov_y / 2))
        self.FoVx, self.FoVy = fov_x, fov_y
        self.image_width, self.image_height = width, height
        self.znear, self.zfar = znear, zfar
        w2c = np.linalg.inv(c2w)
        self.world_view_transform = torch.tensor(w2c.T, dtype=torch.float32, device="cuda")
        self.projection_matrix = getProjectionMatrix(znear, zfar, fov_x, fov_y).transpose(0, 1).cuda()
        self.full_proj_transform = self.world_view_transform.unsqueeze(0).bmm(
            self.projection_matrix.unsqueeze(0)
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        v, u = torch.meshgrid(
            torch.arange(height, device="cuda"),
            torch.arange(width, device="cuda"),
            indexing="ij",
        )
        fx = width / (2 * math.tan(fov_x / 2))
        fy = height / (2 * math.tan(fov_y / 2))
        rays = torch.stack(
            [(u - width / 2 + 0.5) / fx, (v - height / 2 + 0.5) / fy, torch.ones_like(u)], -1
        ).reshape(-1, 3)
        rays = rays @ self.world_view_transform[:3, :3].T
        self.rays_d_hw_unnormalized = rays.reshape(height, width, 3)
        self.rays_d_hw = F.normalize(rays, dim=-1).reshape(height, width, 3)


def look_at_colmap(eye, target, world_up):
    eye, target, world_up = map(lambda x: np.asarray(x, dtype=np.float64), (eye, target, world_up))
    outward = eye - target
    outward /= np.linalg.norm(outward)
    right = np.cross(world_up, outward)
    right /= np.linalg.norm(right)
    look = -outward
    down = np.cross(right, outward)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([right, down, look], axis=1)
    c2w[:3, 3] = eye
    return c2w


def make_gaussian_model(data, base_color_min=0.0, persistent_key=None,
                        update_cached_geometry=True):
    if persistent_key is not None and persistent_key in _PERSISTENT_MODEL_CACHE:
        model = _PERSISTENT_MODEL_CACHE[persistent_key]
        if len(model.get_xyz) != len(data["xyz"]):
            raise RuntimeError("persistent Gaussian cache size mismatch")
        model.base_color_min = float(base_color_min)
        if update_cached_geometry:
            update_geometry(model, data)
        model._persistent_cache_reused = True
        return model

    # Keep at most one resident scene per GPU. A changed asset signature must
    # not accumulate multiple ~16 GB models in a long-lived worker.
    if persistent_key is not None and _PERSISTENT_MODEL_CACHE:
        _PERSISTENT_MODEL_CACHE.clear()
        gc.collect()
        torch.cuda.empty_cache()
    model = GaussianModel(3)
    model.base_color_min = float(base_color_min)
    for attr, key in [
        ("_xyz", "xyz"),
        ("_base_color", "base_color"),
        ("_roughness", "roughness"),
        ("_metallic", "metallic"),
        ("_opacity", "opacity"),
        ("_scaling", "scale"),
        ("_rotation", "rot"),
        ("_features_dc", "f_dc"),
        ("_features_rest", "f_rest"),
    ]:
        setattr(model, attr, nn.Parameter(torch.as_tensor(data[key], dtype=torch.float32, device="cuda")))
    model.active_sh_degree = 3
    model.max_radii2D = torch.zeros(len(data["xyz"]), device="cuda")
    model._persistent_cache_reused = False
    model._persistent_component_signature = None
    if persistent_key is not None:
        _PERSISTENT_MODEL_CACHE[persistent_key] = model
    return model


def update_geometry(model, data):
    model._xyz.data.copy_(torch.as_tensor(data["xyz"], dtype=torch.float32, device="cuda"))
    model._scaling.data.copy_(torch.as_tensor(data["scale"], dtype=torch.float32, device="cuda"))
    model._rotation.data.copy_(torch.as_tensor(data["rot"], dtype=torch.float32, device="cuda"))
    model.invalidate_trace_cache()


def path_identity(path):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns


def persistent_asset_key(args, table_path, object_path, robot_root, urdf_path,
                         table_bounds, object_bounds, table_transform,
                         object_transform, world_from_base, gaussian_counts):
    """Hash every invariant that affects resident Gaussian data or rigid GASes."""
    payload = {
        "table": path_identity(table_path),
        "object": path_identity(object_path),
        "robot": [
            path_identity(resolve_ply_path(Path(robot_root) / link))
            for link in LINKS
        ],
        "urdf": path_identity(urdf_path),
        "scene": args.scene,
        "bvh_layout": args.bvh_layout,
        "bounding_polyhedron": os.environ.get(
            "IRGS_GS_BOUNDING_POLYHEDRON", "icosphere320"),
        "base_color_min": args.base_color_min,
        "gaussian_scale_multiplier": args.gaussian_scale_multiplier,
        "table_z_offset_m": args.table_z_offset_m,
        "table_bounds": None if table_bounds is None else np.asarray(
            table_bounds).tolist(),
        "object_bounds": None if object_bounds is None else np.asarray(
            object_bounds).tolist(),
        "table_transform": np.asarray(table_transform).tolist(),
        "object_transform": np.asarray(object_transform).tolist(),
        "world_from_base": np.asarray(world_from_base).tolist(),
        "gaussian_counts": list(map(int, gaussian_counts)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def draw_label(image, row, render_seconds, extra=None):
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    text = f"step={row['step']}  t={float(row['sim_time_s']):.2f}s  {row['stage']}"
    if extra:
        text += f"  {extra}"
    text += f"  render={render_seconds:.2f}s"
    draw.rectangle((0, 0, min(image.width, 570), 28), fill=(255, 255, 255))
    draw.text((8, 7), text, fill=(0, 0, 0))
    return image


def save_compositing_debug(debug_dir, step, rendered, debug, args):
    """Save exact depth-compositing inputs and an easy-to-read region overlay."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    stem = f"step_{int(step):06d}"

    rgb = rendered.detach().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
    foreground_alpha = debug["foreground_alpha"].detach().cpu().squeeze().numpy()
    table_alpha = debug["table_alpha"].detach().cpu().squeeze().numpy()
    depth_gap_m = (
        debug["depth_gap"].detach().cpu().squeeze().numpy() / SCENE_SCALE
    )
    valid_depth = debug["valid_depth"].detach().cpu().squeeze().numpy().astype(bool)
    correction = (
        debug["correction_confidence"].detach().cpu().squeeze().numpy() > 0.0
    )
    overlap = (
        valid_depth
        & (foreground_alpha > 1.0 / 255.0)
        & (table_alpha > 1.0 / 255.0)
    )
    ordinary_alpha = overlap & ~correction
    far_ordinary = ordinary_alpha & (depth_gap_m > args.occlusion_depth_high)
    uncertain_ordinary = ordinary_alpha & ~far_ordinary

    rgb_u8 = np.rint(rgb * 255.0).astype(np.uint8)
    Image.fromarray(rgb_u8).save(debug_dir / f"{stem}_render.png")
    Image.fromarray(np.rint(foreground_alpha.clip(0, 1) * 255).astype(np.uint8)).save(
        debug_dir / f"{stem}_foreground_alpha.png")
    Image.fromarray(np.rint(table_alpha.clip(0, 1) * 255).astype(np.uint8)).save(
        debug_dir / f"{stem}_table_alpha.png")

    # Dim the RGB so the decision colors remain legible without losing scene
    # context. Red is corrected; cyan is ordinary alpha where both layers are
    # present. Pixels without cross-layer overlap remain as dimmed RGB.
    overlay = np.rint(rgb_u8.astype(np.float32) * 0.35).astype(np.uint8)
    overlay[ordinary_alpha] = (
        0.25 * rgb_u8[ordinary_alpha] + 0.75 * np.array([0, 220, 255])
    ).astype(np.uint8)
    overlay[correction] = (
        0.25 * rgb_u8[correction] + 0.75 * np.array([255, 45, 45])
    ).astype(np.uint8)
    overlay_image = Image.fromarray(overlay)
    canvas = Image.new("RGB", (overlay_image.width, overlay_image.height + 48), "black")
    canvas.paste(overlay_image, (0, 48))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((8, 8, 27, 27), fill=(255, 45, 45))
    draw.text((34, 9), "depth correction", fill="white")
    draw.rectangle((190, 8, 209, 27), fill=(0, 220, 255))
    draw.text((216, 9), "ordinary alpha (both layers)", fill="white")
    draw.text((470, 9), "dim RGB: one layer/background", fill="white")
    canvas.save(debug_dir / f"{stem}_composition_regions.png")

    # Signed depth-gap visualization over +/- the strict far cutoff: blue
    # means table is in front, white is near-equal, red means table is behind.
    depth_limit = max(args.occlusion_depth_high, 1e-9)
    normalized_gap = np.clip(depth_gap_m / depth_limit, -1.0, 1.0)
    gap_rgb = np.zeros_like(rgb_u8)
    positive = normalized_gap >= 0
    gap_rgb[..., 0] = np.where(positive, 255, 255 * (1 + normalized_gap))
    gap_rgb[..., 1] = 255 * (1 - np.abs(normalized_gap))
    gap_rgb[..., 2] = np.where(positive, 255 * (1 - normalized_gap), 255)
    gap_rgb[~valid_depth] = 0
    Image.fromarray(gap_rgb.astype(np.uint8)).save(
        debug_dir / f"{stem}_depth_gap_signed.png")

    np.savez_compressed(
        debug_dir / f"{stem}_compositing_arrays.npz",
        rendered_rgb=rgb,
        foreground_alpha=foreground_alpha,
        table_alpha=table_alpha,
        depth_gap_m=depth_gap_m,
        valid_depth=valid_depth,
        overlap=overlap,
        correction=correction,
        ordinary_alpha=ordinary_alpha,
        far_ordinary=far_ordinary,
        uncertain_ordinary=uncertain_ordinary,
    )
    summary = {
        "step": int(step),
        "resolution": [int(rgb.shape[1]), int(rgb.shape[0])],
        "mode": args.table_occlusion_mode,
        "thresholds": {
            "foreground_alpha_min": args.occlusion_alpha_high,
            "depth_gap_min_m": args.occlusion_depth_epsilon,
            "depth_gap_max_m": args.occlusion_depth_high,
        },
        "pixel_counts": {
            "total": int(overlap.size),
            "cross_layer_overlap": int(overlap.sum()),
            "depth_correction_red": int(correction.sum()),
            "ordinary_alpha_cyan": int(ordinary_alpha.sum()),
            "ordinary_alpha_far": int(far_ordinary.sum()),
            "ordinary_alpha_uncertain_or_boundary": int(uncertain_ordinary.sum()),
        },
        "legend": {
            "red": "strict depth correction",
            "cyan": "ordinary alpha compositing where both layers contribute",
            "dim_rgb": "only one layer or background; no table/foreground decision",
            "depth_gap": "blue=table in front, white=near equal, red=table behind",
        },
    }
    (debug_dir / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")


class RawVideoWriter:
    """Asynchronously stream packed RGB frames to ffmpeg."""

    _STOP = object()

    def __init__(self, ffmpeg_bin, path, width, height, fps, env,
                 preset="slow", queue_size=1):
        self.command = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "warning", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-preset", preset, "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
        ]
        self.env = env
        self.process = None
        self.frames = queue.Queue(maxsize=max(1, int(queue_size)))
        self.error = None
        self.thread = None
        _ACTIVE_VIDEO_WRITERS.add(self)

    def _start(self):
        if self.process is not None:
            return
        self.process = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, env=self.env)
        self.thread = threading.Thread(
            target=self._write_loop, name="ffmpeg-rgb-writer", daemon=True)
        self.thread.start()

    def _write_loop(self):
        while True:
            frame = self.frames.get()
            try:
                if frame is self._STOP:
                    return
                if self.error is None:
                    try:
                        self.process.stdin.write(frame)
                    except (BrokenPipeError, OSError) as error:
                        self.error = error
            finally:
                self.frames.task_done()

    def write(self, frame_bytes):
        self._start()
        if self.error is not None:
            raise RuntimeError(
                "ffmpeg exited while receiving video frames") from self.error
        self.frames.put(frame_bytes)
        if self.error is not None:
            raise RuntimeError(
                "ffmpeg exited while receiving video frames") from self.error

    def close(self):
        if self.process is None:
            _ACTIVE_VIDEO_WRITERS.discard(self)
            return 0
        try:
            self.frames.put(self._STOP)
            self.thread.join()
            if self.process.stdin is not None:
                self.process.stdin.close()
            returncode = self.process.wait()
            if self.error is not None:
                raise RuntimeError(
                    "ffmpeg exited while receiving video frames") from self.error
            return returncode
        finally:
            _ACTIVE_VIDEO_WRITERS.discard(self)

    def abort(self):
        """Best-effort cleanup after a render exception or interruption."""
        try:
            if self.process is None:
                return
            if self.thread is not None and self.thread.is_alive():
                self.frames.put(self._STOP)
                self.thread.join(timeout=5.0)
            if self.process.stdin is not None:
                try:
                    self.process.stdin.close()
                except OSError:
                    pass
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        finally:
            _ACTIVE_VIDEO_WRITERS.discard(self)


def cleanup_video_writers():
    for writer in list(_ACTIVE_VIDEO_WRITERS):
        writer.abort()


def main(argv=None):
    ply_cache_hits_before = _PLY_CACHE_HITS
    ply_cache_misses_before = _PLY_CACHE_MISSES
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", default=None,
        help="Comma-separated trace row indices. Defaults are chosen per trajectory set.",
    )
    parser.add_argument("--full-video", action="store_true")
    parser.add_argument(
        "--delete-video-frames", action="store_true",
        help=("After successful MP4 encoding, delete the temporary PNG frame "
              "directory. Frames are retained if ffmpeg fails."),
    )
    parser.add_argument(
        "--stream-video", action="store_true",
        help="Stream video RGB frames directly to ffmpeg; do not create PNG frames.",
    )
    frame_label = parser.add_mutually_exclusive_group()
    frame_label.add_argument(
        "--frame-label", dest="frame_label", action="store_true",
        help="Draw trajectory/timing text on every output frame (legacy default).",
    )
    frame_label.add_argument(
        "--no-frame-label", dest="frame_label", action="store_false",
        help="Skip PIL text annotation for cleaner and faster dataset videos.",
    )
    parser.set_defaults(frame_label=True)
    parser.add_argument(
        "--ffmpeg-preset", default="slow",
        choices=("ultrafast", "superfast", "veryfast", "faster", "fast",
                 "medium", "slow", "slower", "veryslow"),
        help="libx264 speed/compression preset (legacy default: slow).",
    )
    parser.add_argument(
        "--video-queue-size", type=int, default=1,
        help="Number of encoded-frame buffers used to overlap rendering and ffmpeg.",
    )
    parser.add_argument(
        "--persistent-model-cache", action="store_true",
        help="Reuse PLY arrays, GPU Gaussian tensors and rigid IAS across in-process jobs.",
    )
    parser.add_argument(
        "--geometry-update-mode", choices=("gpu", "cpu"), default="gpu",
        help=("Update rigid Gaussian poses directly on the GPU (default), or "
              "use the legacy NumPy assembly/upload path for validation."),
    )
    parser.add_argument(
        "--physical-gpu-id", type=int, default=None,
        help="Physical GPU identifier recorded in the report by persistent workers.",
    )
    parser.add_argument(
        "--full-video-tail-frames", type=int, default=0,
        help=("With --full-video, render only the final N sampled trajectory "
              "frames while preserving the original environment-rotation phase."),
    )
    parser.add_argument("--env-rotate-count", type=int, default=0)
    parser.add_argument("--env-rotate-axis", choices=("x", "y", "z"), default="y")
    parser.add_argument(
        "--trajectory-env-rotations", type=float, default=0.0,
        help="While rendering --full-video, continuously rotate the environment by this many turns.",
    )
    parser.add_argument(
        "--trajectory-env-deg-per-sec", type=float, default=0.0,
        help="While rendering --full-video, continuously rotate the environment at this angular speed.",
    )
    parser.add_argument("--camera-orbit-count", type=int, default=0)
    parser.add_argument("--camera-name", choices=("main", "contact"), default="main")
    parser.add_argument(
        "--camera-eye-m", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help="Override the selected camera eye in Isaac-world metres.",
    )
    parser.add_argument(
        "--camera-target-m", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help="Override the selected camera target in Isaac-world metres.",
    )
    parser.add_argument(
        "--trajectory-set", choices=("guiji1", "guiji2"), default="guiji1",
        help="Select the archived Isaac trajectory and its audited scene assets.",
    )
    parser.add_argument(
        "--trajectory-dir", default=None,
        help=("Optional custom trajectory package root. It should contain "
              "trajectories/*.csv, configs/execution_contract.json and "
              "provenance/complete_coordinate_transforms.json."),
    )
    parser.add_argument(
        "--trace-csv", default=None,
        help=("Explicit formal trajectory CSV. This is useful for early-stop "
              "episodes whose trace ends in .partial.csv."),
    )
    parser.add_argument(
        "--robot-root", default=None,
        help=("Directory containing one subdirectory per SO101 link. Both "
              "<link>/irgs and legacy <link>/irgs_full layouts are accepted."),
    )
    parser.add_argument(
        "--urdf", default=None,
        help="SO101 URDF path; alternatively set SO101_IRGS_URDF.",
    )
    parser.add_argument(
        "--scene",
        choices=("full", "robot-only", "table-only", "object-only"),
        default="full",
        help=("Render the full scene or isolate the articulated robot, table, "
              "or trajectory object (cup/lego)."),
    )
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument(
        "--fovy-deg", type=float, default=None,
        help=("Vertical camera FOV. By default use contract fovy_deg, derive "
              "it from focal_length_mm with a 36 mm sensor, or fall back to 27 degrees."),
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--diffuse-samples", type=int, default=32)
    parser.add_argument(
        "--light-samples", type=int, default=32,
        help="Environment-importance samples; useful for compact point-like light sources.",
    )
    parser.add_argument(
        "--diffuse-sampling-mode", choices=("uniform", "cosine"),
        default="cosine",
    )
    parser.add_argument(
        "--light-sampling-mode", choices=("iid", "stratified_shared"),
        default="stratified_shared",
    )
    parser.add_argument(
        "--light-type", choices=("env", "point", "area"), default="env",
        help="Use an EXR/HDR environment, a finite point light, or a rectangular area light.",
    )
    parser.add_argument(
        "--analytic-light-samples", type=int, default=64,
        help="Samples per shaded point for area lights; point lights always use one.",
    )
    parser.add_argument(
        "--light-color", type=float, nargs=3, default=(1.0, 1.0, 1.0),
        metavar=("R", "G", "B"), help="Linear RGB color of a point/area light.",
    )
    parser.add_argument(
        "--light-intensity", type=float, default=None,
        help=("Point-light radiant intensity or area-light emitted radiance, "
              "in renderer-linear units. Defaults: point=0.5, area=5.0."),
    )
    parser.add_argument(
        "--trajectory-light-intensity-end", type=float, default=None,
        help=("For a full trajectory with point/area light, linearly change "
              "the intensity from --light-intensity to this value."),
    )
    parser.add_argument(
        "--point-light-position-m", type=float, nargs=3,
        default=(0.30, -0.20, 0.60), metavar=("X", "Y", "Z"),
        help="Point-light position in Isaac world metres.",
    )
    parser.add_argument(
        "--area-light-center-m", type=float, nargs=3,
        default=(0.30, -0.20, 0.60), metavar=("X", "Y", "Z"),
        help="Rectangle center in Isaac world metres.",
    )
    parser.add_argument(
        "--area-light-target-m", type=float, nargs=3,
        default=(0.00, 0.00, 0.10), metavar=("X", "Y", "Z"),
        help="Point toward which the emitting side of the rectangle faces.",
    )
    parser.add_argument(
        "--area-light-up", type=float, nargs=3,
        default=(0.0, 1.0, 0.0), metavar=("X", "Y", "Z"),
        help="Rectangle in-plane up direction in Isaac world axes.",
    )
    parser.add_argument(
        "--area-light-size-m", type=float, nargs=2,
        default=(0.30, 0.20), metavar=("WIDTH", "HEIGHT"),
        help="Rectangle width and height in metres.",
    )
    parser.add_argument(
        "--area-light-two-sided", action="store_true",
        help="Emit from both sides of the rectangle (default: target-facing side only).",
    )
    parser.add_argument("--light-t-min", type=float, default=0.10)
    parser.add_argument(
        "--render-ray-budget", type=int, default=2**24,
        help="Maximum incident-light rays per shading chunk (default: 2^24).",
    )
    parser.add_argument(
        "--bvh-layout",
        choices=("single", "component-ias-identity", "component-ias-rigid"),
        default="single",
        help=("OptiX acceleration layout. 'single' preserves the original "
              "full-scene GAS path; identity IAS validates generic multi-GAS "
              "partitioning; rigid IAS keeps rigid components in local GASes "
              "and updates only their instance transforms."),
    )
    parser.add_argument(
        "--fg-lut-query-layout", choices=("flat", "tiled"), default="tiled",
        help="Layout used for nvdiffrast FG LUT queries (default: tiled).",
    )
    parser.add_argument(
        "--fg-lut-tile-width", type=int, default=2048,
        help="Width of each 2-D FG LUT query grid (default: 2048).",
    )
    parser.add_argument(
        "--disable-indirect", action="store_true",
        help="Keep visibility/shadows but skip secondary-surface indirect lighting.",
    )
    parser.add_argument(
        "--table-z-offset-m", type=float, default=0.0,
        help="Additional table-only vertical offset in Isaac-world metres.",
    )
    parser.add_argument(
        "--table-ply", default=None,
        help="Optional replacement table IRGS PLY (guiji1/guiji2 default is unchanged).",
    )
    parser.add_argument(
        "--object-ply", default=None,
        help="Optional replacement object IRGS PLY (cup/lego/other rigid object).",
    )
    parser.add_argument(
        "--object-name", default=None,
        help="Object label used in reports and provenance lookup (default: cup/lego).",
    )
    parser.add_argument(
        "--table-source-registration", default=None,
        help=("Optional JSON containing T_old_source_from_replacement_source. "
              "The replacement table is registered before the audited table-to-world transform."),
    )
    parser.add_argument(
        "--table-surface-mesh", default=None,
        help="Optional replacement-table mesh used to reject GS far from its reconstructed surface.",
    )
    parser.add_argument(
        "--table-surface-max-distance", type=float, default=0.0,
        help="Maximum source-coordinate GS-to-mesh-vertex distance; 0 disables filtering.",
    )
    parser.add_argument(
        "--base-color-min", type=float, default=0.0,
        help="Global lower bound used to decode stored IRGS base-color logits.",
    )
    parser.add_argument(
        "--uniform-base-color", type=float, nargs=3, default=None,
        metavar=("R", "G", "B"),
        help="Override all visible and indirect-surface base colors (linear RGB).",
    )
    parser.add_argument(
        "--uniform-roughness", type=float, default=None,
        help="Override all visible and indirect-surface roughness values.",
    )
    parser.add_argument(
        "--uniform-metallic", type=float, default=None,
        help="Override all metallic values (only affects metallic-enabled BRDFs).",
    )
    parser.add_argument(
        "--normal-source", choices=("learned", "depth"), default="learned",
        help="Use learned normals or normals reconstructed from rasterized depth.",
    )
    parser.add_argument(
        "--depth-occlude-table", action="store_true",
        help=("Render foreground and table raster layers separately, compare "
              "their camera depths, and harden reliable foreground pixels "
              "where the table is behind them."),
    )
    parser.add_argument(
        "--table-occlusion-mode", choices=("smooth", "hard", "strict"),
        default="smooth",
        help=(
            "Use feathered alpha/depth confidence, the legacy hard cutoff, "
            "or strict binary compositing (only high-confidence pixels are "
            "hardened)."
        ),
    )
    parser.add_argument(
        "--depth-compositing-render-mode",
        choices=("legacy-two-pass", "selective"), default="legacy-two-pass",
        help=("Use the validated fully shaded foreground/table passes "
              "(default), or the experimental selective-correction path."),
    )
    parser.add_argument(
        "--save-compositing-debug", action="store_true",
        help=(
            "Save per-frame RGB, alpha/depth maps, decision overlay and raw "
            "arrays under <out>/compositing_debug."
        ),
    )
    parser.add_argument("--object-alpha-threshold", type=float, default=0.10)
    parser.add_argument(
        "--occlusion-depth-epsilon", type=float, default=0.001,
        help="Hard-mode minimum foreground/table depth separation in metres.",
    )
    parser.add_argument("--occlusion-alpha-low", type=float, default=0.70)
    parser.add_argument("--occlusion-alpha-high", type=float, default=0.95)
    parser.add_argument(
        "--occlusion-depth-low", type=float, default=0.005,
        help="Smooth-mode lower depth separation in Isaac-world metres.",
    )
    parser.add_argument(
        "--occlusion-depth-high", type=float, default=0.015,
        help=(
            "Smooth-mode upper transition and strict-mode far-distance cutoff "
            "in Isaac-world metres."
        ),
    )
    parser.add_argument(
        "--occlusion-boundary-protection", type=int, default=0,
        help="Erode depth-correction confidence by this many pixels.",
    )
    parser.add_argument(
        "--gaussian-scale-multiplier", type=float, default=1.0,
        help="Diagnostic multiplier applied to every Gaussian scale.",
    )
    parser.add_argument(
        "--material-variant-name", default="baseline",
        help="Label recorded in the output report for material/geometry ablations.",
    )
    parser.add_argument(
        "--output-buffer",
        choices=("render", "base_color", "roughness", "metallic", "rend_normal", "render_sh"),
        default="render",
        help="Diagnostic image buffer to save; default keeps the relit RGB result.",
    )
    parser.add_argument("--envmap", default=str(ROOT / "assets/env_map/envmap6.exr"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--profile-timing", action="store_true",
        help=("Record synchronized CUDA timings for rasterization, G-buffer/3D "
              "reconstruction, DS/LS sampling, environment lookup, BVH trace, "
              "indirect light, BRDF integration, and final composition."),
    )
    parser.add_argument(
        "--profile-warmup", type=int, default=1,
        help="Untimed warm-up renders before each profiled frame (default: 1).",
    )
    parser.add_argument(
        "--save-assembled-ply", default=None,
        help="Save the first selected dynamic scene as one IRGS/3DGS PLY.",
    )
    parser.add_argument(
        "--save-only", action="store_true",
        help="Exit after --save-assembled-ply without loading lighting or rendering.",
    )
    args = parser.parse_args(argv)
    startup_timings = {}

    if args.light_intensity is None and args.light_type != "env":
        args.light_intensity = 0.5 if args.light_type == "point" else 5.0
    if args.light_intensity is not None and args.light_intensity < 0:
        raise ValueError("--light-intensity must be non-negative")
    if any(component < 0 for component in args.light_color):
        raise ValueError("--light-color components must be non-negative")
    if args.uniform_roughness is not None and not 0.0 <= args.uniform_roughness <= 1.0:
        raise ValueError("--uniform-roughness must be in [0, 1]")
    if args.uniform_metallic is not None and not 0.0 <= args.uniform_metallic <= 1.0:
        raise ValueError("--uniform-metallic must be in [0, 1]")
    if args.gaussian_scale_multiplier <= 0.0:
        raise ValueError("--gaussian-scale-multiplier must be positive")
    if args.analytic_light_samples < 1:
        raise ValueError("--analytic-light-samples must be at least 1")
    if any(size <= 0 for size in args.area_light_size_m):
        raise ValueError("--area-light-size-m values must be positive")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.video_queue_size <= 0:
        raise ValueError("--video-queue-size must be positive")
    if args.depth_occlude_table and args.scene != "full":
        raise ValueError("--depth-occlude-table requires --scene full")
    if args.depth_occlude_table and args.output_buffer != "render":
        raise ValueError("--depth-occlude-table currently supports --output-buffer render")
    if args.depth_occlude_table and args.profile_timing:
        raise ValueError(
            "--depth-occlude-table and --profile-timing cannot be combined yet; "
            "the latter currently describes one render pass")
    if not 0.0 < args.object_alpha_threshold < 1.0:
        raise ValueError("--object-alpha-threshold must be between 0 and 1")
    if args.occlusion_depth_epsilon < 0:
        raise ValueError("--occlusion-depth-epsilon must be non-negative")
    if not 0 <= args.occlusion_alpha_low < args.occlusion_alpha_high <= 1:
        raise ValueError("occlusion alpha bounds must satisfy 0 <= low < high <= 1")
    if not 0 <= args.occlusion_depth_low < args.occlusion_depth_high:
        raise ValueError("occlusion depth bounds must satisfy 0 <= low < high")
    if args.occlusion_boundary_protection < 0:
        raise ValueError("--occlusion-boundary-protection must be non-negative")

    custom_trace_root = Path(args.trajectory_dir).expanduser().resolve() if args.trajectory_dir else None
    if custom_trace_root is not None:
        trace_root = custom_trace_root
        trace_path = resolve_trajectory_csv(trace_root, args.trace_csv)
        contract_path = trace_root / "configs/execution_contract.json"
        dossier_path = trace_root / "provenance/complete_coordinate_transforms.json"
        audit_path = trace_root / "results/asset_conversion_audit.json"
        table_ply_path = TABLE_PLY
        object_ply_path = CUP_PLY
        # Most V32-style packages call the carried item ``cup``; callers can
        # override this with --object-name (for example, lego).
        object_name = args.object_name or "cup"
    elif args.trajectory_set == "guiji2":
        trace_root = ROOT / "dataset/guiji2"
        trace_path = resolve_trajectory_csv(trace_root, args.trace_csv)
        contract_path = trace_root / "configs/execution_contract.json"
        dossier_path = trace_root / "provenance/complete_coordinate_transforms.json"
        audit_path = trace_root / "results/asset_conversion_audit.json"
        table_ply_path = ROOT / "outputs_syn4_no_pseudo_t005/Synthetic4Relight/wide_four_corner_table_full_sphere_close/irgs/point_cloud/iteration_20000/point_cloud.ply"
        object_ply_path = ROOT / "outputs_syn4_no_pseudo_t005/Synthetic4Relight/lego/irgs/point_cloud/iteration_20000/point_cloud.ply"
        object_name = "lego"
    else:
        trace_root = TRACE_ROOT
        trace_path = resolve_trajectory_csv(trace_root, args.trace_csv or TRACE)
        contract_path = CONTRACT
        dossier_path = TRANSFORM_DOSSIER
        audit_path = None
        table_ply_path = TABLE_PLY
        object_ply_path = CUP_PLY
        object_name = "cup"

    if args.object_ply:
        object_ply_path = resolve_ply_path(args.object_ply)

    if args.table_ply:
        table_ply_path = resolve_ply_path(args.table_ply)
    elif custom_trace_root is None:
        table_ply_path = resolve_ply_path(table_ply_path)
    if custom_trace_root is None or args.object_ply:
        object_ply_path = resolve_ply_path(object_ply_path)
    if args.env_rotate_count and args.camera_orbit_count:
        raise ValueError("environment rotation and camera orbit are separate render modes")
    trajectory_env_moves = bool(args.trajectory_env_rotations or args.trajectory_env_deg_per_sec)
    if trajectory_env_moves and not args.full_video:
        raise ValueError("trajectory environment rotation requires --full-video")
    if args.trajectory_env_rotations and args.trajectory_env_deg_per_sec:
        raise ValueError("choose either trajectory rotations or angular speed, not both")
    if trajectory_env_moves and (args.env_rotate_count or args.camera_orbit_count):
        raise ValueError("trajectory environment rotation cannot be combined with fixed-scene rotation modes")
    out = Path(args.out)
    if args.env_rotate_count:
        frames_dir = out / "envrotate_frames"
    elif args.camera_orbit_count:
        frames_dir = out / "camera_orbit_frames"
    else:
        frames_dir = out / ("video_frames" if args.full_video else "preview_frames")
    video_mode = bool(args.full_video or args.env_rotate_count or args.camera_orbit_count)
    out.mkdir(parents=True, exist_ok=True)
    if args.stream_video and not video_mode:
        raise ValueError("--stream-video requires a video render mode")
    if not args.stream_video:
        frames_dir.mkdir(parents=True, exist_ok=True)
    stage_start = time.perf_counter()
    contract = json.loads(contract_path.read_text())
    transform_dossier = json.loads(dossier_path.read_text())
    if custom_trace_root is not None:
        if not args.object_name:
            asset_path = str(contract.get("cup_visual_asset_path", ""))
            path_tokens = {token.lower() for token in Path(asset_path).parts}
            candidates = [key for key in transform_dossier.keys()
                          if key not in ("table", "robot")]
            preferred = [key for key in candidates if key.lower() in path_tokens]
            object_name = (preferred[0] if preferred else
                           (candidates[0] if len(candidates) == 1 else "cup"))
        else:
            object_name = args.object_name
        if not args.table_ply:
            declared_table = contract_output_path(contract, "table", trace_root)
            if declared_table is not None:
                table_ply_path = declared_table
        if not args.object_ply:
            declared_object = contract_output_path(contract, "object", trace_root)
            if declared_object is not None:
                object_ply_path = declared_object
        if not args.table_ply and not contract_output_path(contract, "table", trace_root):
            table_ply_path = resolve_ply_path(table_ply_path)
        if not args.object_ply and not contract_output_path(contract, "object", trace_root):
            object_ply_path = resolve_ply_path(object_ply_path)
    with trace_path.open(newline="") as trace_stream:
        trace_reader = csv.DictReader(trace_stream)
        fieldnames = set(trace_reader.fieldnames or ())
        required_trace_fields = {
            "step", "sim_time_s", "stage",
            "object_x_m", "object_y_m", "object_z_m",
            "object_qw", "object_qx", "object_qy", "object_qz",
            *(f"{column}__actual_q_rad" for column in JOINT_COLUMN.values()),
        }
        missing_trace_fields = sorted(required_trace_fields - fieldnames)
        if missing_trace_fields:
            raise ValueError(
                f"轨迹 CSV 缺少必要字段 {missing_trace_fields}: {trace_path}")
        rows = list(trace_reader)
    if not rows:
        raise ValueError(f"轨迹 CSV 没有数据行: {trace_path}")
    if args.steps is None:
        if custom_trace_root is not None:
            preview_indices = np.linspace(
                0, len(rows) - 1, min(6, len(rows)), dtype=np.int64)
            args.steps = ",".join(str(index) for index in np.unique(preview_indices))
        else:
            args.steps = (
                "0,455,585,765,1256,1490" if args.trajectory_set == "guiji2"
                else "0,455,585,765,1256,1495"
            )

    # Load canonical Gaussians.  Bounds are from the exact GLBs supplied to Isaac.
    if args.trajectory_set == "guiji2":
        asset_audit = json.loads(audit_path.read_text())
        table_bounds = np.asarray(asset_audit["table"]["source_bounds"], dtype=np.float64)
        cup_bounds = np.asarray(asset_audit["lego"]["source_bounds"], dtype=np.float64)
    elif custom_trace_root is not None and audit_path.exists():
        asset_audit = json.loads(audit_path.read_text())
        table_bounds = np.asarray(asset_audit["table"]["source_bounds"], dtype=np.float64)
        object_key = object_name or next(
            (key for key in asset_audit if key not in ("table", "robot")), None
        )
        cup_bounds = (np.asarray(asset_audit[object_key]["source_bounds"], dtype=np.float64)
                      if object_key else None)
    else:
        if custom_trace_root is not None:
            table_bounds = None
            cup_bounds = None
        else:
            table_alignment = json.loads(TABLE_ALIGNMENT.read_text())
            table_bounds = np.asarray(table_alignment["source_bounds_m"], dtype=np.float64)
            cup_bounds = np.asarray(contract.get("cup_source_bounds_m", [[-1e9] * 3, [1e9] * 3]), dtype=np.float64)
    startup_timings["metadata_and_trajectory_load_s"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    table_registration = None
    if args.table_source_registration:
        registration_path = Path(args.table_source_registration).expanduser().resolve()
        registration_data = json.loads(registration_path.read_text())
        table_registration = np.asarray(
            registration_data["T_old_source_from_replacement_source"],
            dtype=np.float64,
        )
        table, _ = load_irgs_ply(table_ply_path)
        table_rejected_surface = 0
        if args.table_surface_mesh and args.table_surface_max_distance > 0:
            surface_mesh = trimesh.load(
                str(Path(args.table_surface_mesh).expanduser().resolve()),
                force="mesh", process=False,
            )
            distances = cKDTree(np.asarray(surface_mesh.vertices)).query(
                table["xyz"], k=1, workers=-1
            )[0]
            surface_keep = distances <= args.table_surface_max_distance
            table_rejected_surface = int((~surface_keep).sum())
            table = subset(table, surface_keep)
        table = affine_transform_surfels(
            table, table_registration[:3, :3], table_registration[:3, 3]
        )
        lo, hi = table_bounds
        keep = ((table["xyz"] >= lo - 0.02) &
                (table["xyz"] <= hi + 0.02)).all(axis=1)
        table_rejected = table_rejected_surface + int((~keep).sum())
        table = subset(table, keep)
    else:
        table, table_rejected = load_irgs_ply(
            table_ply_path, table_bounds, margin=0.02)
    cup, cup_rejected = load_irgs_ply(object_ply_path, cup_bounds, margin=0.01)
    robot_root = Path(args.robot_root).expanduser().resolve() if args.robot_root else ROBOT_ROOT.resolve()
    robot = {}
    robot_rejected = 0
    for link in LINKS:
        robot[link], rejected = load_irgs_ply(resolve_ply_path(robot_root / link))
        robot_rejected += rejected
    startup_timings["gaussian_ply_loading_s"] = time.perf_counter() - stage_start

    # Exact column-vector source transforms from the V32 provenance dossier.
    stage_start = time.perf_counter()
    table_cfg = contract["base_visual_transform"]
    table_entry = transform_dossier.get("table", {})
    if args.trajectory_set == "guiji2":
        table_transform_key = "T_world_from_source"
    else:
        table_transform_key = next(
            (key for key in ("T_canonical_from_source", "T_canonical_from_source_column_vector",
                             "T_world_from_source", "T_v32_visual_local_from_source")
             if key in table_entry),
            "T_canonical_from_source",
        )
    table_transform = np.asarray(
        transform_dossier["table"][table_transform_key], dtype=np.float64
    )
    if args.trajectory_set == "guiji2":
        # guiji2's audited table map is translation-only.  Preserve the raw
        # scale/rotation parameterization byte-for-byte instead of running an
        # unnecessary SVD, because external GS viewers may not recognize an
        # equivalent swapped-axis surfel representation.
        table_world = {key: value.copy() for key, value in table.items()}
        table_world["xyz"] = SCENE_SCALE * (
            table["xyz"] @ table_transform[:3, :3].T + table_transform[:3, 3]
        )
        table_world["scale"] = table["scale"] + math.log(SCENE_SCALE)
    else:
        table_affine_cache_key = None
        if args.persistent_model_cache:
            table_affine_cache_key = (
                "table_world",
                path_identity(table_ply_path),
                tuple((SCENE_SCALE * table_transform[:3, :3]).reshape(-1)),
                tuple(SCENE_SCALE * table_transform[:3, 3]),
            )
        table_world = affine_transform_surfels(
            table,
            SCENE_SCALE * table_transform[:3, :3],
            SCENE_SCALE * table_transform[:3, 3],
            cache_key=table_affine_cache_key,
        )
        # V32/guiji3 stores the canonical table asset at its own origin and
        # places that asset in Isaac world with the contract translation
        # (the tabletop centre is at y=0.2650608656 m).  The provenance
        # matrix above only converts source reconstruction coordinates to
        # canonical coordinates; it does not include this scene placement.
        # Apply it here, otherwise the table is left near the robot origin.
        if custom_trace_root is not None:
            contract_translation = np.asarray(
                table_cfg.get("translation_world_m", [0.0, 0.0, 0.0]),
                dtype=np.float64,
            )
            table_world["xyz"] = (
                table_world["xyz"] + SCENE_SCALE * contract_translation)
    if args.table_z_offset_m:
        table_world["xyz"] = table_world["xyz"].copy()
        table_world["xyz"][:, 2] += SCENE_SCALE * args.table_z_offset_m

    # Exact source reconstruction -> V32 visible local transform.  This matrix
    # includes PCA lid-up canonicalization, uniform scale to 80 mm, XY
    # centering, bottom Z=0, USD numeric x100/runtime x0.01 cancellation, and
    # the visual-local translation [0, 0, -0.04] m.
    if args.trajectory_set == "guiji2":
        cup_dimensions = np.asarray(
            transform_dossier["lego"]["canonical_bounds_m"], dtype=np.float64
        )
        cup_dimensions = cup_dimensions[1] - cup_dimensions[0]
        cup_transform = np.asarray(
            transform_dossier["lego"]["T_canonical_from_source"], dtype=np.float64
        )
        cup_local_shift = np.zeros(3, dtype=np.float64)
    elif custom_trace_root is not None:
        object_key = object_name
        object_entry = transform_dossier.get(object_key, {})
        canonical_bounds = object_entry.get(
            "canonical_bounds_m", object_entry.get("target_bounds_m")
        )
        if canonical_bounds is not None:
            cup_dimensions = np.asarray(canonical_bounds, dtype=np.float64)
            cup_dimensions = cup_dimensions[1] - cup_dimensions[0]
        else:
            # Some V32 dossiers record extents directly under ``fit`` rather
            # than as a min/max bounds pair.
            cup_dimensions = np.asarray(
                object_entry.get("fit", {}).get(
                    "canonical_extents_m", [0.0, 0.0, 0.0]
                ),
                dtype=np.float64,
            )
        transform_key = next(
            (key for key in ("T_v32_visual_local_from_source", "T_canonical_from_source",
                             "T_canonical_from_source_column_vector", "T_world_from_source")
             if key in object_entry),
            None,
        )
        if transform_key is None:
            raise KeyError(f"对象 {object_key!r} 的 provenance 中没有可用变换矩阵")
        cup_transform = np.asarray(object_entry[transform_key], dtype=np.float64)
        cup_local_shift = np.zeros(3, dtype=np.float64)
    else:
        cup_dimensions = np.asarray(contract["cylinder"]["dimensions_m"], dtype=np.float64)
        cup_transform = np.asarray(
            transform_dossier["cup"]["T_v32_visual_local_from_source"],
            dtype=np.float64,
        )
        cup_local_shift = np.asarray(contract["cup_visual_transform"]["translation_local_m"])
    if args.trajectory_set == "guiji2":
        object_uniform_scale = float(np.linalg.norm(cup_transform[:3, 0]))
        object_rotation = cup_transform[:3, :3] / object_uniform_scale
        cup_local = rigid_transform(
            cup,
            object_rotation,
            SCENE_SCALE * cup_transform[:3, 3],
            uniform_scale=SCENE_SCALE * object_uniform_scale,
        )
    else:
        cup_affine_cache_key = None
        if args.persistent_model_cache:
            cup_affine_cache_key = (
                "object_local",
                path_identity(object_ply_path),
                tuple((SCENE_SCALE * cup_transform[:3, :3]).reshape(-1)),
                tuple(SCENE_SCALE * cup_transform[:3, 3]),
            )
        cup_local = affine_transform_surfels(
            cup,
            SCENE_SCALE * cup_transform[:3, :3],
            SCENE_SCALE * cup_transform[:3, 3],
            cache_key=cup_affine_cache_key,
        )

    urdf_path = resolve_urdf_path(args.urdf)
    joints = parse_urdf(urdf_path)
    world_from_base = robot_world_from_base(contract)

    def object_world_transform(row):
        """Rigid object-local -> enlarged IRGS-world transform."""
        object_position = [float(row[f"object_{axis}_m"]) for axis in "xyz"]
        object_quaternion = [float(row[f"object_q{axis}"]) for axis in "wxyz"]
        transform = pose_matrix(object_position, object_quaternion)
        transform[:3, 3] *= SCENE_SCALE
        return transform

    def dynamic_parts(row):
        angles = {joint: float(row[f"{column}__actual_q_rad"]) for joint, column in JOINT_COLUMN.items()}
        link_fk = fk(joints, angles)
        parts = [table_world] if args.scene in ("full", "table-only") else []
        if args.scene in ("full", "robot-only"):
            for link in LINKS:
                world_from_link = world_from_base @ link_fk[link]
                parts.append(
                    rigid_transform(
                        robot[link],
                        world_from_link[:3, :3],
                        SCENE_SCALE * world_from_link[:3, 3],
                    )
                )
        if args.scene in ("full", "object-only"):
            world_from_cup = object_world_transform(row)
            parts.append(
                rigid_transform(
                    cup_local,
                    world_from_cup[:3, :3],
                    world_from_cup[:3, 3],
                )
            )
        return concatenate(parts)

    table_geometry = {key: table_world[key] for key in ("xyz", "scale", "rot")}

    def dynamic_geometry(row):
        """Fast per-frame path: materials/SH stay resident on the GPU."""
        angles = {joint: float(row[f"{column}__actual_q_rad"]) for joint, column in JOINT_COLUMN.items()}
        link_fk = fk(joints, angles)
        parts = [table_geometry] if args.scene in ("full", "table-only") else []
        if args.scene in ("full", "robot-only"):
            for link in LINKS:
                world_from_link = world_from_base @ link_fk[link]
                parts.append(
                    rigid_geometry(
                        robot[link],
                        world_from_link[:3, :3],
                        SCENE_SCALE * world_from_link[:3, 3],
                    )
                )
        if args.scene in ("full", "object-only"):
            world_from_cup = object_world_transform(row)
            parts.append(
                rigid_geometry(
                    cup_local,
                    world_from_cup[:3, :3],
                    world_from_cup[:3, 3],
                )
            )
        return concatenate_geometry(parts)

    trajectory_frame_offset = 0
    if args.env_rotate_count or args.camera_orbit_count:
        requested = [int(x) for x in args.steps.split(",") if x.strip()]
        if len(requested) != 1:
            raise ValueError("rotation modes require exactly one source step in --steps")
        if not -len(rows) <= requested[0] < len(rows):
            raise IndexError(f"trajectory row index out of range: {requested[0]}")
        selected = [rows[requested[0]]] * (args.env_rotate_count or args.camera_orbit_count)
    elif args.full_video:
        full_selected = rows[:: args.stride]
        if full_selected[-1]["step"] != rows[-1]["step"]:
            full_selected.append(rows[-1])
        if args.full_video_tail_frames < 0:
            raise ValueError("--full-video-tail-frames must be non-negative")
        if args.full_video_tail_frames:
            selected = full_selected[-args.full_video_tail_frames:]
            trajectory_frame_offset = len(full_selected) - len(selected)
        else:
            selected = full_selected
    else:
        requested = [int(x) for x in args.steps.split(",") if x.strip()]
        invalid = [index for index in requested if not -len(rows) <= index < len(rows)]
        if invalid:
            raise IndexError(f"trajectory row indices out of range: {invalid}")
        selected = [rows[index] for index in requested]

    video_path = None
    if video_mode:
        if args.env_rotate_count:
            video_path = out / f"step_{int(selected[0]['step']):04d}_envrotate_irgs.mp4"
        elif args.camera_orbit_count:
            video_path = out / f"step_{int(selected[0]['step']):04d}_camera_orbit_irgs.mp4"
        elif args.camera_name == "contact":
            video_path = out / f"so101_table_{object_name}_trajectory_contact_camera_irgs.mp4"
        else:
            video_path = out / f"so101_table_{object_name}_trajectory_irgs.mp4"

    model_cache_key = None
    if args.persistent_model_cache:
        model_cache_key = persistent_asset_key(
            args,
            table_ply_path,
            object_ply_path,
            robot_root,
            urdf_path,
            table_bounds,
            cup_bounds,
            table_transform,
            cup_transform,
            world_from_base,
            (len(table_world["xyz"]),
             *(len(robot[link]["xyz"]) for link in LINKS),
             len(cup_local["xyz"])),
        )
    # On a cache hit all material/SH tensors are already resident. Assemble
    # only the three dynamic geometry arrays needed for the first frame.
    first_scene = (
        dynamic_geometry(selected[0])
        if model_cache_key in _PERSISTENT_MODEL_CACHE
        else dynamic_parts(selected[0])
    )
    if args.gaussian_scale_multiplier != 1.0:
        first_scene["scale"] = first_scene["scale"] + math.log(
            args.gaussian_scale_multiplier)

    # Scene-specific loading ends here.  From this point down the IAS code
    # sees only ordered generic components and their motion type; no asset name
    # is interpreted by the CUDA/OptiX implementation.
    component_plan = []
    component_cursor = 0

    def append_component(label, motion_type, count, transform_fn=None,
                         local_geometry=None):
        nonlocal component_cursor
        count = int(count)
        component_plan.append({
            "label": label,
            "motion_type": motion_type,
            "begin": component_cursor,
            "end": component_cursor + count,
            "transform_fn": transform_fn,
            "local_geometry": local_geometry,
        })
        component_cursor += count

    if args.scene in ("full", "table-only"):
        append_component("scene_static_0", "static", len(table_world["xyz"]))
    if args.scene in ("full", "robot-only"):
        for link_index, link in enumerate(LINKS):
            append_component(
                f"articulated_rigid_{link_index}", "rigid_instance",
                len(robot[link]["xyz"]),
                transform_fn=lambda context, link=link: context["links"][link],
                local_geometry=robot[link],
            )
    if args.scene in ("full", "object-only"):
        append_component(
            "trajectory_rigid_0", "rigid_instance", len(cup_local["xyz"]),
            transform_fn=lambda context: context["object"],
            local_geometry=cup_local,
        )
    if component_cursor != len(first_scene["xyz"]):
        raise RuntimeError(
            "component ranges do not match assembled Gaussian ordering")
    startup_timings["coordinate_transform_and_scene_assembly_s"] = (
        time.perf_counter() - stage_start)

    stage_start = time.perf_counter()
    model = make_gaussian_model(
        first_scene,
        base_color_min=args.base_color_min,
        persistent_key=model_cache_key,
        # Rigid component geometry is refreshed below directly on the GPU.
        # Avoid uploading a full CPU-assembled scene on persistent cache hits.
        update_cached_geometry=args.geometry_update_mode == "cpu",
    )
    persistent_model_reused = bool(model._persistent_cache_reused)
    torch.cuda.synchronize()
    startup_timings["gaussian_gpu_upload_s"] = time.perf_counter() - stage_start

    def component_motion_context(row):
        angles = {
            joint: float(row[f"{column}__actual_q_rad"])
            for joint, column in JOINT_COLUMN.items()
        }
        link_fk = fk(joints, angles)
        link_transforms = {}
        for link in LINKS:
            transform = world_from_base @ link_fk[link]
            transform = transform.copy()
            transform[:3, 3] *= SCENE_SCALE
            link_transforms[link] = transform
        return {
            "links": link_transforms,
            "object": object_world_transform(row),
        }

    def component_transforms_from_context(context):
        transforms = []
        for component in component_plan:
            if (args.bvh_layout == "component-ias-rigid" and
                    component["motion_type"] == "rigid_instance"):
                transform = component["transform_fn"](context)[:3, :4]
            else:
                transform = np.eye(4, dtype=np.float64)[:3]
            transforms.append(transform)
        return np.stack(transforms, axis=0).astype(np.float32)

    def component_transforms(row):
        return component_transforms_from_context(component_motion_context(row))

    def build_component_bvh(row, fixed_geometry):
        transforms = component_transforms(row)
        components = []
        for component_index, component in enumerate(component_plan):
            motion_type = component["motion_type"]
            static = bool(
                fixed_geometry or motion_type == "static" or
                (args.bvh_layout == "component-ias-rigid" and
                 motion_type == "rigid_instance")
            )
            entry = {
                "begin": component["begin"],
                "end": component["end"],
                "static": static,
                "transform": transforms[component_index],
            }
            if (args.bvh_layout == "component-ias-rigid" and
                    motion_type == "rigid_instance"):
                world_vertices, _, _ = model.get_boundings_range(
                    component["begin"], component["end"],
                    alpha_min=model.alpha_min,
                )
                transform = torch.as_tensor(
                    transforms[component_index], dtype=world_vertices.dtype,
                    device=world_vertices.device,
                )
                # OptiX stores the rigid proxy in component-local coordinates;
                # the rasterizer and exact alpha integration remain in world
                # coordinates in the ordinary concatenated GaussianModel.
                entry["proxy_vertices"] = (
                    world_vertices - transform[:, 3]
                ) @ transform[:, :3]
            components.append(entry)
        model.build_component_ias(components)

    def update_component_bvh(transforms):
        for component_index, component in enumerate(component_plan):
            motion_type = component["motion_type"]
            needs_geometry_refit = (
                motion_type == "dynamic_geometry" or
                (motion_type == "rigid_instance" and
                 args.bvh_layout == "component-ias-identity")
            )
            if needs_geometry_refit:
                model.update_component_gas(
                    component_index, component["begin"], component["end"])
        # Also refreshes GAS handles in the IAS after any refit; for rigid
        # components this is the only per-frame acceleration-structure update.
        model.update_component_transforms(transforms)

    component_signature = (
        model_cache_key,
        args.bvh_layout,
        tuple(
            (component["begin"], component["end"], component["motion_type"])
            for component in component_plan
        ),
    )

    # Keep component-local xyz/quaternions on the GPU. All moving scene
    # components in the SO101 handoff are rigid, so rebuilding transformed
    # NumPy arrays and uploading the complete scene every frame is unnecessary.
    rigid_source_signature = (
        model_cache_key,
        tuple(
            (component["begin"], component["end"],
             component["local_geometry"] is not None)
            for component in component_plan
        ),
    )
    if getattr(model, "_persistent_rigid_source_signature", None) != \
            rigid_source_signature:
        rigid_gpu_sources = []
        for component in component_plan:
            local_geometry = component["local_geometry"]
            if local_geometry is None:
                rigid_gpu_sources.append(None)
                continue
            rigid_gpu_sources.append({
                "xyz": torch.as_tensor(
                    local_geometry["xyz"], dtype=torch.float64,
                    device="cuda").contiguous(),
                "rot": torch.as_tensor(
                    local_geometry["rot"], dtype=torch.float64,
                    device="cuda").contiguous(),
            })
        model._persistent_rigid_gpu_sources = rigid_gpu_sources
        model._persistent_rigid_source_signature = rigid_source_signature
    else:
        rigid_gpu_sources = model._persistent_rigid_gpu_sources

    @torch.no_grad()
    def update_raster_geometry_on_gpu(transforms):
        """Apply rigid component poses without full-scene CPU assembly/upload."""
        for component, source, transform_np in zip(
                component_plan, rigid_gpu_sources, transforms):
            if source is None:
                continue
            begin, end = component["begin"], component["end"]
            transform = torch.as_tensor(
                transform_np, dtype=source["xyz"].dtype, device="cuda")
            rotation = transform[:, :3]
            translation = transform[:, 3]
            model._xyz.data[begin:end].copy_(
                source["xyz"] @ rotation.T + translation)

            q = torch.as_tensor(
                matrix_to_quat_wxyz(transform_np[:, :3]),
                dtype=source["rot"].dtype, device="cuda")
            local_q = source["rot"]
            qw, qx, qy, qz = q.unbind()
            lw, lx, ly, lz = local_q.unbind(dim=-1)
            world_q = torch.stack((
                qw * lw - qx * lx - qy * ly - qz * lz,
                qw * lx + qx * lw + qy * lz - qz * ly,
                qw * ly - qx * lz + qy * lw + qz * lx,
                qw * lz + qx * ly - qy * lx + qz * lw,
            ), dim=-1)
            model._rotation.data[begin:end].copy_(
                F.normalize(world_q, dim=-1))
        model.invalidate_trace_cache()

    persistent_ias_reused = False
    if args.save_assembled_ply:
        assembled_ply = Path(args.save_assembled_ply).expanduser().resolve()
        model.save_ply(str(assembled_ply))
        print(f"Saved assembled PLY: {assembled_ply}", flush=True)
        if args.save_only:
            return
    stage_start = time.perf_counter()
    environment = None
    model.direct_light = None
    if args.light_type == "env":
        environment = EnvLight(
            path=args.envmap, device="cuda", max_res=1024, activation="none"
        ).cuda()
        environment.build_mips()
        environment.update_pdf()
        environment.set_transform(torch.eye(3, device="cuda"))
        model.env_map = environment
        active_light = environment
    elif args.light_type == "point":
        # Geometry is enlarged by SCENE_SCALE.  Multiplying radiant intensity
        # by scale^2 makes I/r_scene^2 numerically equal to I/r_metres^2.
        model.env_map = None
        model.direct_light = PointLight(
            position=SCENE_SCALE * np.asarray(args.point_light_position_m),
            intensity=(SCENE_SCALE ** 2) * args.light_intensity
            * np.asarray(args.light_color),
            device="cuda",
        ).cuda()
        active_light = model.direct_light
    else:
        center_m = np.asarray(args.area_light_center_m, dtype=np.float64)
        target_m = np.asarray(args.area_light_target_m, dtype=np.float64)
        normal = target_m - center_m
        if np.linalg.norm(normal) < 1e-9:
            raise ValueError("--area-light-center-m and --area-light-target-m must differ")
        normal /= np.linalg.norm(normal)
        model.env_map = None
        model.direct_light = RectAreaLight(
            center=SCENE_SCALE * center_m,
            normal=normal,
            up=args.area_light_up,
            size=SCENE_SCALE * np.asarray(args.area_light_size_m),
            radiance=args.light_intensity * np.asarray(args.light_color),
            two_sided=args.area_light_two_sided,
            device="cuda",
        ).cuda()
        active_light = model.direct_light
    torch.cuda.synchronize()
    startup_timings["lighting_preprocess_s"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    pipeline_parser = argparse.ArgumentParser()
    pp = PipelineParams(pipeline_parser)
    pipe = pp.extract(pipeline_parser.parse_args([]))
    pipe.diffuse_sample_num = args.diffuse_samples
    pipe.light_sample_num = args.light_samples
    pipe.diffuse_sampling_mode = args.diffuse_sampling_mode
    pipe.light_sampling_mode = args.light_sampling_mode
    pipe.light_t_min = args.light_t_min
    pipe.render_ray_budget = args.render_ray_budget
    pipe.fg_lut_query_layout = args.fg_lut_query_layout
    pipe.fg_lut_tile_width = args.fg_lut_tile_width
    pipe.wo_indirect_relight = args.disable_indirect
    pipe.analytic_light_sample_num = args.analytic_light_samples

    camera_key = "camera" if args.camera_name == "main" else "contact_camera"
    camera_cfg = contract[camera_key]
    eye_m = args.camera_eye_m if args.camera_eye_m is not None else camera_cfg["eye_m"]
    target_m = args.camera_target_m if args.camera_target_m is not None else camera_cfg["target_m"]
    base_eye = SCENE_SCALE * np.asarray(eye_m, dtype=np.float64)
    camera_target = SCENE_SCALE * np.asarray(target_m, dtype=np.float64)
    camera_up = np.asarray(camera_cfg["up"], dtype=np.float64)
    if args.fovy_deg is not None:
        camera_fovy_deg = args.fovy_deg
        camera_fovy_source = "command_line"
    elif camera_cfg.get("fovy_deg") is not None:
        camera_fovy_deg = float(camera_cfg["fovy_deg"])
        camera_fovy_source = "contract_fovy_deg"
    elif camera_cfg.get("focal_length_mm") is not None:
        sensor_width_mm = float(camera_cfg.get("sensor_width_mm", 36.0))
        vertical_aperture_mm = float(camera_cfg.get(
            "vertical_aperture_mm", sensor_width_mm * args.height / args.width))
        camera_fovy_deg = math.degrees(2.0 * math.atan(
            vertical_aperture_mm / (2.0 * float(camera_cfg["focal_length_mm"]))))
        camera_fovy_source = "contract_focal_length_assuming_36mm_sensor"
    else:
        camera_fovy_deg = 27.0
        camera_fovy_source = "legacy_default"

    def make_camera(orbit_angle_deg=0.0):
        eye_offset = base_eye - camera_target
        if orbit_angle_deg:
            eye_offset = Rotation.from_rotvec(
                math.radians(orbit_angle_deg) * np.array([0.0, 0.0, 1.0])
            ).apply(eye_offset)
        c2w = look_at_colmap(camera_target + eye_offset, camera_target, camera_up)
        return SceneCamera(c2w, args.width, args.height, math.radians(camera_fovy_deg))

    camera = make_camera()
    background = torch.ones(3, dtype=torch.float32, device="cuda")
    foreground_background = torch.zeros(3, dtype=torch.float32, device="cuda")
    video_writer = None
    video_frames_deleted = False
    if args.stream_video:
        ffmpeg_env = os.environ.copy()
        ffmpeg_env.pop("LD_LIBRARY_PATH", None)
        ffmpeg_candidates = [
            shutil.which("ffmpeg"),
            str(Path(sys.executable).resolve().parent / "ffmpeg"),
            "/usr/bin/ffmpeg",
        ]
        ffmpeg_bin = next(
            (candidate for candidate in ffmpeg_candidates
             if candidate and Path(candidate).is_file()),
            None,
        )
        if ffmpeg_bin is None:
            raise FileNotFoundError(
                "找不到 ffmpeg；请将其加入 PATH 或安装到当前 Python 环境的 bin 目录。")
        video_writer = RawVideoWriter(
            ffmpeg_bin, video_path, args.width, args.height, args.fps,
            ffmpeg_env, preset=args.ffmpeg_preset,
            queue_size=args.video_queue_size)
    common_render_kwargs = {
        "uniform_base_color": args.uniform_base_color,
        "uniform_roughness": args.uniform_roughness,
        "uniform_metallic": args.uniform_metallic,
    }

    table_gaussian_count = len(table_world["xyz"])
    foreground_gaussian_count = len(first_scene["xyz"]) - table_gaussian_count
    if args.depth_occlude_table:
        transition_description = (
            f"alpha={args.occlusion_alpha_low:.3f}.."
            f"{args.occlusion_alpha_high:.3f}, "
            f"depth={args.occlusion_depth_low:.4f}.."
            f"{args.occlusion_depth_high:.4f}m, "
            f"border={args.occlusion_boundary_protection}px"
            if args.table_occlusion_mode == "smooth"
            else (
                f"alpha>={args.occlusion_alpha_high:.3f}, "
                f"{args.occlusion_depth_epsilon:.4f}m<depth gap"
                f"<={args.occlusion_depth_high:.4f}m"
                if args.table_occlusion_mode == "strict"
                else f"alpha>={args.object_alpha_threshold:.3f}, "
                     f"depth epsilon={args.occlusion_depth_epsilon:.4f}m"
            )
        )
        print(
            "[depth occlusion] table GS="
            f"{table_gaussian_count}, foreground GS={foreground_gaussian_count}, "
            f"mode={args.table_occlusion_mode}, {transition_description}",
            flush=True,
        )

    def render_scene(frame_render_kwargs, profile_timing=False):
        """Render once, or depth-compose foreground over the table.

        Raster opacity splits only camera-visible layers.  Lighting and shadow
        rays still see the complete Gaussian model in both passes.
        """
        render_kwargs = {
            "training": False,
            "relight": True,
            "base_color_scale": torch.ones(
                3, dtype=torch.float32, device="cuda"),
            "profile_timing": profile_timing,
            **frame_render_kwargs,
        }
        if not args.depth_occlude_table:
            return render_ir(
                camera, model, pipe, background, **render_kwargs), 0.0

        full_raster_context = render_kwargs["raster_context"]
        table_render_kwargs = dict(render_kwargs)
        foreground_render_kwargs = dict(render_kwargs)
        table_render_kwargs["raster_context"] = {
            key: value[:table_gaussian_count]
            for key, value in full_raster_context.items()
        }
        foreground_render_kwargs["raster_context"] = {
            key: value[table_gaussian_count:]
            for key, value in full_raster_context.items()
        }

        selective_render = args.depth_compositing_render_mode == "selective"
        if selective_render:
            # These two inexpensive passes provide only per-layer alpha/depth.
            # Ordinary pixels come from one normal full-scene render below;
            # only pixels selected for hardening receive extra foreground
            # relighting.
            foreground_package = render_ir(
                camera, model, pipe, foreground_background,
                material_only=True, **foreground_render_kwargs)
            table_package = render_ir(
                camera, model, pipe, background,
                material_only=True, **table_render_kwargs)
        else:
            foreground_package = render_ir(
                camera, model, pipe, foreground_background,
                **foreground_render_kwargs)
            table_package = render_ir(
                camera, model, pipe, background,
                **table_render_kwargs)
        foreground_alpha = foreground_package["rend_alpha"]
        table_alpha = table_package["rend_alpha"]
        foreground_depth = foreground_package["surf_depth"]
        table_depth = table_package["surf_depth"]
        depth_gap = table_depth - foreground_depth
        valid_depth = (
            (foreground_depth > 0.0)
            & (table_depth > 0.0)
            & (table_alpha > 1.0 / 255.0)
        )

        # CLI depth thresholds are expressed in Isaac-world metres, whereas
        # this renderer uniformly enlarges the scene by SCENE_SCALE.
        if args.table_occlusion_mode == "strict":
            # Strict mode deliberately has no transition band: only pixels
            # that satisfy the alpha threshold and are close enough in depth
            # to potentially interact are allowed to fully occlude the table.
            # Boundary/uncertain pixels, and layers that are far apart in
            # depth, retain correction_confidence=0 and use ordinary alpha
            # compositing below.
            correction_confidence = (
                (foreground_alpha >= args.occlusion_alpha_high)
                & valid_depth
                & (depth_gap > SCENE_SCALE * args.occlusion_depth_epsilon)
                & (depth_gap <= SCENE_SCALE * args.occlusion_depth_high)
            ).to(foreground_alpha.dtype)
        elif args.table_occlusion_mode == "hard":
            correction_confidence = (
                (foreground_alpha >= args.object_alpha_threshold)
                & valid_depth
                & (depth_gap > SCENE_SCALE * args.occlusion_depth_epsilon)
            ).to(foreground_alpha.dtype)
        else:
            alpha_t = (
                (foreground_alpha - args.occlusion_alpha_low)
                / (args.occlusion_alpha_high - args.occlusion_alpha_low)
            ).clamp(0.0, 1.0)
            depth_t = (
                (depth_gap - SCENE_SCALE * args.occlusion_depth_low)
                / (SCENE_SCALE * (
                    args.occlusion_depth_high - args.occlusion_depth_low))
            ).clamp(0.0, 1.0)
            alpha_confidence = alpha_t.square() * (3.0 - 2.0 * alpha_t)
            depth_confidence = depth_t.square() * (3.0 - 2.0 * depth_t)
            correction_confidence = (
                alpha_confidence * depth_confidence
                * valid_depth.to(foreground_alpha.dtype)
            )
            if args.occlusion_boundary_protection:
                radius = args.occlusion_boundary_protection
                correction_confidence = -F.max_pool2d(
                    -correction_confidence,
                    kernel_size=2 * radius + 1,
                    stride=1,
                    padding=radius,
                )

        if selective_render:
            ordinary_package = render_ir(
                camera, model, pipe, background, **render_kwargs)
            correction_package = render_ir(
                camera, model, pipe, foreground_background,
                shading_mask_override=correction_confidence[0] > 0.0,
                **foreground_render_kwargs)
            foreground_straight = (
                correction_package["render"]
                / foreground_alpha.clamp_min(1e-6)
            )
            package = dict(ordinary_package)
            package["render"] = (
                ordinary_package["render"] * (1.0 - correction_confidence)
                + foreground_straight * correction_confidence
            )
            package["rend_alpha"] = (
                ordinary_package["rend_alpha"]
                + correction_confidence *
                (1.0 - ordinary_package["rend_alpha"])
            )
            package["mask"] = package["rend_alpha"][0] > 0.0
            package["surf_depth"] = torch.where(
                correction_confidence > 0.0, foreground_depth,
                ordinary_package["surf_depth"])
        else:
            # The foreground pass uses a black background, so its RGB is
            # premultiplied by raster alpha and can safely be converted to
            # straight color before reliable foreground pixels are hardened.
            foreground_straight = (
                foreground_package["render"]
                / foreground_alpha.clamp_min(1e-6)
            )
            composed_alpha = foreground_alpha + correction_confidence * (
                1.0 - foreground_alpha)
            composed_render = (
                foreground_straight * composed_alpha
                + table_package["render"] * (1.0 - composed_alpha)
            )
            final_alpha = torch.maximum(
                composed_alpha, table_alpha * (1.0 - composed_alpha))
            package = dict(foreground_package)
            package["render"] = composed_render
            package["rend_alpha"] = final_alpha
            package["mask"] = final_alpha[0] > 0.0
            package["surf_depth"] = torch.where(
                correction_confidence > 0.0, foreground_depth, table_depth)
        if args.save_compositing_debug:
            package["_composition_debug"] = {
                "foreground_alpha": foreground_alpha,
                "table_alpha": table_alpha,
                "depth_gap": depth_gap,
                "valid_depth": valid_depth,
                "correction_confidence": correction_confidence,
            }
        return package, float(correction_confidence.sum().item())

    def depth_normal_override():
        with torch.no_grad():
            material_package = render_ir(
                camera, model, pipe, background, training=False,
                material_only=True,
            )
        depth_normal = material_package["surf_normal"] / \
            material_package["rend_alpha"].clamp_min(1e-6)
        return F.normalize(depth_normal, dim=0, eps=1e-6)

    fixed_depth_normal = None
    if args.normal_source == "depth" and args.env_rotate_count:
        fixed_depth_normal = depth_normal_override()
    torch.cuda.synchronize()
    startup_timings["pipeline_and_camera_setup_s"] = time.perf_counter() - stage_start

    render_times = []
    depth_occlusion_corrected_pixels = []
    video_finalize_seconds = 0.0
    start = time.perf_counter()
    # Environment-only and camera-only rotations repeat exactly the same
    # trajectory row.  Keep both the Gaussian geometry and its OptiX GAS
    # resident instead of rebuilding millions of bounding triangles per
    # output frame.
    fixed_geometry_sequence = bool(
        args.env_rotate_count or args.camera_orbit_count
    )
    for output_index, row in enumerate(selected):
        # End-to-end interactive latency starts when a new trajectory state is
        # consumed.  It ends after the relit RGB is complete on the GPU, before
        # CPU readback, annotation, PNG output, or video encoding.
        torch.cuda.synchronize()
        trajectory_to_render_start = time.perf_counter()
        frame_component_transforms = None
        if output_index == 0 or not fixed_geometry_sequence:
            motion_context = component_motion_context(row)
            frame_component_transforms = component_transforms_from_context(
                motion_context)
            # The cold model already contains the first frame assembled on
            # the CPU. A reused model still contains the preceding video's
            # final pose, so refresh its first frame as well.
            if args.geometry_update_mode == "gpu" and (
                    (output_index and not fixed_geometry_sequence) or
                    persistent_model_reused):
                update_raster_geometry_on_gpu(frame_component_transforms)
            elif (args.geometry_update_mode == "cpu" and output_index and
                  not fixed_geometry_sequence):
                scene = dynamic_geometry(row)
                if args.gaussian_scale_multiplier != 1.0:
                    scene["scale"] = scene["scale"] + math.log(
                        args.gaussian_scale_multiplier)
                update_geometry(model, scene)
        torch.cuda.synchronize()
        bvh_start = time.perf_counter()
        if output_index == 0:
            can_reuse_ias = (
                persistent_model_reused
                and args.bvh_layout == "component-ias-rigid"
                and model._persistent_component_signature == component_signature
            )
            if can_reuse_ias:
                model.update_component_transforms(frame_component_transforms)
                persistent_ias_reused = True
            elif args.bvh_layout == "single":
                model.build_bvh(static=fixed_geometry_sequence)
            else:
                build_component_bvh(row, fixed_geometry_sequence)
                if (args.persistent_model_cache
                        and args.bvh_layout == "component-ias-rigid"):
                    model._persistent_component_signature = component_signature
        elif not fixed_geometry_sequence:
            if args.bvh_layout == "single":
                model.update_bvh()
            else:
                update_component_bvh(frame_component_transforms)
        env_angle = None
        camera_angle = None
        if args.env_rotate_count:
            env_angle = output_index * 360.0 / args.env_rotate_count
        elif args.trajectory_env_rotations:
            # Reach the requested number of full turns at the final trajectory
            # frame while geometry continues to follow the Isaac trace.
            denominator = max(len(selected) - 1, 1)
            env_angle = output_index * 360.0 * args.trajectory_env_rotations / denominator
        elif args.trajectory_env_deg_per_sec:
            env_angle = (
                trajectory_frame_offset + output_index
            ) * args.trajectory_env_deg_per_sec / args.fps
        if env_angle is not None:
            axis_index = {"x": 0, "y": 1, "z": 2}[args.env_rotate_axis]
            axis = np.zeros(3, dtype=np.float64)
            axis[axis_index] = 1.0
            env_rotation = Rotation.from_rotvec(math.radians(env_angle) * axis).as_matrix()
            active_light.set_transform(
                torch.as_tensor(env_rotation, dtype=torch.float32, device="cuda")
            )
        if (args.trajectory_light_intensity_end is not None and
                args.light_type in ("point", "area")):
            denominator = max(len(selected) - 1, 1)
            fraction = output_index / denominator
            start_intensity = args.light_intensity
            end_intensity = args.trajectory_light_intensity_end
            current_intensity = (
                start_intensity + fraction * (end_intensity - start_intensity)
            )
            if args.light_type == "point":
                active_light.intensity.copy_(
                    (SCENE_SCALE ** 2) * current_intensity *
                    torch.as_tensor(args.light_color, dtype=torch.float32, device="cuda")
                )
            else:
                active_light.radiance.copy_(
                    current_intensity *
                    torch.as_tensor(args.light_color, dtype=torch.float32, device="cuda")
                )
        if args.camera_orbit_count:
            camera_angle = output_index * 360.0 / args.camera_orbit_count
            camera = make_camera(camera_angle)
        frame_render_kwargs = dict(common_render_kwargs)
        frame_render_kwargs["raster_context"] = prepare_ir_raster_context(model)
        trace_mode = (
            "visibility"
            if model.direct_light is not None or pipe.wo_indirect_relight
            else "material"
        )
        with torch.no_grad():
            frame_render_kwargs["trace_context"] = model.prepare_trace_context(
                camera_center=camera.camera_center,
                trace_mode=trace_mode,
                use_metallic=getattr(pipe, "use_metallic_brdf", False),
            )
        frame_render_kwargs["minimal_output"] = (
            args.output_buffer == "render" and not args.save_compositing_debug)
        if args.normal_source == "depth":
            frame_render_kwargs["override_normal_map"] = (
                fixed_depth_normal
                if fixed_depth_normal is not None
                else depth_normal_override()
            )
        torch.cuda.synchronize()
        bvh_seconds = time.perf_counter() - bvh_start
        warmup_seconds = 0.0
        if args.profile_timing and args.profile_warmup > 0:
            warmup_start = time.perf_counter()
            for _ in range(args.profile_warmup):
                with torch.no_grad():
                    render_scene(frame_render_kwargs)
            torch.cuda.synchronize()
            warmup_seconds = time.perf_counter() - warmup_start

        baseline_render_seconds = None
        baseline_render_cuda_ms = None
        if args.profile_timing:
            baseline_start = torch.cuda.Event(enable_timing=True)
            baseline_end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            baseline_wall_start = time.perf_counter()
            baseline_start.record()
            with torch.no_grad():
                baseline_package, _ = render_scene(frame_render_kwargs)
            baseline_end.record()
            baseline_end.synchronize()
            baseline_render_seconds = time.perf_counter() - baseline_wall_start
            baseline_render_cuda_ms = float(
                baseline_start.elapsed_time(baseline_end))
            del baseline_package

        render_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cuda_start.record()
        with torch.no_grad():
            package, corrected_pixels = render_scene(
                frame_render_kwargs, profile_timing=args.profile_timing)
        cuda_end.record()
        torch.cuda.synchronize()
        render_seconds = time.perf_counter() - render_start
        trajectory_to_render_seconds = (
            time.perf_counter() - trajectory_to_render_start
        )
        render_cuda_ms = float(cuda_start.elapsed_time(cuda_end))
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        visible_pixels = int(package["mask"].sum().item())
        depth_occlusion_corrected_pixels.append(corrected_pixels)
        frame_timing = {
            "step": int(row["step"]),
            "bvh_s": bvh_seconds,
            "render_s": render_seconds,
            "trajectory_state_to_render_complete_s": trajectory_to_render_seconds,
            "render_cuda_ms": render_cuda_ms,
            "baseline_unprofiled_render_s": baseline_render_seconds,
            "baseline_unprofiled_render_cuda_ms": baseline_render_cuda_ms,
            "profile_warmup_count": args.profile_warmup if args.profile_timing else 0,
            "profile_warmup_s": warmup_seconds,
            "visible_pixels": visible_pixels,
            "depth_occlusion_corrected_pixels": corrected_pixels,
            "peak_memory_allocated_bytes": peak_allocated,
            "peak_memory_reserved_bytes": peak_reserved,
        }
        if args.profile_timing:
            timing_profile = package["timing_profile"]
            stage_sum_ms = float(sum(timing_profile["stages_ms"].values()))
            timing_profile["accounted_stage_sum_ms"] = stage_sum_ms
            timing_profile["unaccounted_cuda_ms"] = render_cuda_ms - stage_sum_ms
            timing_profile["stage_percent_of_render_cuda"] = {
                name: 100.0 * value / max(render_cuda_ms, 1e-9)
                for name, value in timing_profile["stages_ms"].items()
            }
            timing_profile["stage_percent_of_accounted_time"] = {
                name: 100.0 * value / max(stage_sum_ms, 1e-9)
                for name, value in timing_profile["stages_ms"].items()
            }
            bvh_substages = timing_profile.get("bvh_substages_ms", {})
            if bvh_substages:
                bvh_total_ms = timing_profile["stages_ms"]["bvh_trace"]
                bvh_substage_sum_ms = float(sum(bvh_substages.values()))
                timing_profile["bvh_substage_sum_ms"] = bvh_substage_sum_ms
                timing_profile["bvh_substage_unaccounted_ms"] = (
                    bvh_total_ms - bvh_substage_sum_ms)
                timing_profile["bvh_substage_percent_of_bvh_trace"] = {
                    name: 100.0 * value / max(bvh_total_ms, 1e-9)
                    for name, value in bvh_substages.items()
                }
                metadata = timing_profile["metadata"]
                metadata["bvh_candidate_ray_ratio"] = (
                    metadata["bvh_candidate_rays"]
                    / max(metadata["bvh_input_rays"], 1)
                )
            frame_timing["detailed_cuda_timing"] = timing_profile
        render_times.append(frame_timing)
        extra = (
            f"{args.light_type}={env_angle:.1f}deg"
            if env_angle is not None else None
        )
        if camera_angle is not None:
            extra = f"camera={camera_angle:.1f}deg"
        output_start = time.perf_counter()
        output_tensor = package[args.output_buffer]
        if args.save_compositing_debug:
            if not args.depth_occlude_table:
                raise RuntimeError(
                    "--save-compositing-debug requires --depth-occlude-table")
            save_compositing_debug(
                out / "compositing_debug",
                row["step"],
                package["render"],
                package["_composition_debug"],
                args,
            )
        if args.output_buffer in ("roughness", "metallic"):
            output_tensor = output_tensor.expand(3, -1, -1)
        elif args.output_buffer == "rend_normal":
            output_tensor = 0.5 * (output_tensor + 1.0)
        # Quantize on the GPU before readback: RGB8 transfers one quarter of
        # the bytes of the former float32 -> CPU -> PIL path.
        output_rgb8 = (
            output_tensor.clamp(0.0, 1.0).mul(255.0).round()
            .to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()
        )
        if args.frame_label:
            image = Image.fromarray(output_rgb8, mode="RGB")
            image = draw_label(
                image, row, render_seconds, extra=extra)
            output_bytes = image.tobytes()
        else:
            output_bytes = output_rgb8.tobytes()
        if video_writer is not None:
            video_writer.write(output_bytes)
        else:
            image = Image.fromarray(output_rgb8, mode="RGB")
            image.save(frames_dir / f"frame_{output_index:05d}.png")
        frame_output_seconds = time.perf_counter() - output_start
        frame_timing["frame_output_s"] = frame_output_seconds
        # Backward-compatible field name used by existing timing reports.
        frame_timing["cpu_transfer_label_png_s"] = frame_output_seconds
        print(
            f"frame {output_index + 1}/{len(selected)} step={row['step']} "
            f"bvh={bvh_seconds:.3f}s render={render_seconds:.3f}s "
            f"trajectory_to_rgb={trajectory_to_render_seconds:.3f}s",
            flush=True,
        )

    if video_mode:
        video_finalize_start = time.perf_counter()
        if video_writer is not None:
            if video_writer.close() != 0:
                raise RuntimeError("ffmpeg failed while encoding streamed video")
            video_frames_deleted = True
        else:
            ffmpeg_env = os.environ.copy()
            ffmpeg_env.pop("LD_LIBRARY_PATH", None)
            ffmpeg_candidates = [
                shutil.which("ffmpeg"),
                str(Path(sys.executable).resolve().parent / "ffmpeg"),
                "/usr/bin/ffmpeg",
            ]
            ffmpeg_bin = next(
                (candidate for candidate in ffmpeg_candidates
                 if candidate and Path(candidate).is_file()),
                None,
            )
            if ffmpeg_bin is None:
                raise FileNotFoundError(
                    "找不到 ffmpeg；请将其加入 PATH 或安装到当前 Python 环境的 bin 目录。")
            subprocess.run(
                [
                    ffmpeg_bin, "-hide_banner", "-loglevel", "warning", "-y",
                    "-framerate", str(args.fps),
                    "-i", str(frames_dir / "frame_%05d.png"),
                    "-c:v", "libx264", "-preset", args.ffmpeg_preset,
                    "-crf", "18", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(video_path),
                ],
                check=True,
                env=ffmpeg_env,
            )
            if args.delete_video_frames:
                shutil.rmtree(frames_dir)
                video_frames_deleted = True
        video_finalize_seconds = time.perf_counter() - video_finalize_start

    report = {
        "mode": (
            "fixed_scene_environment_rotates" if args.env_rotate_count else
            "fixed_scene_camera_orbits" if args.camera_orbit_count else
            "dynamic_trajectory_environment_rotates" if trajectory_env_moves else
            "all_geometry_rendered_as_dynamic_irgs"
        ),
        "scene_contents": args.scene,
        "trajectory_set": args.trajectory_set,
        "scene_scale": SCENE_SCALE,
        "table_z_offset_m": args.table_z_offset_m,
        "trace": str(trace_path),
        "contract": str(contract_path),
        "selected_source_steps": [int(row["step"]) for row in selected],
        "source_physics_hz": contract.get("physics_rate_hz"),
        "source_stride": args.stride if args.full_video else None,
        "source_output_frame_offset": trajectory_frame_offset if args.full_video else None,
        "output_fps": args.fps if (args.full_video or args.env_rotate_count or args.camera_orbit_count) else None,
        "env_rotation": {
            "frame_count": len(selected) if trajectory_env_moves else args.env_rotate_count,
            "axis": args.env_rotate_axis if (args.env_rotate_count or trajectory_env_moves) else None,
            "degrees": (
                (len(selected) - 1) * args.trajectory_env_deg_per_sec / args.fps
                if args.trajectory_env_deg_per_sec else
                360.0 * args.trajectory_env_rotations
                if args.trajectory_env_rotations else
                360.0 if args.env_rotate_count else 0.0
            ),
            "turns_during_trajectory": args.trajectory_env_rotations,
            "degrees_per_second": args.trajectory_env_deg_per_sec,
            "start_degrees": (
                trajectory_frame_offset * args.trajectory_env_deg_per_sec / args.fps
                if args.trajectory_env_deg_per_sec else 0.0
            ),
        },
        "camera": {
            "name": args.camera_name,
            "contract_key": camera_key,
            "prim_path": camera_cfg["prim_path"],
            "eye_m": list(map(float, eye_m)),
            "target_m": list(map(float, target_m)),
            "orbit_frame_count": args.camera_orbit_count,
            "orbit_axis": "world_z" if args.camera_orbit_count else None,
            "orbit_degrees": 360.0 if args.camera_orbit_count else 0.0,
        },
        "resolution": [args.width, args.height],
        "camera_fovy_deg_assumed": camera_fovy_deg,
        "camera_fovy_source": camera_fovy_source,
        "environment": args.envmap if args.light_type == "env" else None,
        "lighting": {
            "type": args.light_type,
            "color_linear_rgb": (
                args.light_color if args.light_type != "env" else None
            ),
            "intensity": (
                args.light_intensity if args.light_type != "env" else None
            ),
            "analytic_samples": (
                1 if args.light_type == "point"
                else args.analytic_light_samples if args.light_type == "area"
                else None
            ),
            "point_position_m": (
                args.point_light_position_m if args.light_type == "point" else None
            ),
            "area_center_m": (
                args.area_light_center_m if args.light_type == "area" else None
            ),
            "area_target_m": (
                args.area_light_target_m if args.light_type == "area" else None
            ),
            "area_up": args.area_light_up if args.light_type == "area" else None,
            "area_size_m": (
                args.area_light_size_m if args.light_type == "area" else None
            ),
            "area_two_sided": (
                args.area_light_two_sided if args.light_type == "area" else None
            ),
            "analytic_indirect_supported": (
                False if args.light_type != "env" else None
            ),
        },
        "diffuse_samples": args.diffuse_samples,
        "light_samples": args.light_samples,
        "diffuse_sampling_mode": args.diffuse_sampling_mode,
        "light_sampling_mode": args.light_sampling_mode,
        "fg_lut_query_layout": args.fg_lut_query_layout,
        "fg_lut_tile_width": args.fg_lut_tile_width,
        "render_ray_budget": args.render_ray_budget,
        "bvh_layout": {
            "mode": args.bvh_layout,
            "bounding_polyhedron": model.bounding_polyhedron,
            "legacy_single_gas_fallback": args.bvh_layout == "single",
            "components": [
                {
                    "index": index,
                    "label": component["label"],
                    "motion_type": component["motion_type"],
                    "gaussian_range": [component["begin"], component["end"]],
                    "gaussian_count": component["end"] - component["begin"],
                }
                for index, component in enumerate(component_plan)
            ],
        },
        "indirect_relighting": (
            not args.disable_indirect if args.light_type == "env" else False
        ),
        "base_color_min": args.base_color_min,
        "diagnostic_ablation": {
            "variant_name": args.material_variant_name,
            "uniform_base_color_linear": args.uniform_base_color,
            "uniform_roughness": args.uniform_roughness,
            "uniform_metallic": args.uniform_metallic,
            "normal_source": args.normal_source,
            "gaussian_scale_multiplier": args.gaussian_scale_multiplier,
            "indirect_relighting_enabled": not args.disable_indirect,
            "use_metallic_brdf": bool(getattr(pipe, "use_metallic_brdf", False)),
        },
        "light_t_min_scene_units": args.light_t_min,
        "profile_timing_enabled": args.profile_timing,
        "python_render_optimizations": {
            "gpu_rigid_geometry_update": args.geometry_update_mode == "gpu",
            "layer_raster_uses_gaussian_subsets": bool(
                args.depth_occlude_table),
            "shared_trace_context_per_frame": True,
            "minimal_output": (
                args.output_buffer == "render" and
                not args.save_compositing_debug),
            "table_gbuffer_cached": False,
        },
        "depth_aware_table_occlusion": {
            "enabled": bool(args.depth_occlude_table),
            "mode": args.table_occlusion_mode if args.depth_occlude_table else None,
            "render_mode": (
                args.depth_compositing_render_mode
                if args.depth_occlude_table else None),
            "table_gaussians": table_gaussian_count if args.depth_occlude_table else None,
            "foreground_gaussians": (
                foreground_gaussian_count if args.depth_occlude_table else None),
            "object_alpha_threshold": (
                args.object_alpha_threshold if args.depth_occlude_table else None),
            "depth_epsilon_m": (
                args.occlusion_depth_epsilon if args.depth_occlude_table else None),
            "alpha_transition": (
                [args.occlusion_alpha_low, args.occlusion_alpha_high]
                if args.depth_occlude_table else None),
            "depth_transition_m": (
                [args.occlusion_depth_low, args.occlusion_depth_high]
                if args.depth_occlude_table else None),
            "boundary_protection_px": (
                args.occlusion_boundary_protection
                if args.depth_occlude_table else None),
            "mean_corrected_pixels": (
                float(np.mean(depth_occlusion_corrected_pixels))
                if args.depth_occlude_table and depth_occlusion_corrected_pixels
                else None),
            "raster_layers": "table versus robot_and_object",
            "lighting_visibility_uses_full_scene": bool(args.depth_occlude_table),
        },
        "startup_timings": startup_timings,
        "gaussian_counts": {
            "table": len(table_world["xyz"]) if args.scene in ("full", "table-only") else 0,
            "robot": (sum(len(robot[link]["xyz"]) for link in LINKS)
                      if args.scene in ("full", "robot-only") else 0),
            object_name: (len(cup_local["xyz"])
                          if args.scene in ("full", "object-only") else 0),
            "total": len(first_scene["xyz"]),
        },
        "rejected_floaters": {"table": table_rejected, object_name: cup_rejected, "robot": robot_rejected},
        "world_from_urdf_base_m": world_from_base.tolist(),
        "robot_assets": {
            "root": str(robot_root),
            "urdf": str(urdf_path),
        },
        "table_transform": {
            "ply": str(table_ply_path),
            "replacement_source_registration": (
                table_registration.tolist() if table_registration is not None else None
            ),
            "replacement_source_registration_file": args.table_source_registration,
            "surface_filter_mesh": args.table_surface_mesh,
            "surface_filter_max_distance": args.table_surface_max_distance,
            "source": str(dossier_path),
            "matrix_convention": "column vector",
            table_transform_key: table_transform.tolist(),
            "uniform_scale": table_cfg.get("uniform_scale"),
            "fit_max_residual_m": transform_dossier["table"].get("max_final_usd_point_error_m"),
        },
        f"{object_name}_transform": {
            "status": "EXACT_PROVENANCE_TRANSFORM",
            "asset": str(
                trace_root / ("assets/lego/lego_excavator_full_mesh.usd" if object_name == "lego"
                              else "assets/cup/cup_canonical_80mm_kraft_black_lid_converted.usd")
            ),
            "source": str(dossier_path),
            "matrix_convention": "column vector",
            "T_canonical_or_visual_local_from_source": cup_transform.tolist(),
            "method": "audited source-to-rigid-object-local transform from provenance",
            "source_bounds": cup_bounds.tolist() if cup_bounds is not None else None,
            "target_dimensions_m": cup_dimensions.tolist(),
            "visual_translation_local_m": cup_local_shift.tolist(),
            "source_glb_sha256": transform_dossier[object_name].get(
                "source_glb_sha256", transform_dossier[object_name].get("source_sha256")
            ),
        },
        "timings": render_times,
        "elapsed_s": time.perf_counter() - start,
        "video": str(video_path) if video_path else None,
        "video_frames_retained": not video_frames_deleted,
        "video_output": {
            "streamed": bool(args.stream_video),
            "frame_label": bool(args.frame_label),
            "ffmpeg_encoder": "libx264",
            "ffmpeg_preset": args.ffmpeg_preset,
            "queue_size": args.video_queue_size if args.stream_video else 0,
            "gpu_uint8_readback": True,
            "finalize_seconds": video_finalize_seconds,
        },
        "geometry_update_mode": args.geometry_update_mode,
        "persistent_cache": {
            "enabled": bool(args.persistent_model_cache),
            "asset_signature": model_cache_key,
            "ply_hits_this_job": _PLY_CACHE_HITS - ply_cache_hits_before,
            "ply_misses_this_job": _PLY_CACHE_MISSES - ply_cache_misses_before,
            "gpu_model_reused": persistent_model_reused,
            "rigid_ias_reused": persistent_ias_reused,
        },
        "physical_gpu_id": args.physical_gpu_id,
    }
    if args.env_rotate_count:
        report_name = "envrotate_report.json"
    elif args.camera_orbit_count:
        report_name = "camera_orbit_report.json"
    elif args.full_video:
        report_name = "video_report.json"
    else:
        report_name = "preview_report.json"
    (out / report_name).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
