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
import json
import math
import os
import subprocess
import sys
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
from torchvision.transforms.functional import to_pil_image
from trimesh.transformations import euler_matrix, rotation_matrix


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from arguments import PipelineParams  # noqa: E402
from gaussian_renderer import render_ir  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.light import EnvLight, PointLight, RectAreaLight  # noqa: E402
from utils.graphics_utils import getProjectionMatrix  # noqa: E402


SCENE_SCALE = 4.0
URDF = REPO_ROOT / "SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
if not URDF.exists():
    # In this checkout SO-ARM100 is vendored inside IRGS rather than next to
    # it; support both layouts without changing the robot coordinate model.
    URDF = ROOT / "SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
TRACE_ROOT = ROOT / "dataset/guiji1"
TRACE = TRACE_ROOT / "trajectories/formal_task_lift_trace.csv"
CONTRACT = TRACE_ROOT / "configs/execution_contract.json"
TABLE_ALIGNMENT = TRACE_ROOT / "results/newbase_geometry_mapping.json"
TRANSFORM_DOSSIER = TRACE_ROOT / "provenance/complete_coordinate_transforms.json"
TABLE_PLY = ROOT / "outputs/input_video_frames_newbase/irgs/point_cloud/iteration_20000/point_cloud.ply"
CUP_PLY = ROOT / "outputs/input_video_frames_beizi/irgs/point_cloud/iteration_20000/point_cloud.ply"
ROBOT_ROOT = ROOT / "outputs/so101_links"
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
    return result, int((~keep).sum())


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


def affine_transform_surfels(data, linear, translation):
    """Apply a general affine map and refactor each rank-2 Gaussian frame."""
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
    return result


def concatenate(parts):
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}


def concatenate_geometry(parts):
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in ("xyz", "scale", "rot")}


def parse_urdf():
    joints = {}
    for element in ET.parse(URDF).getroot().findall("joint"):
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


def make_gaussian_model(data, base_color_min=0.0):
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
    return model


def update_geometry(model, data):
    model._xyz.data.copy_(torch.as_tensor(data["xyz"], dtype=torch.float32, device="cuda"))
    model._scaling.data.copy_(torch.as_tensor(data["scale"], dtype=torch.float32, device="cuda"))
    model._rotation.data.copy_(torch.as_tensor(data["rot"], dtype=torch.float32, device="cuda"))


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", default=None,
        help="Comma-separated trace row indices. Defaults are chosen per trajectory set.",
    )
    parser.add_argument("--full-video", action="store_true")
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
        "--scene",
        choices=("full", "robot-only", "table-only", "object-only"),
        default="full",
        help=("Render the full scene or isolate the articulated robot, table, "
              "or trajectory object (cup/lego)."),
    )
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fovy-deg", type=float, default=27.0)
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
    args = parser.parse_args()
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

    custom_trace_root = Path(args.trajectory_dir).expanduser().resolve() if args.trajectory_dir else None
    if custom_trace_root is not None:
        trace_root = custom_trace_root
        trace_candidates = sorted((trace_root / "trajectories").glob("*.csv"))
        trace_path = trace_candidates[0] if trace_candidates else trace_root / "trajectory.csv"
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
        trace_path = trace_root / "trajectories/formal_task_lift_trace.csv"
        contract_path = trace_root / "configs/execution_contract.json"
        dossier_path = trace_root / "provenance/complete_coordinate_transforms.json"
        audit_path = trace_root / "results/asset_conversion_audit.json"
        table_ply_path = ROOT / "outputs_syn4_no_pseudo_t005/Synthetic4Relight/wide_four_corner_table_full_sphere_close/irgs/point_cloud/iteration_20000/point_cloud.ply"
        object_ply_path = ROOT / "outputs_syn4_no_pseudo_t005/Synthetic4Relight/lego/irgs/point_cloud/iteration_20000/point_cloud.ply"
        object_name = "lego"
    else:
        trace_root = TRACE_ROOT
        trace_path = TRACE
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
    rows = list(csv.DictReader(trace_path.open()))
    if args.steps is None:
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
    robot = {}
    robot_rejected = 0
    for link in LINKS:
        robot[link], rejected = load_irgs_ply(
            ROBOT_ROOT / link / "irgs_full/point_cloud/iteration_20000/point_cloud.ply"
        )
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
        table_world = affine_transform_surfels(
            table,
            SCENE_SCALE * table_transform[:3, :3],
            SCENE_SCALE * table_transform[:3, 3],
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
            table_world["xyz"] += SCENE_SCALE * contract_translation
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
        cup_local = affine_transform_surfels(
            cup,
            SCENE_SCALE * cup_transform[:3, :3],
            SCENE_SCALE * cup_transform[:3, 3],
        )

    joints = parse_urdf()
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
        selected = [rows[index] for index in requested]

    first_scene = dynamic_parts(selected[0])
    if args.gaussian_scale_multiplier != 1.0:
        first_scene["scale"] = first_scene["scale"] + math.log(
            args.gaussian_scale_multiplier)

    # Scene-specific loading ends here.  From this point down the IAS code
    # sees only ordered generic components and their motion type; no asset name
    # is interpreted by the CUDA/OptiX implementation.
    component_plan = []
    component_cursor = 0

    def append_component(label, motion_type, count, transform_fn=None):
        nonlocal component_cursor
        count = int(count)
        component_plan.append({
            "label": label,
            "motion_type": motion_type,
            "begin": component_cursor,
            "end": component_cursor + count,
            "transform_fn": transform_fn,
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
            )
    if args.scene in ("full", "object-only"):
        append_component(
            "trajectory_rigid_0", "rigid_instance", len(cup_local["xyz"]),
            transform_fn=lambda context: context["object"],
        )
    if component_cursor != len(first_scene["xyz"]):
        raise RuntimeError(
            "component ranges do not match assembled Gaussian ordering")
    startup_timings["coordinate_transform_and_scene_assembly_s"] = (
        time.perf_counter() - stage_start)

    stage_start = time.perf_counter()
    model = make_gaussian_model(first_scene, base_color_min=args.base_color_min)
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

    def component_transforms(row):
        transforms = []
        context = component_motion_context(row)
        for component in component_plan:
            if (args.bvh_layout == "component-ias-rigid" and
                    component["motion_type"] == "rigid_instance"):
                transform = component["transform_fn"](context)[:3, :4]
            else:
                transform = np.eye(4, dtype=np.float64)[:3]
            transforms.append(transform)
        return np.stack(transforms, axis=0).astype(np.float32)

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

    def update_component_bvh(row):
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
        model.update_component_transforms(component_transforms(row))
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

    def make_camera(orbit_angle_deg=0.0):
        eye_offset = base_eye - camera_target
        if orbit_angle_deg:
            eye_offset = Rotation.from_rotvec(
                math.radians(orbit_angle_deg) * np.array([0.0, 0.0, 1.0])
            ).apply(eye_offset)
        c2w = look_at_colmap(camera_target + eye_offset, camera_target, camera_up)
        return SceneCamera(c2w, args.width, args.height, math.radians(args.fovy_deg))

    camera = make_camera()
    background = torch.ones(3, dtype=torch.float32, device="cuda")
    common_render_kwargs = {
        "uniform_base_color": args.uniform_base_color,
        "uniform_roughness": args.uniform_roughness,
        "uniform_metallic": args.uniform_metallic,
    }

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
        scene = (
            first_scene
            if output_index == 0 or fixed_geometry_sequence
            else dynamic_geometry(row)
        )
        if output_index and args.gaussian_scale_multiplier != 1.0:
            scene["scale"] = scene["scale"] + math.log(
                args.gaussian_scale_multiplier)
        if output_index and not fixed_geometry_sequence:
            update_geometry(model, scene)
        torch.cuda.synchronize()
        bvh_start = time.perf_counter()
        if output_index == 0:
            if args.bvh_layout == "single":
                model.build_bvh(static=fixed_geometry_sequence)
            else:
                build_component_bvh(row, fixed_geometry_sequence)
        elif not fixed_geometry_sequence:
            if args.bvh_layout == "single":
                model.update_bvh()
            else:
                update_component_bvh(row)
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
                    render_ir(
                        camera, model, pipe, background, training=False,
                        relight=True,
                        base_color_scale=torch.ones(
                            3, dtype=torch.float32, device="cuda"),
                        **frame_render_kwargs,
                    )
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
                baseline_package = render_ir(
                    camera, model, pipe, background, training=False,
                    relight=True,
                    base_color_scale=torch.ones(
                        3, dtype=torch.float32, device="cuda"),
                    **frame_render_kwargs,
                )
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
            package = render_ir(
                camera,
                model,
                pipe,
                background,
                training=False,
                relight=True,
                base_color_scale=torch.ones(3, dtype=torch.float32, device="cuda"),
                profile_timing=args.profile_timing,
                **frame_render_kwargs,
            )
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
        if args.output_buffer in ("roughness", "metallic"):
            output_tensor = output_tensor.expand(3, -1, -1)
        elif args.output_buffer == "rend_normal":
            output_tensor = 0.5 * (output_tensor + 1.0)
        image = draw_label(
            to_pil_image(output_tensor.clamp(0, 1).cpu()), row, render_seconds, extra=extra
        )
        image.save(frames_dir / f"frame_{output_index:05d}.png")
        frame_timing["cpu_transfer_label_png_s"] = time.perf_counter() - output_start
        print(
            f"frame {output_index + 1}/{len(selected)} step={row['step']} "
            f"bvh={bvh_seconds:.3f}s render={render_seconds:.3f}s "
            f"trajectory_to_rgb={trajectory_to_render_seconds:.3f}s",
            flush=True,
        )

    video_path = None
    if args.full_video or args.env_rotate_count or args.camera_orbit_count:
        if args.env_rotate_count:
            video_path = out / f"step_{int(selected[0]['step']):04d}_envrotate_irgs.mp4"
        elif args.camera_orbit_count:
            video_path = out / f"step_{int(selected[0]['step']):04d}_camera_orbit_irgs.mp4"
        else:
            if args.camera_name == "contact":
                video_path = out / f"so101_table_{object_name}_trajectory_contact_camera_irgs.mp4"
            else:
                video_path = out / f"so101_table_{object_name}_trajectory_irgs.mp4"
        ffmpeg_env = os.environ.copy()
        ffmpeg_env.pop("LD_LIBRARY_PATH", None)
        subprocess.run(
            [
                "/usr/bin/ffmpeg", "-y", "-framerate", str(args.fps),
                "-i", str(frames_dir / "frame_%05d.png"),
                "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(video_path),
            ],
            check=True,
            env=ffmpeg_env,
        )

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
        "source_physics_hz": 120,
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
        "camera_fovy_deg_assumed": args.fovy_deg,
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
