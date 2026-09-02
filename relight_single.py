#!/usr/bin/env python3
"""Relight one trained IRGS object and write an RGB video.

The camera can stay fixed or interpolate through the dataset cameras.  The
environment can stay fixed or rotate continuously, so pose interpolation and
environment rotation may be enabled at the same time.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline, Slerp


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import render_ir  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from scene.light import EnvLight, PointLight  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402


class RawVideoWriter:
    def __init__(self, path: Path, width: int, height: int, fps: float, crf: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.pop("LD_LIBRARY_PATH", None)
        self.path = path
        self.process = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                "-an", "-c:v", "libx264", "-preset", "medium",
                "-crf", str(crf), "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(path),
            ],
            stdin=subprocess.PIPE,
            env=environment,
        )

    def write(self, image: Image.Image):
        self.process.stdin.write(np.ascontiguousarray(image).tobytes())

    def close(self):
        self.process.stdin.close()
        return_code = self.process.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg failed ({return_code}): {self.path}")


def rotation_matrix(
    angle_degrees: float, axis: str, axis_vector=None
) -> torch.Tensor:
    angle = math.radians(angle_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    if axis_vector is not None:
        vector = np.asarray(axis_vector, dtype=np.float64)
        norm = np.linalg.norm(vector)
        if norm < 1e-12:
            raise ValueError("Environment rotation vector cannot be zero")
        x, y, z = vector / norm
        one_minus_cosine = 1.0 - cosine
        matrix = [
            [cosine + x*x*one_minus_cosine,
             x*y*one_minus_cosine - z*sine,
             x*z*one_minus_cosine + y*sine],
            [y*x*one_minus_cosine + z*sine,
             cosine + y*y*one_minus_cosine,
             y*z*one_minus_cosine - x*sine],
            [z*x*one_minus_cosine - y*sine,
             z*y*one_minus_cosine + x*sine,
             cosine + z*z*one_minus_cosine],
        ]
        return torch.tensor(matrix, dtype=torch.float32, device="cuda")
    matrices = {
        "x": [[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]],
        "y": [[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]],
        "z": [[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]],
    }
    return torch.tensor(matrices[axis], dtype=torch.float32, device="cuda")


def load_environment(path_string: str, max_resolution: int):
    path = Path(path_string).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    environment = EnvLight(
        path=str(path), device="cuda", max_res=max_resolution, activation="none"
    ).cuda()
    environment.build_mips()
    environment.update_pdf()
    environment.set_transform(torch.eye(3, dtype=torch.float32, device="cuda"))
    return path, environment


def sorted_cameras(scene):
    cameras = list(scene.getTrainCameras()) + list(scene.getTestCameras())
    cameras.sort(key=lambda camera: str(camera.image_name))
    if not cameras:
        raise RuntimeError("The model has no cameras")
    return cameras


def sorted_train_cameras(scene):
    cameras = list(scene.getTrainCameras())
    cameras.sort(key=lambda camera: str(camera.image_name))
    if not cameras:
        raise RuntimeError("The model has no training cameras")
    return cameras


def camera_to_c2w(camera) -> np.ndarray:
    return np.linalg.inv(camera.world_view_transform.T.detach().cpu().numpy())


def interpolate_camera_poses(cameras, steps_per_pair: int):
    if len(cameras) < 2:
        raise RuntimeError("Pose interpolation requires at least two cameras")
    key_poses = [camera_to_c2w(camera) for camera in cameras]
    frames = []
    for left_index in range(len(key_poses) - 1):
        right_index = left_index + 1
        left_pose, right_pose = key_poses[left_index], key_poses[right_index]
        rotations = Rotation.from_matrix(
            np.stack((left_pose[:3, :3], right_pose[:3, :3]), axis=0)
        )
        slerp = Slerp([0.0, 1.0], rotations)
        for step in range(steps_per_pair):
            t = step / steps_per_pair
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = slerp([t]).as_matrix()[0].astype(np.float32)
            c2w[:3, 3] = (
                (1.0 - t) * left_pose[:3, 3] + t * right_pose[:3, 3]
            ).astype(np.float32)
            frames.append((c2w, left_index, right_index, t))
    frames.append((key_poses[-1].astype(np.float32), len(key_poses) - 2,
                   len(key_poses) - 1, 1.0))
    return frames


def smooth_interpolate_camera_poses(cameras, frame_count: int, keyframe_stride: int):
    """Smooth camera path with continuous translation and angular velocity.

    Sparse keyframes suppress small frame-to-frame COLMAP jitter.  The spline
    is densely evaluated and then resampled by a joint translation/rotation
    arc-length metric to avoid visible speed changes around camera nodes.
    """
    if len(cameras) < 4:
        raise RuntimeError("Smooth pose interpolation requires at least four cameras")
    key_indices = list(range(0, len(cameras), keyframe_stride))
    if key_indices[-1] != len(cameras) - 1:
        key_indices.append(len(cameras) - 1)
    key_poses = [camera_to_c2w(cameras[index]) for index in key_indices]
    positions = np.stack([pose[:3, 3] for pose in key_poses], axis=0)
    rotations = Rotation.from_matrix(np.stack([pose[:3, :3] for pose in key_poses], axis=0))

    translation_step = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    rotation_step = (rotations[:-1].inv() * rotations[1:]).magnitude()
    translation_scale = max(float(np.median(translation_step[translation_step > 1e-8])), 1e-6)
    rotation_scale = max(float(np.median(rotation_step[rotation_step > 1e-8])), 1e-6)
    parameter_step = np.sqrt(
        (translation_step / translation_scale) ** 2
        + (rotation_step / rotation_scale) ** 2
    )
    # Limit a single noisy COLMAP jump from consuming most of the trajectory.
    parameter_step = np.clip(parameter_step, 0.25, 4.0)
    parameter = np.concatenate(([0.0], np.cumsum(parameter_step)))

    position_spline = CubicSpline(parameter, positions, axis=0, bc_type="natural")
    rotation_spline = RotationSpline(parameter, rotations)
    dense_count = max(frame_count * 10, 2000)
    dense_parameter = np.linspace(parameter[0], parameter[-1], dense_count)
    dense_positions = position_spline(dense_parameter)
    dense_rotations = rotation_spline(dense_parameter)
    dense_translation = np.linalg.norm(np.diff(dense_positions, axis=0), axis=1)
    dense_rotation = (dense_rotations[:-1].inv() * dense_rotations[1:]).magnitude()
    arc_step = np.sqrt(
        (dense_translation / translation_scale) ** 2
        + (dense_rotation / rotation_scale) ** 2
    )
    arc = np.concatenate(([0.0], np.cumsum(arc_step)))
    sample_arc = np.linspace(arc[0], arc[-1], frame_count)
    sample_parameter = np.interp(sample_arc, arc, dense_parameter)
    sample_positions = position_spline(sample_parameter)
    sample_rotations = rotation_spline(sample_parameter).as_matrix()

    frames = []
    for index, (position, rotation) in enumerate(zip(sample_positions, sample_rotations)):
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = rotation.astype(np.float32)
        c2w[:3, 3] = position.astype(np.float32)
        progress = index / max(frame_count - 1, 1)
        frames.append((c2w, 0, len(cameras) - 1, progress))
    return frames, len(key_indices)


def set_camera_pose(camera, c2w: np.ndarray):
    """Update the camera matrices and per-pixel rays for a novel pose."""
    w2c = np.linalg.inv(c2w).astype(np.float32)
    camera.world_view_transform = torch.from_numpy(w2c.T.copy()).cuda()
    camera.full_proj_transform = (
        camera.world_view_transform.unsqueeze(0)
        .bmm(camera.projection_matrix.unsqueeze(0))
        .squeeze(0)
    )
    camera.camera_center = camera.world_view_transform.inverse()[3, :3]
    camera.R = torch.from_numpy(w2c[:3, :3].T.copy()).cuda()
    camera.T = torch.from_numpy(w2c[:3, 3].copy()).cuda()
    camera.c2w = torch.from_numpy(c2w.copy()).cuda()

    height, width = int(camera.image_height), int(camera.image_width)
    v, u = torch.meshgrid(
        torch.arange(height, device="cuda"),
        torch.arange(width, device="cuda"),
        indexing="ij",
    )
    focal_x = width / (2.0 * np.tan(camera.FoVx * 0.5))
    focal_y = height / (2.0 * np.tan(camera.FoVy * 0.5))
    rays_camera = torch.stack(
        (
            (u - width / 2 + 0.5) / focal_x,
            (v - height / 2 + 0.5) / focal_y,
            torch.ones_like(u),
        ),
        dim=-1,
    ).reshape(-1, 3)
    rays = rays_camera @ camera.world_view_transform[:3, :3].T
    camera.rays_d_unnormalized = rays
    camera.rays_d = F.normalize(rays, dim=-1)
    camera.rays_o = camera.camera_center[None].expand_as(camera.rays_d)
    camera.rays_d_hw = camera.rays_d.reshape(height, width, 3)
    camera.rays_d_hw_unnormalized = rays.reshape(height, width, 3)


def look_at_colmap(eye, target, world_up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Build a COLMAP/OpenCV c2w pose (x right, y down, z forward)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(world_up, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.stack((right, down, forward), axis=1)
    c2w[:3, 3] = eye
    return c2w


def rgb_image(tensor: torch.Tensor) -> Image.Image:
    rgb = tensor.detach().clamp(0.0, 1.0).mul(255).byte()
    rgb = rgb.permute(1, 2, 0).contiguous().cpu().numpy()
    height, width = rgb.shape[:2]
    if width % 2:
        rgb = rgb[:, :-1]
    if height % 2:
        rgb = rgb[:-1]
    return Image.fromarray(rgb, mode="RGB")


def main():
    parser = ArgumentParser(description=__doc__)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=20000)
    parser.add_argument("--envmap", default=None)
    parser.add_argument("--output", required=True, help="Output .mp4 or single-frame .png path")
    parser.add_argument(
        "--pose-mode", choices=("fixed", "interpolate"), default="fixed",
        help="Keep one camera fixed or interpolate through all dataset cameras.",
    )
    parser.add_argument("--view-id", type=int, default=0)
    parser.add_argument("--camera-eye", type=float, nargs=3, default=None)
    parser.add_argument("--camera-target", type=float, nargs=3, default=None)
    parser.add_argument(
        "--camera-up", type=float, nargs=3, default=None,
        help="Optional world-space up vector for a custom camera; defaults to +Z.",
    )
    parser.add_argument("--camera-end-eye", type=float, nargs=3, default=None)
    parser.add_argument("--camera-end-target", type=float, nargs=3, default=None)
    parser.add_argument("--transition-frames", type=int, default=72)
    parser.add_argument(
        "--top-between-training", action="store_true",
        help="Visit the end camera between consecutive sequence cameras.",
    )
    parser.add_argument(
        "--train-only-sequence", action="store_true",
        help="Use only COLMAP training cameras for a camera sequence.",
    )
    parser.add_argument(
        "--output-buffer",
        choices=(
            "render", "diffuse", "specular", "visibility",
            "base_color", "roughness", "rend_normal",
        ),
        default="render",
    )
    parser.add_argument(
        "--force-visibility-one", action="store_true",
        help="Disable direct-light occlusion for a no-shadow diagnostic render.",
    )
    parser.add_argument("--steps-per-pair", type=int, default=6)
    parser.add_argument(
        "--smooth-pose-interpolation", action="store_true",
        help="Use cubic/rotation splines and approximately constant-speed resampling.",
    )
    parser.add_argument("--smooth-frame-count", type=int, default=600)
    parser.add_argument(
        "--smooth-keyframe-stride", type=int, default=3,
        help="Use every Nth ordered camera as a spline keyframe to suppress COLMAP jitter.",
    )
    parser.add_argument(
        "--env-rotate", action="store_true",
        help="Rotate the environment while rendering; compatible with both pose modes.",
    )
    parser.add_argument("--env-rotate-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument(
        "--env-rotate-vector", type=float, nargs=3, default=None,
        help="Rotate the environment around this world-space axis instead of x/y/z.",
    )
    parser.add_argument("--env-rotations", type=float, default=1.0)
    parser.add_argument(
        "--env-rotate-deg-per-sec", type=float, default=None,
        help="Fixed environment rotation speed in degrees per second; takes precedence over --env-rotations.",
    )
    parser.add_argument(
        "--fixed-frames", type=int, default=72,
        help="Frame count in fixed-pose mode when --env-rotate is enabled.",
    )
    parser.add_argument(
        "--point-light-orbit", action="store_true",
        help="Orbit a finite point light around a scene-space center.",
    )
    parser.add_argument("--point-light-center", type=float, nargs=3, default=None)
    parser.add_argument("--point-light-axis", type=float, nargs=3, default=None)
    parser.add_argument(
        "--point-light-radial-direction", type=float, nargs=3, default=None,
        help="Initial in-plane direction from the orbit center to the light.",
    )
    parser.add_argument("--point-light-radius", type=float, default=2.5)
    parser.add_argument("--point-light-height", type=float, default=None)
    parser.add_argument("--point-light-elevation-deg", type=float, default=30.0)
    parser.add_argument("--point-light-intensity", type=float, default=10.0)
    parser.add_argument(
        "--point-light-color", type=float, nargs=3,
        default=(1.0, 0.86, 0.68),
    )
    parser.add_argument("--point-light-rotations", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=18.0)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--envmap-max-res", type=int, default=1024)
    parser.add_argument("--diffuse-samples", type=int, default=32)
    parser.add_argument("--light-samples", type=int, default=32)
    parser.add_argument("--light-t-min", type=float, default=0.10)
    parser.add_argument(
        "--ambient-intensity", type=float, default=0.0,
        help="Constant environment fill intensity added in linear RGB.",
    )
    parser.add_argument(
        "--ambient-color", type=float, nargs=3, default=(1.0, 1.0, 1.0),
        help="RGB color of the constant environment fill.",
    )
    parser.add_argument(
        "--depth-occlude-table", action=BooleanOptionalAction, default=None,
        help=("Render table and placed objects separately, then make object "
              "pixels opaque only where the table is geometrically behind "
              "them. By default this is enabled automatically when model_path "
              "contains compose_models_on_table.py layout.json."),
    )
    parser.add_argument(
        "--object-alpha-threshold", type=float, default=0.10,
        help="Legacy hard-mode minimum object alpha.",
    )
    parser.add_argument(
        "--occlusion-depth-epsilon", type=float, default=0.003,
        help="Legacy hard-mode minimum camera-depth separation.",
    )
    parser.add_argument(
        "--table-occlusion-mode", choices=("smooth", "hard"), default="smooth",
        help="Use feathered alpha/depth confidence or the legacy hard cutoff.",
    )
    parser.add_argument("--occlusion-alpha-low", type=float, default=0.70)
    parser.add_argument("--occlusion-alpha-high", type=float, default=0.95)
    parser.add_argument("--occlusion-depth-low", type=float, default=0.005)
    parser.add_argument("--occlusion-depth-high", type=float, default=0.015)
    parser.add_argument(
        "--occlusion-boundary-protection", type=int, default=0,
        help="Erode the correction confidence by this many pixels.",
    )
    parser.add_argument(
        "--occlusion-color-dilation-radius", type=int, default=0,
        help=("Extend reliable interior object color into corrected silhouette "
              "pixels by this radius; zero disables color extension."),
    )
    parser.add_argument(
        "--occlusion-color-source-alpha", type=float, default=0.50,
        help="Minimum alpha from which silhouette extension gathers color.",
    )
    parser.add_argument(
        "--unshadow-low-alpha-table", action="store_true",
        help=("At partially covered object silhouette pixels, use an "
              "unshadowed table layer as the alpha-blending fallback. Full "
              "object interiors and cast shadows outside the silhouette are "
              "unchanged."),
    )
    parser.add_argument("--proxy", default="icosphere320")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    # ``get_combined_args`` intentionally drops ``None`` defaults when no
    # value is present in cfg_args; restore defaults for the diagnostic camera
    # and output options added by this script.
    args.camera_eye = getattr(args, "camera_eye", None)
    args.camera_target = getattr(args, "camera_target", None)
    args.camera_up = getattr(args, "camera_up", None)
    args.camera_end_eye = getattr(args, "camera_end_eye", None)
    args.camera_end_target = getattr(args, "camera_end_target", None)
    args.transition_frames = getattr(args, "transition_frames", 72)
    args.output_buffer = getattr(args, "output_buffer", "render")
    # ``get_combined_args`` drops command-line defaults whose value is None.
    # This option intentionally defaults to None, so restore it before the
    # validation and frame-generation logic below accesses the attribute.
    args.env_rotate_deg_per_sec = getattr(args, "env_rotate_deg_per_sec", None)
    args.env_rotate_vector = getattr(args, "env_rotate_vector", None)
    args.envmap = getattr(args, "envmap", None)
    args.point_light_center = getattr(args, "point_light_center", None)
    args.point_light_axis = getattr(args, "point_light_axis", None)
    args.point_light_radial_direction = getattr(
        args, "point_light_radial_direction", None
    )
    args.point_light_height = getattr(args, "point_light_height", None)
    args.depth_occlude_table = getattr(args, "depth_occlude_table", None)

    if args.point_light_orbit and args.env_rotate:
        parser.error("--point-light-orbit and --env-rotate are separate light modes")
    if args.point_light_orbit:
        if args.point_light_center is None:
            parser.error("--point-light-orbit requires --point-light-center")
        if args.point_light_axis is None:
            parser.error("--point-light-orbit requires --point-light-axis")
        if args.point_light_radial_direction is None:
            parser.error(
                "--point-light-orbit requires --point-light-radial-direction"
            )
    elif args.envmap is None:
        parser.error("--envmap is required unless --point-light-orbit is used")

    if args.steps_per_pair < 1 or args.fixed_frames < 1 or args.transition_frames < 2 or args.fps <= 0:
        parser.error("steps-per-pair, fixed-frames, transition-frames and fps must be positive")
    if args.smooth_frame_count < 2 or args.smooth_keyframe_stride < 1:
        parser.error("smooth-frame-count must be >= 2 and smooth-keyframe-stride must be positive")
    if args.env_rotations < 0:
        parser.error("env-rotations must be non-negative")
    if args.env_rotate_deg_per_sec is not None and args.env_rotate_deg_per_sec < 0:
        parser.error("env-rotate-deg-per-sec must be non-negative")
    if args.env_rotate_vector is not None and np.linalg.norm(args.env_rotate_vector) < 1e-12:
        parser.error("env-rotate-vector cannot be zero")
    if args.point_light_radius <= 0 or args.point_light_intensity <= 0:
        parser.error("point-light radius and intensity must be positive")
    if args.point_light_rotations < 0:
        parser.error("point-light-rotations must be non-negative")
    if args.ambient_intensity < 0:
        parser.error("ambient-intensity must be non-negative")
    if any(value < 0 for value in args.ambient_color):
        parser.error("ambient-color values must be non-negative")
    if not 0.0 < args.object_alpha_threshold < 1.0:
        parser.error("object-alpha-threshold must be between 0 and 1")
    if args.occlusion_depth_epsilon < 0:
        parser.error("occlusion-depth-epsilon must be non-negative")
    if not 0 <= args.occlusion_alpha_low < args.occlusion_alpha_high <= 1:
        parser.error("occlusion alpha bounds must satisfy 0 <= low < high <= 1")
    if not 0 <= args.occlusion_depth_low < args.occlusion_depth_high:
        parser.error("occlusion depth bounds must satisfy 0 <= low < high")
    if args.occlusion_boundary_protection < 0:
        parser.error("occlusion-boundary-protection must be non-negative")
    if args.occlusion_color_dilation_radius < 0:
        parser.error("occlusion-color-dilation-radius must be non-negative")
    if not 0.0 < args.occlusion_color_source_alpha < 1.0:
        parser.error("occlusion-color-source-alpha must be between 0 and 1")

    os.environ["IRGS_GS_BOUNDING_POLYHEDRON"] = args.proxy
    safe_state(args.quiet)
    dataset = model.extract(args)
    # Composed table/object models carry an exact Gaussian component split in
    # layout.json. Enable depth-aware composition for those models by default,
    # while preserving the original single-model renderer everywhere else.
    layout_path = Path(dataset.model_path) / "layout.json"
    if args.depth_occlude_table is None:
        args.depth_occlude_table = layout_path.is_file()
    pipe = pipeline.extract(args)
    pipe.diffuse_sample_num = args.diffuse_samples
    pipe.light_sample_num = args.light_samples
    pipe.light_t_min = args.light_t_min

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    gaussians.build_bvh(static=True)

    table_opacity = None
    object_opacity = None
    table_gaussian_count = None
    if args.depth_occlude_table:
        if not layout_path.is_file():
            parser.error(
                "--depth-occlude-table requires layout.json in model_path")
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        object_gaussian_count = sum(
            int(item["gaussian_count"]) for item in layout.get("objects", []))
        total_gaussian_count = int(gaussians.get_xyz.shape[0])
        declared_total = int(layout.get("total_gaussians", total_gaussian_count))
        if declared_total != total_gaussian_count:
            parser.error(
                "layout/model Gaussian count mismatch: "
                f"layout={declared_total}, model={total_gaussian_count}")
        table_gaussian_count = total_gaussian_count - object_gaussian_count
        if table_gaussian_count <= 0 or object_gaussian_count <= 0:
            parser.error("layout.json does not contain a valid table/object split")
        full_opacity = gaussians.get_opacity.detach()
        table_opacity = full_opacity.clone()
        table_opacity[table_gaussian_count:] = 0.0
        object_opacity = full_opacity.clone()
        object_opacity[:table_gaussian_count] = 0.0
        transition_description = (
            f"alpha={args.occlusion_alpha_low:.3f}..{args.occlusion_alpha_high:.3f}, "
            f"depth={args.occlusion_depth_low:.4f}..{args.occlusion_depth_high:.4f}, "
            f"border={args.occlusion_boundary_protection}px"
            if args.table_occlusion_mode == "smooth"
            else f"alpha>={args.object_alpha_threshold:.3f}, "
                 f"depth epsilon={args.occlusion_depth_epsilon:.4f}"
        )
        print(
            "[depth occlusion] table GS="
            f"{table_gaussian_count}, object GS={object_gaussian_count}, "
            f"mode={args.table_occlusion_mode}, {transition_description}",
            flush=True,
        )
    cameras = sorted_cameras(scene)
    sequence_cameras = sorted_train_cameras(scene) if args.train_only_sequence else cameras
    # The shared pipeline uses -1 to request the middle training view.
    if args.view_id == -1:
        args.view_id = len(sequence_cameras) // 2
    if not 0 <= args.view_id < len(sequence_cameras):
        parser.error(f"view-id must be in [0, {len(sequence_cameras) - 1}]")
    camera = sequence_cameras[args.view_id]

    if (args.camera_eye is None) != (args.camera_target is None):
        parser.error("--camera-eye and --camera-target must be provided together")
    if (args.camera_end_eye is None) != (args.camera_end_target is None):
        parser.error("--camera-end-eye and --camera-end-target must be provided together")
    if args.top_between_training and args.camera_end_eye is None:
        parser.error("--top-between-training requires --camera-end-eye and --camera-end-target")
    if args.camera_eye is not None and args.pose_mode != "fixed":
        parser.error("custom camera coordinates require --pose-mode fixed")

    if args.top_between_training:
        camera_up = args.camera_up if args.camera_up is not None else (0.0, 0.0, 1.0)
        top_pose = look_at_colmap(args.camera_end_eye, args.camera_end_target, camera_up)
        poses = [camera_to_c2w(cam).astype(np.float32) for cam in sequence_cameras]
        def pair_frames(start_pose, end_pose, count, include_start=True, include_end=True):
            pair_rot = Slerp([0.0, 1.0], Rotation.from_matrix(
                np.stack((start_pose[:3, :3], end_pose[:3, :3]), axis=0)))
            first = 0 if include_start else 1
            last = count if include_end else count - 1
            result = []
            for index in range(first, last + 1):
                t = index / count
                c2w = np.eye(4, dtype=np.float32)
                c2w[:3, :3] = pair_rot([t]).as_matrix()[0].astype(np.float32)
                c2w[:3, 3] = ((1.0 - t) * start_pose[:3, 3] + t * end_pose[:3, 3]).astype(np.float32)
                result.append((c2w, 0, 0, t))
            return result

        frames = []
        for index, train_pose in enumerate(poses):
            frames.extend(pair_frames(train_pose, top_pose, args.transition_frames,
                                       include_start=True, include_end=index == len(poses) - 1))
            if index + 1 < len(poses):
                frames.extend(pair_frames(top_pose, poses[index + 1], args.transition_frames,
                                          include_start=False, include_end=True))
    elif args.camera_end_eye is not None:
        camera_up = args.camera_up if args.camera_up is not None else (0.0, 0.0, 1.0)
        start_pose = (look_at_colmap(args.camera_eye, args.camera_target, camera_up)
                      if args.camera_eye is not None else camera_to_c2w(camera).astype(np.float32))
        end_pose = look_at_colmap(args.camera_end_eye, args.camera_end_target, camera_up)
        rotations = Rotation.from_matrix(np.stack((start_pose[:3, :3], end_pose[:3, :3]), axis=0))
        slerp = Slerp([0.0, 1.0], rotations)
        frames = []
        for index in range(args.transition_frames):
            t = index / (args.transition_frames - 1)
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = slerp([t]).as_matrix()[0].astype(np.float32)
            c2w[:3, 3] = ((1.0 - t) * start_pose[:3, 3] + t * end_pose[:3, 3]).astype(np.float32)
            frames.append((c2w, args.view_id, args.view_id, t))
    elif args.pose_mode == "interpolate" and args.smooth_pose_interpolation:
        frames, smooth_keyframe_count = smooth_interpolate_camera_poses(
            sequence_cameras, args.smooth_frame_count, args.smooth_keyframe_stride
        )
    elif args.pose_mode == "interpolate":
        frames = interpolate_camera_poses(sequence_cameras, args.steps_per_pair)
    elif args.camera_eye is not None:
        camera_up = args.camera_up if args.camera_up is not None else (0.0, 0.0, 1.0)
        fixed_pose = look_at_colmap(args.camera_eye, args.camera_target, camera_up)
        frame_count = args.fixed_frames if (args.env_rotate or args.point_light_orbit) else 1
        frames = [(fixed_pose, args.view_id, args.view_id, 0.0)] * frame_count
    else:
        frame_count = args.fixed_frames if (args.env_rotate or args.point_light_orbit) else 1
        fixed_pose = camera_to_c2w(camera).astype(np.float32)
        frames = [(fixed_pose, args.view_id, args.view_id, 0.0)] * frame_count

    point_orbit = None
    if args.point_light_orbit:
        center = np.asarray(args.point_light_center, dtype=np.float64)
        axis = np.asarray(args.point_light_axis, dtype=np.float64)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            parser.error("point-light-axis cannot be zero")
        axis /= axis_norm
        radial = np.asarray(args.point_light_radial_direction, dtype=np.float64)
        radial = radial - np.dot(radial, axis) * axis
        radial_norm = np.linalg.norm(radial)
        if radial_norm < 1e-12:
            parser.error("point-light-radial-direction must not be parallel to the axis")
        radial /= radial_norm
        height = (
            args.point_light_height
            if args.point_light_height is not None
            else args.point_light_radius
            * math.tan(math.radians(args.point_light_elevation_deg))
        )
        initial_position = center + height * axis + args.point_light_radius * radial
        intensity = args.point_light_intensity * np.asarray(
            args.point_light_color, dtype=np.float64
        )
        gaussians.env_map = None
        gaussians.direct_light = PointLight(
            position=initial_position, intensity=intensity, device="cuda"
        ).cuda()
        environment = None
        env_path = None
        point_orbit = {
            "center": center,
            "axis": axis,
            "radial": radial,
            "height": float(height),
        }
    else:
        env_path, environment = load_environment(args.envmap, args.envmap_max_res)
        gaussians.env_map = environment
        gaussians.direct_light = None
    background_value = 1.0 if dataset.white_background else 0.0
    render_kwargs = {
        "pc": gaussians,
        "pipe": pipe,
        "bg_color": torch.full(
            (3,), background_value, dtype=torch.float32, device="cuda"
        ),
        "training": False,
        "relight": True,
        "base_color_scale": torch.ones(3, dtype=torch.float32, device="cuda"),
        "force_visibility_one": bool(args.force_visibility_one),
        "ambient_light": (
            args.ambient_intensity
            * torch.as_tensor(
                args.ambient_color, dtype=torch.float32, device="cuda")
        ),
    }

    output_path = Path(args.output).expanduser().resolve()
    if output_path.suffix.lower() not in (".mp4", ".png"):
        parser.error("output must end in .mp4 or .png")
    if output_path.suffix.lower() == ".png" and len(frames) != 1:
        parser.error("PNG output requires exactly one frame")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    render_ms = []
    occluded_pixel_counts = []
    total_start = time.perf_counter()
    with torch.no_grad():
        for frame_index, (c2w, left, right, t) in enumerate(frames):
            set_camera_pose(camera, c2w)
            if args.point_light_orbit:
                denominator = max(len(frames) - 1, 1)
                angle = (
                    360.0 * args.point_light_rotations
                    * frame_index / denominator
                )
                rotation = rotation_matrix(angle, "z", point_orbit["axis"])
                radial = torch.as_tensor(
                    point_orbit["radial"], dtype=torch.float32, device="cuda"
                )
                center = torch.as_tensor(
                    point_orbit["center"], dtype=torch.float32, device="cuda"
                )
                axis = torch.as_tensor(
                    point_orbit["axis"], dtype=torch.float32, device="cuda"
                )
                light_position = (
                    center + point_orbit["height"] * axis
                    + args.point_light_radius * (rotation @ radial)
                )
                gaussians.direct_light.base_position.copy_(light_position)
            elif args.env_rotate:
                if args.env_rotate_deg_per_sec is not None:
                    angle = args.env_rotate_deg_per_sec * frame_index / args.fps
                else:
                    denominator = max(len(frames) - 1, 1)
                    angle = 360.0 * args.env_rotations * frame_index / denominator
                environment.set_transform(rotation_matrix(
                    angle, args.env_rotate_axis, args.env_rotate_vector
                ))
            else:
                angle = 0.0

            torch.cuda.synchronize()
            render_start = time.perf_counter()
            if args.depth_occlude_table:
                object_package = render_ir(
                    viewpoint_camera=camera,
                    raster_opacity_override=object_opacity,
                    **render_kwargs,
                )
                table_package = render_ir(
                    viewpoint_camera=camera,
                    raster_opacity_override=table_opacity,
                    **render_kwargs,
                )
                object_alpha = object_package["rend_alpha"]
                table_alpha = table_package["rend_alpha"]
                object_depth = object_package["surf_depth"]
                table_depth = table_package["surf_depth"]
                depth_gap = table_depth - object_depth
                valid_depth = (
                    (object_depth > 0.0)
                    & (table_depth > 0.0)
                    & (table_alpha > 1.0 / 255.0)
                )
                table_render = table_package["render"]
                if args.unshadow_low_alpha_table:
                    unshadowed_kwargs = dict(render_kwargs)
                    unshadowed_kwargs["force_visibility_one"] = True
                    clear_table_package = render_ir(
                        viewpoint_camera=camera,
                        raster_opacity_override=table_opacity,
                        **unshadowed_kwargs,
                    )
                    # 4*a*(1-a) is zero for empty/opaque pixels and peaks at
                    # the uncertain a=0.5 silhouette. Restrict it to pixels
                    # where the object is geometrically in front of the table.
                    silhouette_weight = (
                        4.0 * object_alpha * (1.0 - object_alpha)
                        * valid_depth.to(object_alpha.dtype)
                        * (depth_gap > args.occlusion_depth_low).to(
                            object_alpha.dtype)
                    ).clamp(0.0, 1.0)
                    table_render = (
                        table_render * (1.0 - silhouette_weight)
                        + clear_table_package["render"] * silhouette_weight
                    )
                if args.table_occlusion_mode == "hard":
                    correction_confidence = (
                        (object_alpha >= args.object_alpha_threshold)
                        & valid_depth
                        & (depth_gap > args.occlusion_depth_epsilon)
                    ).to(object_alpha.dtype)
                else:
                    alpha_t = (
                        (object_alpha - args.occlusion_alpha_low)
                        / (args.occlusion_alpha_high - args.occlusion_alpha_low)
                    ).clamp(0.0, 1.0)
                    depth_t = (
                        (depth_gap - args.occlusion_depth_low)
                        / (args.occlusion_depth_high - args.occlusion_depth_low)
                    ).clamp(0.0, 1.0)
                    # Cubic smoothstep has zero slope at both transition ends.
                    alpha_confidence = alpha_t.square() * (3.0 - 2.0 * alpha_t)
                    depth_confidence = depth_t.square() * (3.0 - 2.0 * depth_t)
                    correction_confidence = (
                        alpha_confidence * depth_confidence
                        * valid_depth.to(object_alpha.dtype)
                    )
                    if args.occlusion_boundary_protection:
                        kernel = 2 * args.occlusion_boundary_protection + 1
                        correction_confidence = -F.max_pool2d(
                            -correction_confidence,
                            kernel_size=kernel,
                            stride=1,
                            padding=args.occlusion_boundary_protection,
                        )

                # Convert the premultiplied object result back to straight
                # color before hardening reliable foreground pixels.  Pixels
                # outside the depth-tested mask retain ordinary alpha blending.
                object_straight = (
                    object_package["render"]
                    / object_alpha.clamp_min(1e-6)
                )
                # Increasing alpha at a sparse silhouette can expose a dark
                # color estimate produced from only a tiny Gaussian coverage.
                # Extend nearby reliable interior color into those corrected
                # pixels before compositing. This changes neither geometry nor
                # alpha and is confined by correction_confidence below.
                if args.occlusion_color_dilation_radius:
                    color_alpha_t = (
                        (object_alpha - args.occlusion_color_source_alpha)
                        / (1.0 - args.occlusion_color_source_alpha)
                    ).clamp(0.0, 1.0)
                    interior_weight = color_alpha_t.square() * (
                        3.0 - 2.0 * color_alpha_t)
                    kernel = 2 * args.occlusion_color_dilation_radius + 1
                    pooled_weight = F.avg_pool2d(
                        interior_weight, kernel, stride=1,
                        padding=args.occlusion_color_dilation_radius)
                    pooled_color = F.avg_pool2d(
                        object_straight * interior_weight,
                        kernel, stride=1,
                        padding=args.occlusion_color_dilation_radius)
                    extended_color = pooled_color / pooled_weight.clamp_min(1e-6)
                    extension_mix = (
                        correction_confidence
                        * (pooled_weight > 1e-6).to(object_alpha.dtype)
                    )
                    object_straight = (
                        object_straight * (1.0 - extension_mix)
                        + extended_color * extension_mix
                    )
                composed_alpha = object_alpha + correction_confidence * (
                    1.0 - object_alpha)
                composed_render = (
                    object_straight * composed_alpha
                    + table_render * (1.0 - composed_alpha)
                )
                package = dict(object_package)
                package["render"] = composed_render
                package["rend_alpha"] = torch.maximum(
                    composed_alpha, table_alpha * (1.0 - composed_alpha))
                occluded_pixel_counts.append(float(correction_confidence.sum().item()))
            else:
                package = render_ir(viewpoint_camera=camera, **render_kwargs)
            torch.cuda.synchronize()
            render_ms.append((time.perf_counter() - render_start) * 1000.0)
            output_tensor = package[args.output_buffer]
            if args.output_buffer in ("roughness", "visibility"):
                output_tensor = output_tensor.expand(3, -1, -1)
            elif args.output_buffer == "rend_normal":
                output_tensor = 0.5 * (output_tensor + 1.0)
            image = rgb_image(output_tensor)
            if output_path.suffix.lower() == ".png":
                image.save(output_path)
                continue
            if writer is None:
                writer = RawVideoWriter(
                    output_path, image.width, image.height, args.fps, args.crf
                )
            writer.write(image)
            if frame_index % 20 == 0 or frame_index + 1 == len(frames):
                print(
                    f"frame {frame_index + 1}/{len(frames)} "
                    f"camera={left}->{right} t={t:.2f} env={angle:.1f}deg "
                    f"render={render_ms[-1]:.1f}ms",
                    flush=True,
                )
    if writer is not None:
        writer.close()

    report = {
        "mode": "single_object_relighting",
        "model_path": str(Path(dataset.model_path).resolve()),
        "iteration": int(scene.loaded_iter),
        "envmap": str(env_path) if env_path is not None else None,
        "light_type": "point" if args.point_light_orbit else "environment",
        "pose_mode": args.pose_mode,
        "smooth_pose_interpolation": bool(args.smooth_pose_interpolation),
        "smooth_frame_count": args.smooth_frame_count if args.smooth_pose_interpolation else None,
        "smooth_keyframe_stride": args.smooth_keyframe_stride if args.smooth_pose_interpolation else None,
        "smooth_keyframe_count": (
            smooth_keyframe_count if args.smooth_pose_interpolation and args.pose_mode == "interpolate" else None
        ),
        "camera_eye": args.camera_eye,
        "camera_target": args.camera_target,
        "camera_up": args.camera_up,
        "camera_end_eye": args.camera_end_eye,
        "camera_end_target": args.camera_end_target,
        "transition_frames": args.transition_frames if args.camera_end_eye is not None else None,
        "output_buffer": args.output_buffer,
        "force_visibility_one": bool(args.force_visibility_one),
        "environment_rotates": bool(args.env_rotate),
        "environment_axis": args.env_rotate_axis if args.env_rotate else None,
        "environment_axis_vector": (
            args.env_rotate_vector if args.env_rotate else None
        ),
        "environment_rotations": (
            args.env_rotations
            if args.env_rotate and args.env_rotate_deg_per_sec is None
            else None
        ),
        "environment_degrees_per_second": (
            args.env_rotate_deg_per_sec if args.env_rotate else None
        ),
        "point_light_orbit": bool(args.point_light_orbit),
        "point_light_center": getattr(args, "point_light_center", None),
        "point_light_axis": getattr(args, "point_light_axis", None),
        "point_light_radius": (
            args.point_light_radius if args.point_light_orbit else None
        ),
        "point_light_height": (
            point_orbit["height"] if point_orbit is not None else None
        ),
        "point_light_elevation_deg": (
            args.point_light_elevation_deg if args.point_light_orbit else None
        ),
        "point_light_intensity": (
            args.point_light_intensity if args.point_light_orbit else None
        ),
        "point_light_color": (
            args.point_light_color if args.point_light_orbit else None
        ),
        "point_light_rotations": (
            args.point_light_rotations if args.point_light_orbit else None
        ),
        "frame_count": len(frames),
        "fps": args.fps,
        "diffuse_samples": args.diffuse_samples,
        "light_samples": args.light_samples,
        "light_t_min": args.light_t_min,
        "ambient_intensity": args.ambient_intensity,
        "ambient_color": args.ambient_color,
        "depth_occlude_table": bool(args.depth_occlude_table),
        "table_gaussian_count": table_gaussian_count,
        "object_alpha_threshold": (
            args.object_alpha_threshold if args.depth_occlude_table else None
        ),
        "occlusion_depth_epsilon": (
            args.occlusion_depth_epsilon if args.depth_occlude_table else None
        ),
        "table_occlusion_mode": (
            args.table_occlusion_mode if args.depth_occlude_table else None
        ),
        "occlusion_alpha_transition": (
            [args.occlusion_alpha_low, args.occlusion_alpha_high]
            if args.depth_occlude_table and args.table_occlusion_mode == "smooth"
            else None
        ),
        "occlusion_depth_transition": (
            [args.occlusion_depth_low, args.occlusion_depth_high]
            if args.depth_occlude_table and args.table_occlusion_mode == "smooth"
            else None
        ),
        "occlusion_boundary_protection": (
            args.occlusion_boundary_protection
            if args.depth_occlude_table and args.table_occlusion_mode == "smooth"
            else None
        ),
        "occlusion_color_dilation_radius": (
            args.occlusion_color_dilation_radius
            if args.depth_occlude_table else None
        ),
        "occlusion_color_source_alpha": (
            args.occlusion_color_source_alpha
            if args.depth_occlude_table else None
        ),
        "unshadow_low_alpha_table": bool(
            args.unshadow_low_alpha_table and args.depth_occlude_table),
        "mean_occluded_pixels": (
            float(np.mean(occluded_pixel_counts))
            if occluded_pixel_counts else None
        ),
        "proxy": args.proxy,
        "mean_render_ms": float(np.mean(render_ms)),
        "median_render_ms": float(np.median(render_ms)),
        "total_generation_seconds": time.perf_counter() - total_start,
        "output_path": str(output_path),
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Output: {output_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
