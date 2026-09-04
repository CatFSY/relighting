#!/usr/bin/env python3
"""Render a SO101 data-engine handoff with this relighting checkout.

The handoff directory supplies manifests, trajectories, contracts, transforms,
and trained IRGS assets.  Rendering code always comes from this repository;
``common/irgs_runtime`` is deliberately not inspected or executed.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import importlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import List, Optional


ROOT = Path(__file__).resolve().parent
RENDERER = ROOT / "scripts/render_guiji_irgs.py"
LINKS = (
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "lower_arm_link",
    "wrist_link",
    "gripper_link",
    "moving_jaw_so101_v1_link",
)
TRACE_COLUMNS = {
    "step",
    "sim_time_s",
    "stage",
    "object_x_m",
    "object_y_m",
    "object_z_m",
    "object_qw",
    "object_qx",
    "object_qy",
    "object_qz",
    "Rotation__actual_q_rad",
    "Pitch__actual_q_rad",
    "Elbow__actual_q_rad",
    "Wrist_Pitch__actual_q_rad",
    "Wrist_Roll__actual_q_rad",
    "Jaw__actual_q_rad",
}


def resolve_program(value: str) -> Path:
    found = shutil.which(value)
    return Path(found if found else value).expanduser().resolve()


def default_renderer_python() -> str:
    """Prefer an explicitly configured or nearby ``irgs`` environment."""
    configured = os.environ.get("IRGS_PYTHON") or os.environ.get(
        "SO101_IRGS_PYTHON")
    if configured:
        return configured
    executable = Path(sys.executable).resolve()
    if executable.parent.parent.name == "irgs":
        return str(executable)
    sibling_irgs = executable.parent.parent / "envs/irgs/bin/python"
    if sibling_irgs.is_file():
        return str(sibling_irgs)
    return str(executable)


def resolve_urdf(explicit: Optional[Path]) -> Path:
    relative = Path("SO-ARM100/Simulation/SO101/so101_new_calib.urdf")
    candidates = []
    if explicit:
        candidates.append(explicit.expanduser())
    if os.environ.get("SO101_IRGS_URDF"):
        candidates.append(Path(os.environ["SO101_IRGS_URDF"]).expanduser())
    candidates.extend((
        ROOT / relative,
        ROOT.parent / relative,
        ROOT.parent / "DiffReg-PBIR" / relative,
    ))
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate
    checked = "\n  ".join(str(path.resolve()) for path in candidates)
    raise FileNotFoundError(
        "找不到 SO101 URDF；请传 --urdf 或设置 SO101_IRGS_URDF。"
        f"\n已检查：\n  {checked}"
    )


def resolve_trace(episode: Path) -> tuple[Path, str]:
    candidates = (
        ("full", episode / "trajectories/formal_task_lift_trace.csv"),
        ("partial", episode / "trajectories/formal_task_lift_trace.partial.csv"),
        ("generic", episode / "trajectory.csv"),
    )
    for trace_kind, path in candidates:
        if path.is_file() and path.stat().st_size:
            return path.resolve(), trace_kind
    raise FileNotFoundError(f"缺少 formal trajectory CSV：{episode}")


def validate_trace(path: Path) -> None:
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        fieldnames = set(next(reader, ()))
        has_row = next(reader, None) is not None
    missing = sorted(TRACE_COLUMNS - fieldnames)
    if missing:
        raise ValueError(f"{path} 缺少轨迹列：{missing}")
    if not has_row:
        raise ValueError(f"轨迹没有数据行：{path}")


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_progress(path: Path, payload: dict) -> None:
    """Atomically persist per-video state so interrupted runs remain auditable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def progress_payload(batch, episode_id, camera, output, gpu, started_epoch):
    return {
        "schema": "SO101_IRGS_VIDEO_PROGRESS_V1",
        "status": "running",
        "batch": batch.name,
        "episode_id": episode_id,
        "camera": camera,
        "gpu": int(gpu),
        "output_dir": str(output),
        "video": str(expected_video(output, camera)),
        "started_epoch": started_epoch,
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_epoch)),
        "frame_count": 0,
        "total_frames": None,
        "elapsed_s": None,
        "renderer_elapsed_s": None,
    }


def validate_lighting(lighting: dict) -> str:
    def vector(key: str, size: int, positive: bool = False) -> None:
        values = lighting[key]
        if not isinstance(values, list) or len(values) != size:
            raise ValueError(f"irgs_lighting.{key} 必须包含 {size} 个数值")
        numbers = [float(value) for value in values]
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError(f"irgs_lighting.{key} 包含非有限值")
        if positive and not all(value > 0 for value in numbers):
            raise ValueError(f"irgs_lighting.{key} 必须全部为正数")

    light_type = lighting["light_type"]
    if light_type not in ("point", "area"):
        raise ValueError(f"不支持 light_type={light_type!r}")
    vector("light_color_rgb", 3)
    vector("light_position_m", 3)
    intensity = float(lighting["light_intensity"])
    if not math.isfinite(intensity) or intensity < 0:
        raise ValueError("irgs_lighting.light_intensity 必须是非负有限值")
    if light_type == "area":
        vector("light_target_m", 3)
        vector("area_light_size_m", 2, positive=True)
    return light_type


def has_irgs_ply(link_dir: Path) -> bool:
    patterns = (
        "irgs/point_cloud/iteration_*/point_cloud.ply",
        "irgs_full/point_cloud/iteration_*/point_cloud.ply",
        "point_cloud/iteration_*/point_cloud.ply",
    )
    return any(any(link_dir.glob(pattern)) for pattern in patterns)


def expected_video(output: Path, camera: str) -> Path:
    suffix = "_contact_camera" if camera == "contact" else ""
    return output / f"so101_table_cup_trajectory{suffix}_irgs.mp4"


def output_directory(
    output_root: Optional[Path], batch: Path, episode: Path, camera: str
) -> Path:
    if output_root is None:
        return episode / "video" / camera
    return output_root / batch.name / episode.name / camera


def load_batches(handoff: Path, requested: Optional[List[int]]):
    discovered = {}
    for path in handoff.glob("batch_*/trajectory_handoff_manifest.json"):
        suffix = path.parent.name[len("batch_"):]
        if suffix.isdigit():
            discovered[int(suffix)] = path
    if not discovered:
        raise FileNotFoundError(f"没有发现 batch_*/trajectory_handoff_manifest.json：{handoff}")
    numbers = sorted(set(requested)) if requested else sorted(discovered)
    missing = [number for number in numbers if number not in discovered]
    if missing:
        raise FileNotFoundError(f"请求的 batch 不存在：{missing}")

    batches = []
    for number in numbers:
        manifest_path = discovered[number]
        manifest = json.loads(manifest_path.read_text())
        records = manifest.get("records")
        if not isinstance(records, list):
            raise ValueError(f"manifest records 不是列表：{manifest_path}")
        expected = manifest.get("expected_episode_count")
        if expected is not None and int(expected) != len(records):
            raise ValueError(
                f"{manifest_path} 声明 {expected} 条，实际 records 为 {len(records)} 条")
        episode_ids = [record.get("episode_id") for record in records]
        if not all(isinstance(episode_id, str) for episode_id in episode_ids):
            raise ValueError(f"{manifest_path} 包含无效 episode_id")
        duplicates = sorted(
            episode_id for episode_id, count in Counter(episode_ids).items()
            if count > 1
        )
        if duplicates:
            raise ValueError(f"{manifest_path} 包含重复 episode_id：{duplicates}")
        batches.append((manifest_path.parent, records))
    return batches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "使用当前 relighting 仓库渲染 SO101 data-engine handoff；"
            "不会使用 handoff/common/irgs_runtime。"
        )
    )
    parser.add_argument("handoff_root", type=Path, help="SO101 data-engine v3 根目录")
    parser.add_argument("--batch", type=int, action="append",
                        help="只处理指定 batch 编号；可重复，默认自动发现全部 batch")
    parser.add_argument("--episode-id", action="append",
                        help="只处理指定 episode；可重复")
    parser.add_argument("--camera", choices=("main", "contact"), action="append",
                        help="只渲染指定相机；可重复，默认两路相机")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0,
                        help="分片后最多处理多少个 episode；0 表示全部")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--fps", type=float, default=30.0,
                        help="输出帧率；默认 30 fps，可显式覆盖")
    parser.add_argument("--width", type=int, default=None,
                        help="覆盖 contract 中的图像宽度")
    parser.add_argument("--height", type=int, default=None,
                        help="覆盖 contract 中的图像高度")
    parser.add_argument("--bounding-polyhedron",
                        choices=("icosahedron", "icosphere80", "icosphere320", "cube", "octahedron"),
                        default="icosphere320")
    parser.add_argument("--diffuse-samples", type=int, default=32)
    parser.add_argument("--light-samples", type=int, default=32)
    parser.add_argument("--analytic-light-samples", type=int, default=64)
    occlusion = parser.add_mutually_exclusive_group()
    occlusion.add_argument(
        "--depth-occlude-table", dest="depth_occlude_table",
        action="store_true",
        help="启用基于前景/桌面 surf depth 的遮挡修正（正式渲染默认）",
    )
    occlusion.add_argument(
        "--no-depth-occlude-table", dest="depth_occlude_table",
        action="store_false",
        help="关闭深度遮挡修正，用于旧版结果对照",
    )
    parser.set_defaults(depth_occlude_table=True)
    parser.add_argument(
        "--table-occlusion-mode", choices=("smooth", "hard", "strict"),
        default="strict",
        help=(
            "遮挡合成模式：strict 仅在高可信区域完整覆盖，边界/不确定区域 "
            "使用普通 alpha 合成；smooth 保留平滑过渡，hard 保留旧版硬阈值。"
        ),
    )
    parser.add_argument(
        "--depth-compositing-render-mode",
        choices=("legacy-two-pass", "selective"), default="legacy-two-pass",
        help="深度合成渲染路径；默认使用经过验证的双层完整着色",
    )
    parser.add_argument("--object-alpha-threshold", type=float, default=0.10)
    parser.add_argument("--occlusion-depth-epsilon", type=float, default=0.001)
    parser.add_argument("--occlusion-alpha-low", type=float, default=0.70)
    parser.add_argument("--occlusion-alpha-high", type=float, default=0.95)
    parser.add_argument("--occlusion-depth-low", type=float, default=0.005)
    parser.add_argument("--occlusion-depth-high", type=float, default=0.015)
    parser.add_argument("--occlusion-boundary-protection", type=int, default=0)
    parser.add_argument(
        "--python", default=default_renderer_python(),
        help="运行底层 renderer 的 Python；优先 IRGS_PYTHON 和相邻 irgs 环境",
    )
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--robot-root", type=Path)
    parser.add_argument("--table-ply", type=Path)
    parser.add_argument("--object-ply", type=Path)
    parser.add_argument("--envmap", type=Path)
    parser.add_argument("--output-root", type=Path,
                        help="集中式视频输出根目录；默认写入各 episode/video/")
    parser.add_argument("--rerender", action="store_true",
                        help="已有视频和 report 时仍重新渲染")
    parser.add_argument("--keep-video-frames", action="store_true",
                        help="保留编码用 PNG 帧；默认 MP4 成功后删除")
    parser.add_argument(
        "--keep-frame-labels", action="store_true",
        help="保留每帧顶部的 step/timing 文字；正式数据视频默认关闭以提高速度",
    )
    parser.add_argument(
        "--ffmpeg-preset",
        choices=("ultrafast", "superfast", "veryfast", "faster", "fast",
                 "medium", "slow", "slower", "veryslow"),
        default="veryfast",
        help="libx264 编码 preset；正式渲染默认 veryfast",
    )
    parser.add_argument(
        "--video-queue-size", type=int, default=4,
        help="异步 ffmpeg 队列长度；默认 4 帧",
    )
    parser.add_argument(
        "--subprocess-per-video", action="store_true",
        help="禁用常驻 renderer/资产缓存，回退到每个视频启动一个子进程",
    )
    parser.add_argument("--preflight-only", action="store_true",
                        help="只验证全部输入，不创建目录、不启动 CUDA 渲染")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.worker_count <= 0 or not 0 <= args.worker_index < args.worker_count:
        parser.error("worker-index 必须在 [0, worker-count) 内")
    if args.stride <= 0 or args.limit < 0:
        parser.error("stride 必须为正数，limit 不能为负数")
    if args.fps is not None and args.fps <= 0:
        parser.error("fps 必须为正数")
    if args.video_queue_size <= 0:
        parser.error("video-queue-size 必须为正数")
    if (args.width is None) != (args.height is None):
        parser.error("--width 和 --height 必须同时提供")
    if args.width is not None and (args.width <= 0 or args.height <= 0):
        parser.error("width 和 height 必须为正数")
    if args.diffuse_samples < 0 or args.light_samples < 0 or args.analytic_light_samples < 1:
        parser.error("sample 数量无效")
    if not 0.0 < args.object_alpha_threshold < 1.0:
        parser.error("object-alpha-threshold 必须在 (0, 1) 内")
    if args.occlusion_depth_epsilon < 0:
        parser.error("occlusion-depth-epsilon 不能为负数")
    if not 0 <= args.occlusion_alpha_low < args.occlusion_alpha_high <= 1:
        parser.error("occlusion alpha 范围必须满足 0 <= low < high <= 1")
    if not 0 <= args.occlusion_depth_low < args.occlusion_depth_high:
        parser.error("occlusion depth 范围必须满足 0 <= low < high")
    if args.occlusion_boundary_protection < 0:
        parser.error("occlusion-boundary-protection 不能为负数")

    handoff = args.handoff_root.expanduser().resolve()
    if not handoff.is_dir():
        parser.error(f"handoff 根目录不存在：{handoff}")
    try:
        batches = load_batches(handoff, args.batch)
        urdf = resolve_urdf(args.urdf)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    common_assets = handoff / "common/assets"
    table = (args.table_ply or common_assets / "table_basev11_point_cloud.ply").resolve()
    cup = (args.object_ply or common_assets / "paper_cup_v2_21_point_cloud.ply").resolve()
    envmap = (args.envmap or common_assets / "pointlike_camera_key_light_fill035_key2500.exr").resolve()
    robot = (args.robot_root or common_assets / "robot_links").resolve()
    python = resolve_program(args.python)
    output_root = args.output_root.expanduser().resolve() if args.output_root else None

    missing = [path for path in (RENDERER, python, table, cup, envmap, urdf) if not path.is_file()]
    missing.extend(robot / link for link in LINKS if not has_irgs_ply(robot / link))
    if missing:
        parser.error("缺少 renderer 或资产：\n  " + "\n  ".join(str(path) for path in missing))

    selected = []
    known_episode_ids = set()
    for batch, records in batches:
        for record in records:
            episode_id = record.get("episode_id")
            if not isinstance(episode_id, str):
                parser.error(f"{batch.name} manifest 中存在无效 episode_id")
            known_episode_ids.add(episode_id)
            selected.append((batch, record))
    if args.episode_id:
        requested = set(args.episode_id)
        unknown = sorted(requested - known_episode_ids)
        if unknown:
            parser.error(f"episode 不在所选 manifest 中：{unknown}")
        selected = [item for item in selected if item[1]["episode_id"] in requested]
    selected.sort(key=lambda item: (item[0].name, item[1]["episode_id"]))
    selected = selected[args.worker_index::args.worker_count]
    if args.limit:
        selected = selected[:args.limit]

    cameras = list(dict.fromkeys(args.camera or ("main", "contact")))
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["IRGS_GS_BOUNDING_POLYHEDRON"] = args.bounding_polyhedron
    persistent_renderer_main = None
    persistent_renderer_cleanup = None
    persistent_renderer_enabled = False
    persistent_renderer_available = False
    if not args.preflight_only and not args.subprocess_per_video:
        current_python = Path(sys.executable).resolve()
        if current_python != python.resolve():
            print(
                "[cache disabled] --python differs from the handoff launcher; "
                "use the requested interpreter to enable the persistent renderer",
                flush=True,
            )
        else:
            persistent_renderer_available = True

    def ensure_persistent_renderer():
        nonlocal persistent_renderer_main
        nonlocal persistent_renderer_cleanup
        nonlocal persistent_renderer_enabled
        if persistent_renderer_enabled or not persistent_renderer_available:
            return
        # Import CUDA only when this worker encounters its first unfinished
        # video. A resume scan containing only completed outputs stays cheap.
        os.environ.update({
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "IRGS_GS_BOUNDING_POLYHEDRON": args.bounding_polyhedron,
        })
        renderer_module = importlib.import_module("scripts.render_guiji_irgs")
        persistent_renderer_main = renderer_module.main
        persistent_renderer_cleanup = renderer_module.cleanup_video_writers
        persistent_renderer_enabled = True
        print(
            f"[persistent renderer] GPU {args.gpu}: assets/model/IAS cached",
            flush=True,
        )
    failures = []
    trace_kinds = Counter()
    validated = rendered = skipped = 0

    for batch, record in selected:
        episode_id = record["episode_id"]
        episode = batch / "episodes" / episode_id
        try:
            trace, trace_kind = resolve_trace(episode)
            validate_trace(trace)
            contract_path = episode / "configs/execution_contract.json"
            transforms_path = episode / "provenance/complete_coordinate_transforms.json"
            contract = json.loads(contract_path.read_text())
            json.loads(transforms_path.read_text())
            physics_rate = float(contract["physics_rate_hz"])
            if physics_rate <= 0:
                raise ValueError("physics_rate_hz 必须为正数")
            lighting = record["irgs_lighting"]
            light_type = validate_lighting(lighting)
            for camera in cameras:
                camera_key = "camera" if camera == "main" else "contact_camera"
                resolution = contract[camera_key]["resolution"]
                if len(resolution) != 2 or min(resolution) <= 0:
                    raise ValueError(f"{camera_key}.resolution 无效：{resolution}")
            trace_kinds[trace_kind] += 1
            validated += 1
        except Exception as error:
            failures.append({"batch": batch.name, "episode_id": episode_id,
                             "reason": str(error)})
            continue

        if args.preflight_only:
            continue

        for camera in cameras:
            output = output_directory(output_root, batch, episode, camera)
            video = expected_video(output, camera)
            report = output / "video_report.json"
            progress = output / "render_progress.json"
            if not args.rerender and video.is_file() and report.is_file():
                # Preserve the original GPU and timestamps. Older code rewrote
                # them with whichever worker happened to scan this output.
                skipped += 1
                print(f"[skip] {batch.name}/{episode_id}/{camera}", flush=True)
                continue

            camera_key = "camera" if camera == "main" else "contact_camera"
            contract_width, contract_height = contract[camera_key]["resolution"]
            width = args.width if args.width is not None else contract_width
            height = args.height if args.height is not None else contract_height
            fps = args.fps if args.fps is not None else physics_rate / args.stride
            output.mkdir(parents=True, exist_ok=True)
            ensure_persistent_renderer()
            started_epoch = time.time()
            write_progress(
                progress,
                progress_payload(
                    batch, episode_id, camera, output, args.gpu, started_epoch),
            )
            command = [
                str(python), str(RENDERER),
                "--trajectory-dir", str(episode),
                "--trace-csv", str(trace),
                "--table-ply", str(table),
                "--object-ply", str(cup),
                "--object-name", "cup",
                "--envmap", str(envmap),
                "--robot-root", str(robot),
                "--urdf", str(urdf),
                "--full-video",
                "--stride", str(args.stride),
                "--fps", str(fps),
                "--width", str(width),
                "--height", str(height),
                "--diffuse-samples", str(args.diffuse_samples),
                "--light-samples", str(args.light_samples),
                "--analytic-light-samples", str(args.analytic_light_samples),
                "--light-t-min", "0.10",
                "--base-color-min", "0.03",
                "--camera-name", camera,
                "--bvh-layout", "component-ias-rigid",
                "--out", str(output),
                "--light-type", light_type,
                "--light-color", *(str(value) for value in lighting["light_color_rgb"]),
                "--light-intensity", str(lighting["light_intensity"]),
            ]
            if not args.keep_video_frames:
                command.append("--stream-video")
            command.extend((
                "--ffmpeg-preset", args.ffmpeg_preset,
                "--video-queue-size", str(args.video_queue_size),
                "--physical-gpu-id", str(args.gpu),
            ))
            command.append("--frame-label" if args.keep_frame_labels else "--no-frame-label")
            if persistent_renderer_enabled:
                command.append("--persistent-model-cache")
            if args.depth_occlude_table:
                command.extend((
                    "--depth-occlude-table",
                    "--table-occlusion-mode", args.table_occlusion_mode,
                    "--depth-compositing-render-mode",
                    args.depth_compositing_render_mode,
                    "--object-alpha-threshold", str(args.object_alpha_threshold),
                    "--occlusion-depth-epsilon", str(args.occlusion_depth_epsilon),
                    "--occlusion-alpha-low", str(args.occlusion_alpha_low),
                    "--occlusion-alpha-high", str(args.occlusion_alpha_high),
                    "--occlusion-depth-low", str(args.occlusion_depth_low),
                    "--occlusion-depth-high", str(args.occlusion_depth_high),
                    "--occlusion-boundary-protection",
                    str(args.occlusion_boundary_protection),
                ))
            if light_type == "point":
                command.extend((
                    "--point-light-position-m",
                    *(str(value) for value in lighting["light_position_m"]),
                ))
            else:
                command.extend((
                    "--area-light-center-m",
                    *(str(value) for value in lighting["light_position_m"]),
                    "--area-light-target-m",
                    *(str(value) for value in lighting["light_target_m"]),
                    "--area-light-size-m",
                    *(str(value) for value in lighting["area_light_size_m"]),
                ))

            log = output.parent / f"{camera}.log"
            print(
                f"[render] {batch.name}/{episode_id}/{camera} -> {output}",
                flush=True,
            )
            with log.open("a", encoding="utf-8") as stream:
                stream.write(f"\n$ {shlex.join(command)}\n")
                stream.flush()
                if persistent_renderer_enabled:
                    returncode = 0
                    try:
                        with contextlib.redirect_stdout(stream), \
                                contextlib.redirect_stderr(stream):
                            persistent_renderer_main(command[2:])
                    except Exception:
                        traceback.print_exc(file=stream)
                        returncode = 1
                    finally:
                        persistent_renderer_cleanup()
                        stream.flush()
                        gc.collect()
                else:
                    result = subprocess.run(
                        command, cwd=ROOT, env=environment,
                        stdout=stream, stderr=subprocess.STDOUT,
                    )
                    returncode = result.returncode
            if returncode:
                try:
                    failed_progress = json.loads(progress.read_text())
                except (OSError, ValueError, json.JSONDecodeError):
                    failed_progress = progress_payload(
                        batch, episode_id, camera, output, args.gpu, started_epoch)
                failed_progress.update({
                    "status": "failed",
                    "finished_at_utc": utc_timestamp(),
                    "elapsed_s": time.time() - started_epoch,
                })
                write_progress(progress, failed_progress)
                print(
                    f"[failed] {batch.name}/{episode_id}/{camera}; log={log}",
                    flush=True,
                )
                failures.append({
                    "batch": batch.name,
                    "episode_id": episode_id,
                    "camera": camera,
                    "returncode": returncode,
                    "log": str(log),
                })
            else:
                try:
                    completed_report = json.loads(report.read_text())
                    completed_timings = completed_report.get("timings", [])
                except (OSError, ValueError, json.JSONDecodeError):
                    completed_report = {}
                    completed_timings = []
                completed_progress = progress_payload(
                    batch, episode_id, camera, output, args.gpu, started_epoch)
                completed_progress.update({
                    "status": "completed",
                    "finished_at_utc": utc_timestamp(),
                    "frame_count": len(completed_timings),
                    "total_frames": len(completed_timings),
                    "elapsed_s": time.time() - started_epoch,
                    "renderer_elapsed_s": completed_report.get("elapsed_s"),
                })
                write_progress(progress, completed_progress)
                rendered += 1
                print(f"[done] {batch.name}/{episode_id}/{camera}", flush=True)

    summary = {
        "handoff_root": str(handoff),
        "renderer": str(RENDERER),
        "ignored_runtime": str(handoff / "common/irgs_runtime"),
        "batches": [batch.name for batch, _ in batches],
        "assigned_episodes": len(selected),
        "validated_episodes": validated,
        "trajectory_types": dict(sorted(trace_kinds.items())),
        "requested_cameras": cameras,
        "rendered_camera_videos": rendered,
        "skipped_existing_camera_videos": skipped,
        "output_root": str(output_root) if output_root else "<episode>/video",
        "output_layout": (
            "<output_root>/<batch>/<episode>/<camera>"
            if output_root else "<episode>/video/<camera>"
        ),
        "resume_rule": "skip when both video and video_report.json exist",
        "persistent_renderer_enabled": persistent_renderer_enabled,
        "frame_labels": args.keep_frame_labels,
        "ffmpeg_preset": args.ffmpeg_preset,
        "video_queue_size": args.video_queue_size,
        "preflight_only": args.preflight_only,
        "failures": failures,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
