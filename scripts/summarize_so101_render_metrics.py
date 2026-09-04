#!/usr/bin/env python3
"""Collect per-video/per-GPU render metrics, including interrupted work.

This script intentionally uses only the Python standard library so it can run
after a worker failure, even when the rendering environment is unavailable.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path


FRAME_RE = re.compile(r"\bframe\s+(\d+)\s*/\s*(\d+)")
RENDER_RE = re.compile(r"\[render\].*? -> (.+?)\s*$")
GPU_LOG_RE = re.compile(r"worker_\d+_gpu_(\d+)\.log$")


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def frame_progress(log_path: Path):
    frame_count = 0
    total_frames = None
    try:
        stream = log_path.open(encoding="utf-8", errors="replace")
    except OSError:
        return frame_count, total_frames
    with stream:
        for line in stream:
            match = FRAME_RE.search(line)
            if match:
                frame_count = max(frame_count, int(match.group(1)))
                total_frames = int(match.group(2))
    return frame_count, total_frames


def infer_identity(path: Path):
    parts = list(path.parts)
    batch = next((part for part in parts if part.startswith("batch_")), None)
    episode = next((part for part in parts if part.startswith("episode_")), None)
    camera = path.name
    return batch, episode, camera


def collect(handoff: Path, global_status: str, scan_roots):
    records = {}
    now = time.time()
    worker_gpu_by_output = {}
    worker_log_paths = set()
    for root in [handoff, *scan_roots]:
        worker_log_paths.update(root.rglob("worker_*_gpu_*.log"))
    for worker_log in worker_log_paths:
        match = GPU_LOG_RE.search(worker_log.name)
        if not match:
            continue
        gpu = int(match.group(1))
        try:
            lines = worker_log.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            rendered = RENDER_RE.search(line)
            if rendered:
                worker_gpu_by_output[str(Path(rendered.group(1)).resolve())] = gpu
    progress_paths = set()
    for root in [handoff, *scan_roots]:
        progress_paths.update(root.rglob("render_progress.json"))
    for progress_path in sorted(progress_paths):
        progress = load_json(progress_path)
        if not isinstance(progress, dict):
            continue
        output = progress_path.parent
        report_path = output / "video_report.json"
        report = load_json(report_path)
        batch, episode, camera = infer_identity(output)
        record = {
            "batch": progress.get("batch", batch),
            "episode_id": progress.get("episode_id", episode),
            "camera": progress.get("camera", camera),
            "gpu": progress.get("gpu"),
            "status": progress.get("status", "running"),
            "output_dir": str(output),
            "video": progress.get("video"),
            "started_at_utc": progress.get("started_at_utc"),
            "frame_count": int(progress.get("frame_count") or 0),
            "total_frames": progress.get("total_frames"),
            "elapsed_s": progress.get("elapsed_s"),
            "renderer_elapsed_s": progress.get("renderer_elapsed_s"),
        }
        if record["gpu"] is None:
            record["gpu"] = worker_gpu_by_output.get(str(output.resolve()))
        if isinstance(report, dict):
            timings = report.get("timings", [])
            report_gpu = report.get("physical_gpu_id")
            record.update({
                "gpu": report_gpu if report_gpu is not None else record["gpu"],
                "status": "completed",
                "frame_count": len(timings) if isinstance(timings, list) else record["frame_count"],
                "total_frames": len(timings) if isinstance(timings, list) else record["total_frames"],
                "elapsed_s": (
                    record["elapsed_s"]
                    if record["elapsed_s"] is not None
                    else report.get("elapsed_s")
                ),
                "renderer_elapsed_s": report.get(
                    "elapsed_s", record["renderer_elapsed_s"]),
                "video": report.get("video", record["video"]),
            })
        elif record["status"] == "running":
            log_path = output.parent / f"{record['camera']}.log"
            count, total = frame_progress(log_path)
            record["frame_count"] = max(record["frame_count"], count)
            record["total_frames"] = total or record["total_frames"]
            started = progress.get("started_epoch")
            if started is not None:
                record["elapsed_s"] = max(0.0, now - float(started))
            record["status"] = "interrupted" if global_status == "interrupted" else global_status
        key = str(progress_path)
        records[key] = record

    # Older completed outputs may not have a progress marker. Include them so
    # the aggregate remains useful while a long-running dataset is upgraded.
    report_paths = set()
    for root in [handoff, *scan_roots]:
        report_paths.update(root.rglob("video_report.json"))
    for report_path in sorted(report_paths):
        progress_path = report_path.parent / "render_progress.json"
        if str(progress_path) in records:
            continue
        report = load_json(report_path)
        if not isinstance(report, dict):
            continue
        batch, episode, camera = infer_identity(report_path.parent)
        timings = report.get("timings", [])
        records[str(progress_path)] = {
            "batch": batch,
            "episode_id": episode,
            "camera": camera,
            "gpu": (
                report.get("physical_gpu_id")
                if report.get("physical_gpu_id") is not None
                else worker_gpu_by_output.get(str(report_path.parent.resolve()))
            ),
            "status": "completed",
            "output_dir": str(report_path.parent),
            "video": report.get("video"),
            "started_at_utc": None,
            "frame_count": len(timings) if isinstance(timings, list) else None,
            "total_frames": len(timings) if isinstance(timings, list) else None,
            "elapsed_s": report.get("elapsed_s"),
            "renderer_elapsed_s": report.get("elapsed_s"),
        }

    videos = list(records.values())
    per_gpu = defaultdict(list)
    for record in videos:
        per_gpu[str(record["gpu"] if record["gpu"] is not None else "unknown")].append(record)

    def aggregate(items):
        elapsed = [float(x["elapsed_s"]) for x in items if x.get("elapsed_s") is not None]
        frames = [int(x["frame_count"]) for x in items if x.get("frame_count") is not None]
        paired = [
            (int(x["frame_count"]), float(x["elapsed_s"]))
            for x in items
            if x.get("frame_count") is not None
            and x.get("elapsed_s") is not None
            and float(x["elapsed_s"]) > 0
        ]
        total_paired_frames = sum(frame_count for frame_count, _ in paired)
        total_paired_elapsed = sum(seconds for _, seconds in paired)
        return {
            "video_count": len(items),
            "completed_count": sum(x.get("status") == "completed" for x in items),
            "interrupted_count": sum(x.get("status") == "interrupted" for x in items),
            "failed_count": sum(x.get("status") == "failed" for x in items),
            "mean_frame_count": sum(frames) / len(frames) if frames else None,
            "total_frame_count": sum(frames) if frames else None,
            "mean_elapsed_s": sum(elapsed) / len(elapsed) if elapsed else None,
            "total_elapsed_s": sum(elapsed) if elapsed else None,
            "aggregate_frames_per_second": (
                total_paired_frames / total_paired_elapsed
                if total_paired_elapsed else None
            ),
            "mean_video_frames_per_second": (
                sum(frame_count / seconds for frame_count, seconds in paired)
                / len(paired) if paired else None
            ),
        }

    return {
        "schema": "SO101_IRGS_RENDER_METRICS_V1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "collection_status": global_status,
        "handoff_root": str(handoff),
        "videos": sorted(videos, key=lambda x: (x.get("batch") or "", x.get("episode_id") or "", x.get("camera") or "")),
        "summary": aggregate(videos),
        "per_gpu": {gpu: aggregate(items) for gpu, items in sorted(per_gpu.items())},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("handoff_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, action="append", default=[])
    parser.add_argument("--status", choices=("completed", "failed", "interrupted"), default="completed")
    args = parser.parse_args()
    metrics = collect(
        args.handoff_root.expanduser().resolve(), args.status,
        [root.expanduser().resolve() for root in args.scan_root],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(metrics["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
