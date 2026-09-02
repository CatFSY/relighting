#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path


def stats(values):
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": ordered[0],
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--discard", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    frames = report["timings"][args.discard:]
    result = {
        "source_report": str(args.report),
        "discarded_warmup_frames": args.discard,
        "measured_frames": len(frames),
        "configuration": {
            key: report.get(key) for key in (
                "resolution", "gaussian_counts", "diffuse_samples",
                "light_samples", "diffuse_sampling_mode",
                "light_sampling_mode", "render_ray_budget",
                "fg_lut_query_layout", "fg_lut_tile_width",
            )
        },
    }
    for key in ("bvh_s", "render_s", "trajectory_state_to_render_complete_s",
                "render_cuda_ms", "visible_pixels",
                "peak_memory_allocated_bytes", "peak_memory_reserved_bytes"):
        result[key] = stats([frame[key] for frame in frames])
    latency = result["trajectory_state_to_render_complete_s"]
    result["trajectory_to_rgb_fps"] = {
        "from_mean_latency": 1.0 / latency["mean"],
        "from_median_latency": 1.0 / latency["median"],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
