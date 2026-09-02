#!/usr/bin/env python3
"""Render a cube-based USD asset into IRGS Synthetic4Relight layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
from pxr import Usd, UsdGeom


CORNERS = np.array(
    [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)],
    dtype=np.float64,
)
TRIANGLES = np.array(
    [
        [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
        [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
        [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
    ], dtype=np.int32,
)


def normalize(v):
    return v / np.linalg.norm(v)


def look_at_opengl(eye, target):
    forward = normalize(target - eye)
    right = normalize(np.cross(forward, np.array([0.0, 0.0, 1.0])))
    up = normalize(np.cross(right, forward))
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = np.column_stack([right, up, -forward])
    c2w[:3, 3] = eye
    return c2w


def srgb(linear):
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear <= 0.0031308, 12.92 * linear,
                    1.055 * np.power(linear, 1.0 / 2.4) - 0.055)


def load_usd_cubes(path):
    stage = Usd.Stage.Open(str(path))
    if not stage:
        raise RuntimeError(f"cannot open USD: {path}")
    cache = UsdGeom.XformCache()
    meshes = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Cube):
            continue
        cube = UsdGeom.Cube(prim)
        size = float(cube.GetSizeAttr().Get() or 1.0)
        matrix = cache.GetLocalToWorldTransform(prim)
        vertices = np.array(
            [matrix.Transform(tuple(point * size)) for point in CORNERS], dtype=np.float64
        )
        colors = cube.GetDisplayColorAttr().Get()
        color = np.array(colors[0] if colors else (0.7, 0.7, 0.7), dtype=np.float32)
        meshes.append((vertices, TRIANGLES.copy(), color, str(prim.GetPath())))
    if not meshes:
        raise RuntimeError("USD contains no UsdGeom.Cube geometry")
    return stage, meshes


def render(meshes, c2w, width, height, fovx):
    w2c = np.linalg.inv(c2w)
    focal = 0.5 * width / math.tan(0.5 * fovx)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    image = np.zeros((height, width, 3), dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.uint8)
    lights = [(normalize(np.array([0.4, -0.5, 1.0])), 0.65),
              (normalize(np.array([-0.6, 0.2, 0.7])), 0.25)]

    for vertices, triangles, base_color, _ in meshes:
        hom = np.column_stack([vertices, np.ones(len(vertices))])
        cam = (hom @ w2c.T)[:, :3]
        z = -cam[:, 2]
        uv = np.column_stack([
            focal * cam[:, 0] / z + width * 0.5,
            -focal * cam[:, 1] / z + height * 0.5,
        ])
        for tri in triangles:
            if np.any(z[tri] <= 1e-5):
                continue
            pts = uv[tri]
            x0 = max(0, int(math.floor(pts[:, 0].min())))
            x1 = min(width - 1, int(math.ceil(pts[:, 0].max())))
            y0 = max(0, int(math.floor(pts[:, 1].min())))
            y1 = min(height - 1, int(math.ceil(pts[:, 1].max())))
            if x1 < x0 or y1 < y0:
                continue
            a, b, c = pts
            denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(denom) < 1e-9:
                continue
            yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
            px, py = xx + 0.5, yy + 0.5
            wa = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / denom
            wb = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / denom
            wc = 1.0 - wa - wb
            inside = (wa >= -1e-6) & (wb >= -1e-6) & (wc >= -1e-6)
            zz = wa * z[tri[0]] + wb * z[tri[1]] + wc * z[tri[2]]
            region_depth = depth[y0:y1 + 1, x0:x1 + 1]
            update = inside & (zz < region_depth)
            if not np.any(update):
                continue
            normal = normalize(np.cross(vertices[tri[1]] - vertices[tri[0]],
                                        vertices[tri[2]] - vertices[tri[0]]))
            illumination = 0.18 + sum(weight * abs(float(np.dot(normal, direction)))
                                      for direction, weight in lights)
            color = np.clip(base_color * illumination, 0.0, 1.0)
            region_depth[update] = zz[update]
            image[y0:y1 + 1, x0:x1 + 1][update] = color
            mask[y0:y1 + 1, x0:x1 + 1][update] = 255
    return image, mask, depth


def camera_pose(index, count, target, radius, elevation_min_deg, elevation_max_deg):
    golden = math.pi * (3.0 - math.sqrt(5.0))
    azimuth = index * golden
    # Deterministic elevation coverage over the requested spherical band.
    frac = ((index * 37) % count + 0.5) / count
    elevation = math.radians(
        elevation_min_deg + (elevation_max_deg - elevation_min_deg) * frac
    )
    eye = target + radius * np.array([
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    ])
    return look_at_opengl(eye, target)


def write_split(out, split, count, meshes, width, height, fovx, target, radius,
                elevation_min_deg, elevation_max_deg):
    folder = out / split
    folder.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(count):
        c2w = camera_pose(i + (0 if split == "train" else 1009), count, target,
                          radius, elevation_min_deg, elevation_max_deg)
        image, mask, _ = render(meshes, c2w, width, height, fovx)
        tag = f"{i:03d}"
        if split == "train":
            ok = cv2.imwrite(str(folder / f"{tag}_rgb.exr"), image[..., ::-1].astype(np.float32))
            if not ok:
                raise RuntimeError("OpenCV failed to write EXR")
            cv2.imwrite(str(folder / f"{tag}_mask.png"), mask)
        else:
            rgba = np.dstack([(srgb(image) * 255.0 + 0.5).astype(np.uint8), mask])
            cv2.imwrite(str(folder / f"{tag}_rgba.png"), rgba[..., [2, 1, 0, 3]])
        frames.append({"file_path": f"{split}/{tag}", "transform_matrix": c2w.tolist()})
        print(f"{split} {i + 1}/{count}", flush=True)
    (out / f"transforms_{split}.json").write_text(
        json.dumps({"camera_angle_x": fovx, "frames": frames}, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-views", type=int, default=100)
    parser.add_argument("--test-views", type=int, default=1)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--fovx-deg", type=float, default=48.0)
    parser.add_argument("--elevation-min-deg", type=float, default=12.0)
    parser.add_argument("--elevation-max-deg", type=float, default=52.0)
    parser.add_argument("--radius-scale", type=float, default=2.15,
                        help="Camera radius as a multiple of the asset AABB diagonal.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {args.out}; pass --overwrite")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    if not -89.0 < args.elevation_min_deg < args.elevation_max_deg < 89.0:
        raise ValueError("elevation range must satisfy -89 < min < max < 89 degrees")
    stage, meshes = load_usd_cubes(args.usd)
    all_vertices = np.concatenate([m[0] for m in meshes])
    bounds = np.stack([all_vertices.min(0), all_vertices.max(0)])
    target = 0.5 * (bounds[0] + bounds[1])
    if args.radius_scale <= 0:
        raise ValueError("radius-scale must be positive")
    radius = args.radius_scale * np.linalg.norm(bounds[1] - bounds[0])
    fovx = math.radians(args.fovx_deg)
    write_split(args.out, "train", args.train_views, meshes, args.width, args.height,
                fovx, target, radius, args.elevation_min_deg, args.elevation_max_deg)
    write_split(args.out, "test", args.test_views, meshes, args.width, args.height,
                fovx, target, radius, args.elevation_min_deg, args.elevation_max_deg)
    (args.out / "test_rli").mkdir()
    digest = hashlib.sha256(args.usd.read_bytes()).hexdigest()
    report = {
        "source_usd": str(args.usd.resolve()), "source_sha256": digest,
        "usd_up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "bounds_m": bounds.tolist(), "target_m": target.tolist(),
        "camera_radius_m": float(radius), "fovx_deg": args.fovx_deg,
        "camera_radius_scale": args.radius_scale,
        "elevation_range_deg": [args.elevation_min_deg, args.elevation_max_deg],
        "resolution": [args.width, args.height],
        "train_views": args.train_views, "test_views": args.test_views,
        "geometry_prims": [m[3] for m in meshes],
        "lighting": "two fixed directional lights plus ambient; linear EXR output",
    }
    (args.out / "generation_report.json").write_text(json.dumps(report, indent=2))
    print(f"DONE -> {args.out}")


if __name__ == "__main__":
    main()
