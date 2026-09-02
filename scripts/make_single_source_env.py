#!/usr/bin/env python
"""Create a lat-long HDR environment with one finite-area light source."""

import argparse
from pathlib import Path

import numpy as np
import pyexr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--elevation-deg", type=float, default=35.0)
    parser.add_argument("--azimuth-deg", type=float, default=0.0)
    parser.add_argument("--radius-deg", type=float, default=5.0)
    parser.add_argument("--intensity", type=float, default=35.0)
    parser.add_argument("--ambient", type=float, default=0.025)
    parser.add_argument("--color", default="1.0,0.86,0.68")
    parser.add_argument(
        "--direction", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help="Explicit world-space direction toward the light; overrides elevation/azimuth.",
    )
    args = parser.parse_args()

    light_color = np.asarray([float(x) for x in args.color.split(",")], dtype=np.float32)
    if light_color.shape != (3,):
        raise ValueError("--color must contain three comma-separated values")

    # Match IRGS EnvLight's lat-long convention:
    # d=(sin(theta)sin(phi), cos(theta), -sin(theta)cos(phi)); Y is the pole.
    v = (np.arange(args.height, dtype=np.float32) + 0.5) / args.height
    u = (np.arange(args.width, dtype=np.float32) + 0.5) / args.width
    theta = v[:, None] * np.pi
    phi = (u[None, :] * 2.0 - 1.0) * np.pi
    directions = np.stack(
        [
            np.sin(theta) * np.sin(phi),
            np.broadcast_to(np.cos(theta), (args.height, args.width)),
            -np.sin(theta) * np.cos(phi),
        ],
        axis=-1,
    )

    if args.direction is not None:
        light_direction = np.asarray(args.direction, dtype=np.float32)
        norm = np.linalg.norm(light_direction)
        if norm < 1e-8:
            raise ValueError("--direction cannot be zero")
        light_direction /= norm
    else:
        elevation = np.deg2rad(args.elevation_deg)
        azimuth = np.deg2rad(args.azimuth_deg)
        light_direction = np.array(
            [
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
                -np.cos(elevation) * np.cos(azimuth),
            ],
            dtype=np.float32,
        )
    angular_distance = np.arccos(
        np.clip(np.sum(directions * light_direction, axis=-1), -1.0, 1.0)
    )
    sigma = np.deg2rad(args.radius_deg) / 2.355  # radius interpreted as FWHM
    spot = np.exp(-0.5 * (angular_distance / sigma) ** 2).astype(np.float32)

    image = np.full((args.height, args.width, 3), args.ambient, dtype=np.float32)
    image += args.intensity * spot[..., None] * light_color[None, None, :]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    pyexr.write(str(output), image)
    print(
        f"saved {output} shape={image.shape} min={image.min():.6f} "
        f"max={image.max():.6f} mean={image.mean():.6f} "
        f"direction={light_direction.tolist()}"
    )


if __name__ == "__main__":
    main()
