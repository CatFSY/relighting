#!/usr/bin/env python3
"""Relight an assembled SO101 scene along an archived object trajectory.

This is the stable public entry for the existing audited assembler.  It keeps
the coordinate conversion and all-rigid OptiX IAS implementation in
``scripts/render_guiji_irgs.py`` while supplying the current production
defaults: 320-face proxies, DS32, LS32, t=0.10, fill0.35/key2500 and rigid IAS.

All options accepted by ``scripts/render_guiji_irgs.py`` remain available.
Use ``--trajectory-env-deg-per-sec`` or ``--trajectory-env-rotations`` to
rotate the environment while the trajectory is moving.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
IMPLEMENTATION = ROOT / "scripts/render_guiji_irgs.py"
DEFAULT_ENV = ROOT / "assets/env_map/pointlike_camera_key_light_fill035_key2500.exr"


def has_option(arguments, option):
    return option in arguments or any(value.startswith(option + "=") for value in arguments)


def add_default(arguments, option, value=None):
    if has_option(arguments, option):
        return
    arguments.append(option)
    if value is not None:
        arguments.append(str(value))


def main():
    arguments = list(sys.argv[1:])
    os.environ.setdefault("IRGS_GS_BOUNDING_POLYHEDRON", "icosphere320")

    add_default(arguments, "--diffuse-samples", 32)
    add_default(arguments, "--light-samples", 32)
    add_default(arguments, "--light-t-min", 0.10)
    add_default(arguments, "--bvh-layout", "component-ias-rigid")
    add_default(arguments, "--envmap", DEFAULT_ENV)

    explicit_render_mode = any(
        has_option(arguments, option)
        for option in (
            "--full-video", "--steps", "--env-rotate-count",
            "--camera-orbit-count", "--save-only",
        )
    )
    if not explicit_render_mode and "--help" not in arguments and "-h" not in arguments:
        arguments.append("--full-video")

    sys.argv = [str(IMPLEMENTATION), *arguments]
    runpy.run_path(str(IMPLEMENTATION), run_name="__main__")


if __name__ == "__main__":
    main()
