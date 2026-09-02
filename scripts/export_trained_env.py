#!/usr/bin/env python3
"""Export an IRGS point_cloud1.map environment to a lat-long EXR."""

import argparse
from pathlib import Path

import numpy as np
import pyexr
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True, dest="map_path")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.map_path, map_location="cpu", weights_only=False)
    base = checkpoint["state_dict"]["base"].detach().float()
    activation = checkpoint.get("activation", "exp")
    if activation == "exp":
        image = torch.exp(base)
    elif activation == "sigmoid":
        image = torch.sigmoid(base)
    elif activation == "none":
        image = base
    else:
        raise ValueError(f"Unsupported environment activation: {activation}")

    image = image.clamp_min(0).numpy().astype(np.float32)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    pyexr.write(str(output), image)
    print(
        f"saved {output} shape={image.shape} activation={activation} "
        f"min={image.min():.6g} max={image.max():.6g} mean={image.mean():.6g}"
    )


if __name__ == "__main__":
    main()
