#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""整臂 GT 渲染: 直接用正确 URDF 几何 (sxyz) + FK 组装, pyrender 平色渲染.

与 assemble_so101.py 的 IRGS 渲染用同一套 orbit_views 相机, 便于并排对比.
只依赖 trimesh + pyrender (refdgs env), 不碰 IRGS/torch.

用法 (refdgs env, PYOPENGL_PLATFORM=egl):
  python gt_arm_render.py [--views 6] [--elevs 15,35,55] [--d-scale 2.4]
                          [--out ...] [--links ...]
输出:
  <out>/gt_%02d.png + <out>/gt_grid_18.png
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pyrender
import trimesh
from PIL import Image
from trimesh.transformations import euler_matrix, rotation_matrix

URDF = "/amax/home/fengshuangyu/relighting/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
ASSETS = os.path.join(os.path.dirname(URDF), "assets")
ALL_LINKS = ["base_link", "shoulder_link", "upper_arm_link", "lower_arm_link",
             "wrist_link", "gripper_link", "moving_jaw_so101_v1_link"]
SCALE = 4.0
FOV_Y = np.deg2rad(45.0)
COLORS = {"3d_printed": np.array([1.0, 0.82, 0.12]), "sts3215": np.array([0.1, 0.1, 0.1])}


def parse_urdf(path):
    root = ET.parse(path).getroot()
    joints = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = np.array([float(v) for v in o.attrib["xyz"].split()]) if o is not None else np.zeros(3)
        rpy = np.array([float(v) for v in o.attrib["rpy"].split()]) if o is not None else np.zeros(3)
        a = j.find("axis")
        axis = np.array([float(v) for v in a.attrib["xyz"].split()]) if a is not None else np.array([0.0, 0.0, 1.0])
        joints[j.attrib["name"]] = {"parent": j.find("parent").attrib["link"],
                                    "child": j.find("child").attrib["link"],
                                    "xyz": xyz, "rpy": rpy, "axis": axis}
    return joints


def fk(joints):
    """0 姿态 FK。joint rpy 固定轴语义 'sxyz' (与 build 脚本一致)。"""
    T = {"base_link": np.eye(4)}
    while True:
        added = 0
        for name, j in joints.items():
            if j["child"] in T or j["parent"] not in T:
                continue
            T_origin = euler_matrix(j["rpy"][0], j["rpy"][1], j["rpy"][2], "sxyz")
            T_origin[:3, 3] = j["xyz"]
            T[j["child"]] = T[j["parent"]] @ T_origin
            added += 1
        if not added:
            break
    return T


def build_arm_meshes(links, T_fk):
    """返回 (trimesh 列表, 全部顶点) — 每个 visual mesh 平色, 放大 SCALE 系。"""
    root = ET.parse(URDF).getroot()
    parts, all_verts = [], []
    for link in links:
        T_link = T_fk[link]
        for link_el in root.findall("link"):
            if link_el.attrib["name"] != link:
                continue
            for vis in link_el.findall("visual"):
                mat_name = vis.find("material").attrib["name"]
                color = COLORS[mat_name]
                o = vis.find("origin")
                xyz = np.array([float(v) for v in o.attrib["xyz"].split()]) if o is not None else np.zeros(3)
                rpy = np.array([float(v) for v in o.attrib["rpy"].split()]) if o is not None else np.zeros(3)
                rel = vis.find("geometry/mesh").attrib["filename"]
                rel = rel[len("assets/"):] if rel.startswith("assets/") else rel
                m = trimesh.load(os.path.join(ASSETS, rel), force="mesh")
                T = np.eye(4)
                T[:3, :3] = euler_matrix(rpy[0], rpy[1], rpy[2], "sxyz")[:3, :3]
                T[:3, 3] = xyz
                m.apply_transform(T)
                m.apply_transform(T_link)
                m.vertices = m.vertices * SCALE
                vc = np.zeros((len(m.vertices), 4))
                vc[:, :3] = color
                vc[:, 3] = 1.0
                m.visual = trimesh.visual.ColorVisuals(vertex_colors=vc)
                parts.append(m)
                all_verts.append(m.vertices)
    return parts, np.concatenate(all_verts)


def orbit_views(center, radius, n_views=4, elevs=(np.deg2rad(25.0),), d_scale=2.2):
    """pyrender 约定 T_wc。与 assemble_so101.py 完全一致。"""
    views = []
    for elev in elevs:
        for az in np.linspace(0, 2 * np.pi, n_views, endpoint=False):
            d = radius * d_scale
            cam_center = center + np.array([
                d * np.cos(elev) * np.cos(az), d * np.cos(elev) * np.sin(az), d * np.sin(elev)])
            fwd = center - cam_center
            fwd /= np.linalg.norm(fwd)
            up = np.array([0.0, 0.0, 1.0])
            right = np.cross(up, fwd)
            right /= np.linalg.norm(right)
            down = np.cross(fwd, right)
            T = np.eye(4)
            T[:3, :3] = np.vstack([right, down, fwd]).T
            T[:3, 3] = cam_center
            views.append(T)
    return views


def render(parts, T_wc):
    scene = pyrender.Scene(ambient_light=np.array([0.35, 0.35, 0.35]))
    for m in parts:
        scene.add(pyrender.Mesh.from_trimesh(m, smooth=False))
    cam = pyrender.PerspectiveCamera(yfov=FOV_Y, aspectRatio=1.0)
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.5)
    cam_node = scene.add(cam, pose=T_wc)
    scene.add(light, parent_node=cam_node)
    rnd = pyrender.OffscreenRenderer(512, 512)
    color, depth = rnd.render(scene)
    rnd.delete()
    alpha = ((depth > 0.08) & (depth < 50.0)).astype(np.uint8)[..., None] * 255
    rgb = color[:, :, :3]
    rgb = (rgb * (alpha / 255.0) + 255.0 * (1.0 - alpha / 255.0)).astype(np.uint8)
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", nargs="*", default=ALL_LINKS)
    ap.add_argument("--views", type=int, default=6)
    ap.add_argument("--elevs", default="15,35,55")
    ap.add_argument("--d-scale", type=float, default=2.4)
    ap.add_argument("--out", default="/amax/home/fengshuangyu/relighting/IRGS/outputs/so101_links/gt_full")
    opt = ap.parse_args()
    os.makedirs(opt.out, exist_ok=True)

    joints = parse_urdf(URDF)
    T_fk = fk(joints)
    parts, verts = build_arm_meshes(opt.links, T_fk)
    center = (verts.min(0) + verts.max(0)) / 2
    radius = float(np.linalg.norm(verts - center, axis=1).max())
    print(f"GT arm: {len(parts)} meshes, bbox center={center.round(3).tolist()}, radius={radius:.3f}")

    elevs = tuple(np.deg2rad(float(e)) for e in opt.elevs.split(","))
    views = orbit_views(center, radius, n_views=opt.views, elevs=elevs, d_scale=opt.d_scale)
    imgs = []
    for i, T_wc in enumerate(views):
        img = render(parts, T_wc)
        Image.fromarray(img).save(os.path.join(opt.out, f"gt_{i:02d}.png"))
        imgs.append(img)
        print(f"  view {i:2d}: d={np.linalg.norm(T_wc[:3, 3] - center):.2f}")

    W, H = imgs[0].shape[1], imgs[0].shape[0]
    cols = opt.views
    rows = len(views) // cols
    grid = Image.new("RGB", (W * cols + (cols + 1) * 5, H * rows + (rows + 1) * 5), (255, 255, 255))
    for idx, im in enumerate(imgs):
        r, c = idx // cols, idx % cols
        grid.paste(Image.fromarray(im), (5 + c * (W + 5), 5 + r * (H + 5)))
    grid.save(os.path.join(opt.out, "gt_grid.png"))
    print(f"[OK] {len(views)} views -> {opt.out}")


if __name__ == "__main__":
    main()
