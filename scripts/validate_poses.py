#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""决定性验证: COLMAP 读回位姿 → pyrender 重渲染 mesh → 与 GT 图像对比

如果位姿写入/读取正确, 重渲染应与数据集 GT 完全一致 (同一几何+同一相机):
  - mask IoU 应 ≈ 1.0
  - 前景 RGB MAE 应很小 (光照/插值差异)

用法 (refdgs env):
  python validate_poses.py <link_name> [cam_indices...]
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

# 直接按文件加载 colmap_loader (绕开 scene/__init__.py 的 torch/pyexr 依赖)
spec = importlib.util.spec_from_file_location(
    "colmap_loader", "/amax/home/fengshuangyu/relighting/IRGS/scene/colmap_loader.py")
colmap_loader = importlib.util.module_from_spec(spec)
sys.modules["colmap_loader"] = colmap_loader
spec.loader.exec_module(colmap_loader)
read_extrinsics_binary = colmap_loader.read_extrinsics_binary
read_intrinsics_binary = colmap_loader.read_intrinsics_binary

ROOT = "/amax/home/fengshuangyu/relighting/IRGS"
URDF = "/amax/home/fengshuangyu/relighting/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
ASSETS = os.path.join(os.path.dirname(URDF), "assets")
SCENE_ROOT = os.path.join(ROOT, "dataset", "so101_links")

import xml.etree.ElementTree as ET
import trimesh
import pyrender
from PIL import Image
from trimesh.transformations import euler_matrix


def load_link_meshes(link_name, scale=4.0):
    root = ET.parse(URDF).getroot()
    materials = {m.attrib["name"]: m for m in root.findall("material")}
    parts = []
    for link in root.findall("link"):
        if link.attrib["name"] != link_name:
            continue
        for vis in link.findall("visual"):
            mat_name = vis.find("material").attrib["name"]
            rgba = materials[mat_name].find("color").attrib["rgba"].split()
            color = np.array([float(v) for v in rgba[:3]])
            origin = vis.find("origin")
            if origin is None:
                xyz, rpy = np.zeros(3), np.zeros(3)
            else:
                xyz = np.array([float(v) for v in origin.attrib["xyz"].split()])
                rpy = np.array([float(v) for v in origin.attrib["rpy"].split()])
            mesh_rel = vis.find("geometry/mesh").attrib["filename"]
            if mesh_rel.startswith("assets/"):
                mesh_rel = mesh_rel[len("assets/"):]
            m = trimesh.load(os.path.join(ASSETS, mesh_rel), force="mesh")
            T = np.eye(4)
            T[:3, 3] = xyz
            # URDF rpy = 固定轴 roll-pitch-yaw -> 'sxyz' (与 build 脚本一致)
            T[:3, :3] = euler_matrix(rpy[0], rpy[1], rpy[2], "sxyz")[:3, :3]
            m.apply_transform(T)
            if scale != 1.0:
                m.vertices = m.vertices * scale
            vc = np.zeros((len(m.vertices), 4))
            vc[:, :3] = color
            vc[:, 3] = 1.0
            m.visual = trimesh.visual.ColorVisuals(vertex_colors=vc)
            parts.append(m)
    return parts


def render_with_pose(parts, T_pose, fov_y=np.deg2rad(45.0), size=512):
    scene = pyrender.Scene(ambient_light=np.array([0.35, 0.35, 0.35]))
    for m in parts:
        scene.add(pyrender.Mesh.from_trimesh(m, smooth=False))
    cam = pyrender.PerspectiveCamera(yfov=fov_y, aspectRatio=1.0)
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.5)
    cam_node = scene.add(cam, pose=T_pose)
    scene.add(light, parent_node=cam_node)
    rnd = pyrender.OffscreenRenderer(size, size)
    color, depth = rnd.render(scene)
    rnd.delete()
    alpha = ((depth > 0.08) & (depth < 50.0)).astype(np.uint8) * 255
    return color[:, :, :3], alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("link")
    ap.add_argument("cams", nargs="*", type=int, default=None)
    opt = ap.parse_args()
    link = opt.link
    scene_dir = os.path.join(SCENE_ROOT, link)

    parts = load_link_meshes(link, scale=4.0)
    intrinsics = read_intrinsics_binary(os.path.join(scene_dir, "sparse", "0", "cameras.bin"))
    extrinsics = read_extrinsics_binary(os.path.join(scene_dir, "sparse", "0", "images.bin"))
    print(f"intrinsics: {len(intrinsics)} cam, extrinsics: {len(extrinsics)} imgs")

    gt_mask0 = np.array(Image.open(os.path.join(scene_dir, "masks", "frame_00000.png")))
    print(f"GT mask area (cam0): {int((gt_mask0 > 0).sum())} px / {512*512} = "
          f"{(gt_mask0 > 0).mean()*100:.1f}%")

    cams = opt.cams if opt.cams else [0, 19, 30, 45, 59]
    for cidx in cams:
        img = extrinsics[cidx + 1]  # id 从 1 开始
        R_w2c = img.qvec2rotmat()  # w2c 旋转, 行 = 相机轴 in world
        tvec = img.tvec
        # COLMAP: X_cam = R_w2c @ X_world + tvec (相机系 x右 y下 z前)
        # pyrender (OpenGL): 相机系 x右 y上 z后 -> X_cam_gl = diag(1,-1,-1) X_cam
        R_pose = R_w2c.T @ np.diag([1.0, -1.0, -1.0])
        t_pose = -R_w2c.T @ tvec
        T_pose = np.eye(4)
        T_pose[:3, :3] = R_pose
        T_pose[:3, 3] = t_pose

        rgb, alpha = render_with_pose(parts, T_pose)
        gt_rgb = np.array(Image.open(
            os.path.join(scene_dir, "images", f"frame_{cidx:05d}.png"))).astype(np.float32)
        gt_mask = np.array(Image.open(
            os.path.join(scene_dir, "masks", f"frame_{cidx:05d}.png")))
        pred_mask = (alpha > 0)

        inter = (pred_mask & (gt_mask > 0)).sum()
        union = (pred_mask | (gt_mask > 0)).sum()
        iou = inter / union if union else 0.0
        # 前景 RGB MAE (mask 并集内)
        fg = pred_mask | (gt_mask > 0)
        mae = np.abs(rgb.astype(np.float32) - gt_rgb)[fg].mean()
        # 覆盖率
        cov = inter / max(1, (gt_mask > 0).sum())
        # 质心偏移 (像素)
        def centroid(m):
            ys, xs = np.nonzero(m)
            return (xs.mean(), ys.mean()) if len(xs) else (None, None)
        c_pred, c_gt = centroid(pred_mask), centroid(gt_mask > 0)
        offset = np.linalg.norm(np.array(c_pred) - np.array(c_gt)) if c_pred[0] else -1
        print(f"cam {cidx:2d}: IoU={iou:.4f} coverage={cov:.4f} "
              f"fgRGB_MAE={mae:.2f} centroid_off={offset:.1f}px "
              f"pred_area={(pred_mask).sum()} gt_area={(gt_mask>0).sum()}")


if __name__ == "__main__":
    main()
