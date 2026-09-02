#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把独立训练的 SO101 link 3DGS 资产按 URDF 关节树组装成整臂,
用 IRGS render_ir (envmap 光照 + PBR 材质) 渲染, 并输出 pyrender GT 对比.

坐标约定 (与 build_so101_link_3dgs.py 一致):
  - 每 link 的 3DGS 坐标系 = URDF link frame × scale 4.0 (build 时放大)
  - 整臂世界系也用放大系 (渲染相机/GT 同尺度, 避开 rasterizer 近裁剪 0.2m)
  - URDF rpy 为固定轴语义, euler 用 'sxyz'

组装变换 (刚体, 无尺度变化):
  p' = R_link @ p_gs + 4 * t_link        # R_link/t_link 为 FK 求得的米制位姿
  q' = quat_mul(q_link, q_gs)            # wxyz, R_link 对应四元数
  scaling/opacity/材质/SH 原样           # SH 仅用于 render_sh 辅助图, 主图不用

用法 (irgs env):
  python assemble_so101.py [--links gripper_link moving_jaw_so101_v1_link] \\
                           [--joints "shoulder_lift=0.3,elbow_flex=-0.8"] [--views 4]
输出:
  <out>/merged.ply            组装后完整点云 (可选 --save-ply)
  <out>/render_%02d.png       IRGS render_ir 渲染 (白底)
  <out>/gt_%02d.png           pyrender URDF 平色渲染 (同相机, 默认开)
  <out>/compare_%02d.png      左右并排
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # refl_utils 读相对路径 assets/bsdf_256_256.bin
import torch.nn.functional as F
from trimesh.transformations import euler_matrix, rotation_matrix

from arguments import ModelParams, PipelineParams
from gaussian_renderer import render_ir
from scene.light import EnvLight
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import getProjectionMatrix

URDF_PATHS = (
    ROOT / "SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
    ROOT.parent / "SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
)
URDF = str(next((path for path in URDF_PATHS if path.is_file()), URDF_PATHS[0]))
ASSETS = os.path.join(os.path.dirname(URDF), "assets")
OUT_ROOT = str(ROOT / "outputs/so101_links")
PLY_ROOT = OUT_ROOT  # outputs/so101_links/<link>/irgs_full/point_cloud/iteration_20000/point_cloud.ply
ALL_LINKS = ["base_link", "shoulder_link", "upper_arm_link", "lower_arm_link",
             "wrist_link", "gripper_link", "moving_jaw_so101_v1_link"]
SCALE = 4.0  # build 时的放大系数
FOV_Y = np.deg2rad(45.0)


# ---------------------------------------------------------------------------
# URDF FK
# ---------------------------------------------------------------------------

def parse_urdf(path):
    root = ET.parse(path).getroot()
    joints = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = np.array([float(v) for v in o.attrib["xyz"].split()]) if o is not None else np.zeros(3)
        rpy = np.array([float(v) for v in o.attrib["rpy"].split()]) if o is not None else np.zeros(3)
        a = j.find("axis")
        axis = np.array([float(v) for v in a.attrib["xyz"].split()]) if a is not None else np.array([0.0, 0.0, 1.0])
        joints[j.attrib["name"]] = {
            "parent": j.find("parent").attrib["link"],
            "child": j.find("child").attrib["link"],
            "xyz": xyz, "rpy": rpy, "axis": axis, "type": j.attrib["type"],
        }
    return joints


def fk(joints, angles=None):
    """0 姿态 FK (或给定角度 dict), 返回 {link: 4x4}。joint rpy 用固定轴语义 'sxyz'。"""
    angles = angles or {}
    T = {"base_link": np.eye(4)}
    while True:
        added = 0
        for name, j in joints.items():
            if j["child"] in T or j["parent"] not in T:
                continue
            T_origin = euler_matrix(j["rpy"][0], j["rpy"][1], j["rpy"][2], "sxyz")
            T_origin[:3, 3] = j["xyz"]
            a = angles.get(name, 0.0)
            T_axis = rotation_matrix(a, j["axis"]) if a != 0.0 else np.eye(4)
            T[j["child"]] = T[j["parent"]] @ T_origin @ T_axis
            added += 1
        if not added:
            break
    return T


def parse_joint_angles(s):
    out = {}
    for kv in s.split(","):
        if kv.strip():
            k, v = kv.split("=")
            out[k.strip()] = float(v)
    return out


# ---------------------------------------------------------------------------
# 四元数 (wxyz, 与 ply rot_0..3 及 rasterizer 约定一致)
# ---------------------------------------------------------------------------

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def mat_to_quat_wxyz(R):
    R = np.asarray(R, dtype=np.float64)
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = (0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2
        q = ((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2
        q = ((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2
        q = ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)
    q = np.array(q)
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------
# ply 读取与组装
# ---------------------------------------------------------------------------

def load_link_ply(link):
    import plyfile
    p = os.path.join(PLY_ROOT, link, "irgs_full", "point_cloud", "iteration_20000", "point_cloud.ply")
    assert os.path.exists(p), f"missing: {p}"
    v = plyfile.PlyData.read(p)["vertex"].data
    N = len(v)
    d = {
        "xyz": np.stack([v["x"], v["y"], v["z"]], -1),
        "base_color": np.stack([v["base_color_0"], v["base_color_1"], v["base_color_2"]], -1),
        "roughness": v["roughness"][:, None],
        "metallic": v["metallic"][:, None],
        "opacity": v["opacity"][:, None],
        "scale": np.stack([v["scale_0"], v["scale_1"]], -1),  # surfel 2D 尺度
        "rot": np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], -1),
        "f_dc": np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], -1)[:, None, :],   # (N,1,3)
        "f_rest": np.stack([v[f"f_rest_{i}"] for i in range(45)], -1).reshape(N, 15, 3),  # (N,15,3)
    }
    return d


def assemble(links, T_fk):
    """组装: p' = R_link @ p_gs + 4*t_link, q' = q_link * q_gs。返回 {attr: np array}"""
    parts = [load_link_ply(l) for l in links]
    for l, d in zip(links, parts):
        T = T_fk[l]
        R, t = T[:3, :3], T[:3, 3]
        d["xyz"] = d["xyz"] @ R.T + SCALE * t
        q_link = mat_to_quat_wxyz(R)
        d["rot"] = np.stack([quat_mul(q_link, q) for q in d["rot"]])
    return {k: np.concatenate([p[k] for p in parts], 0) for k in parts[0]}


# ---------------------------------------------------------------------------
# 相机 (照 IRGS scene/cameras.py Camera 的构造)
# ---------------------------------------------------------------------------

class SimpleCam:
    def __init__(self, T_wc, fov_y=FOV_Y, H=512, W=512, znear=0.01, zfar=100.0):
        self.FoVx = self.FoVy = fov_y
        self.image_height, self.image_width = H, W
        self.znear, self.zfar = znear, zfar
        w2c = np.linalg.inv(T_wc)
        self.world_view_transform = torch.tensor(w2c.T, dtype=torch.float32, device="cuda")
        self.projection_matrix = getProjectionMatrix(znear, zfar, fov_y, fov_y).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(
            self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        v, u = torch.meshgrid(torch.arange(H, device="cuda"),
                              torch.arange(W, device="cuda"), indexing="ij")
        fx = W / (2 * torch.tan(torch.tensor(fov_y * 0.5, device="cuda")))
        fy = H / (2 * torch.tan(torch.tensor(fov_y * 0.5, device="cuda")))
        rays = torch.stack([(u - W / 2 + 0.5) / fx, (v - H / 2 + 0.5) / fy,
                            torch.ones_like(u)], -1).reshape(-1, 3)
        rays = rays @ self.world_view_transform[:3, :3].T
        self.rays_d_hw = F.normalize(rays, dim=-1).reshape(H, W, 3)
        self.rays_d_hw_unnormalized = rays.reshape(H, W, 3)


def orbit_views(center, radius, n_views=4, elevs=(np.deg2rad(25.0),), d_scale=2.2):
    """pyrender 约定 T_wc (x右 y上 z后)。elevs 为多个仰角, 每圈 n_views 个方位。"""
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


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def render_irgs(merged, T_wc, pipe):
    # 不 load_ply: 4b 的 ply 无 ind_* 字段 (RefGaussianModel.load_ply 会崩), 手动赋值
    gaussians = GaussianModel(3)
    N = len(merged["xyz"])
    import torch.nn as nn
    for name, key in [("_xyz", "xyz"), ("_base_color", "base_color"), ("_roughness", "roughness"),
                      ("_metallic", "metallic"), ("_opacity", "opacity"),
                      ("_scaling", "scale"), ("_rotation", "rot"),
                      ("_features_dc", "f_dc"), ("_features_rest", "f_rest")]:
        setattr(gaussians, name, nn.Parameter(
            torch.tensor(merged[key], dtype=torch.float32, device="cuda").requires_grad_(True)))
    gaussians.max_radii2D = torch.zeros(N, device="cuda")
    gaussians.env_map = EnvLight(path=None, device="cuda",
                                 resolution=[64, 128], max_res=128,
                                 init_value=1.5, activation="exp").cuda()
    gaussians.build_bvh()  # 光追 BVH (IRGS 渲染管线必需, 官方 render.py 同样调用)
    cam = SimpleCam(T_wc)
    bg = torch.ones(3, dtype=torch.float32, device="cuda")  # 白底
    pkg = render_ir(cam, gaussians, pipe, bg)
    img = pkg["render"].permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
    return (img * 255).astype(np.uint8)


def render_gt(links, T_fk, T_wc):
    """pyrender URDF 平色渲染 (同 FK/同相机), 放大系"""
    import pyrender
    import trimesh
    from trimesh.transformations import euler_matrix as em
    parts = []
    for link in links:
        root = ET.parse(URDF).getroot()
        materials = {m.attrib["name"]: m for m in root.findall("material")}
        T_link = T_fk[link]
        for link_el in root.findall("link"):
            if link_el.attrib["name"] != link:
                continue
            for vis in link_el.findall("visual"):
                mat = materials[vis.find("material").attrib["name"]].find("color").attrib["rgba"].split()
                color = np.array([float(v) for v in mat[:3]])
                o = vis.find("origin")
                xyz = np.array([float(v) for v in o.attrib["xyz"].split()]) if o is not None else np.zeros(3)
                rpy = np.array([float(v) for v in o.attrib["rpy"].split()]) if o is not None else np.zeros(3)
                rel = vis.find("geometry/mesh").attrib["filename"]
                rel = rel[len("assets/"):] if rel.startswith("assets/") else rel
                m = trimesh.load(os.path.join(ASSETS, rel), force="mesh")
                T = np.eye(4)
                T[:3, :3] = em(rpy[0], rpy[1], rpy[2], "sxyz")[:3, :3]
                T[:3, 3] = xyz
                m.apply_transform(T)
                m.apply_transform(T_link)
                m.vertices = m.vertices * SCALE
                vc = np.zeros((len(m.vertices), 4))
                vc[:, :3] = color
                vc[:, 3] = 1.0
                m.visual = trimesh.visual.ColorVisuals(vertex_colors=vc)
                parts.append(m)
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
    ap.add_argument("--joints", default="", help="如 shoulder_lift=0.3,elbow_flex=-0.8")
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--elevs", default="25", help="逗号分隔仰角度数, 如 15,35,55")
    ap.add_argument("--d-scale", type=float, default=2.2, help="相机距离倍数 (越小越近)")
    ap.add_argument("--out", default=os.path.join(OUT_ROOT, "assembled"))
    ap.add_argument("--save-ply", action="store_true")
    ap.add_argument("--no-gt", action="store_true")
    opt = ap.parse_args()
    os.makedirs(opt.out, exist_ok=True)

    parser = argparse.ArgumentParser()
    pp = PipelineParams(parser)
    args = parser.parse_args([])
    pipe = pp.extract(args)

    joints = parse_urdf(URDF)
    angles = parse_joint_angles(opt.joints)
    T_fk = fk(joints, angles)
    for l in opt.links:
        assert l in T_fk, f"{l} 不在 FK 树中"
    print(f"FK: {len(T_fk)} links, 0 姿态 (angles={angles or 'all 0'})")

    merged = assemble(opt.links, T_fk)
    print(f"assembled: {len(merged['xyz'])} gaussians, "
          f"bbox center={merged['xyz'].mean(0).round(3).tolist()}, "
          f"extent={np.ptp(merged['xyz'], axis=0).round(3).tolist()}")

    center = (merged["xyz"].min(0) + merged["xyz"].max(0)) / 2
    radius = float(np.linalg.norm(merged["xyz"] - center, axis=1).max())
    elevs = tuple(np.deg2rad(float(e)) for e in opt.elevs.split(","))
    views = orbit_views(center, radius, n_views=opt.views, elevs=elevs, d_scale=opt.d_scale)

    from PIL import Image
    for i, T_wc in enumerate(views):
        print(f"  render view {i}: elev=25 azim={360*i/len(views):.0f} d={np.linalg.norm(T_wc[:3,3]-center):.2f}")
        irgs_img = render_irgs(merged, T_wc, pipe)
        Image.fromarray(irgs_img).save(os.path.join(opt.out, f"render_{i:02d}.png"))
        if not opt.no_gt:
            gt_img = render_gt(opt.links, T_fk, T_wc)
            Image.fromarray(gt_img).save(os.path.join(opt.out, f"gt_{i:02d}.png"))
            sep = np.full((512, 8, 3), 200, dtype=np.uint8)
            Image.fromarray(np.concatenate([gt_img, sep, irgs_img], 1)).save(
                os.path.join(opt.out, f"compare_{i:02d}.png"))
    if opt.save_ply:
        from plyfile import PlyData, PlyElement
        d = merged
        n = len(d["xyz"])
        props = {"x": d["xyz"][:, 0], "y": d["xyz"][:, 1], "z": d["xyz"][:, 2]}
        for i, c in enumerate("012"):
            props[f"f_dc_{i}"] = d["f_dc"][:, 0, i]
        for i in range(45):
            props[f"f_rest_{i}"] = d["f_rest"].reshape(n, -1)[:, i]
        props.update({"opacity": d["opacity"][:, 0], "metallic": d["metallic"][:, 0],
                      "roughness": d["roughness"][:, 0]})
        for i, c in enumerate("012"):
            props[f"base_color_{i}"] = d["base_color"][:, i]
        # IRGS uses 2D surfel Gaussians (scale_0/scale_1). Keep this dynamic so
        # the exporter also remains valid for any legacy 3D Gaussian assets.
        for i in range(d["scale"].shape[1]):
            props[f"scale_{i}"] = d["scale"][:, i]
        for i in range(4):
            props[f"rot_{i}"] = d["rot"][:, i]
        arr = np.zeros(n, dtype=[(k, "f4") for k in props])
        for k, v in props.items():
            arr[k] = v
        PlyData([PlyElement.describe(arr, "vertex")]).write(os.path.join(opt.out, "merged.ply"))
        print(f"  merged.ply saved ({n} pts)")
    print(f"[OK] -> {opt.out}")


if __name__ == "__main__":
    main()
