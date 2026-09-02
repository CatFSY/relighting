#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SO101 link → 3DGS 数据集构建 (渲染 → COLMAP sparse → Ref-DGS/GS-ROR 可训练场景)

流程 (每 link):
  1. 解析 URDF visual 标签: (mesh, origin_xyz, origin_rpy, material_color)
  2. 加载 STL, 应用 visual origin (URDF rpy = 固定轴 Rz(y)Ry(p)Rx(r)), 每部件着色
  3. pyrender 离屏渲染环绕视图 (5 仰角环 × 12 方位 = 60 张, 512x512, 白底合成 + alpha)
  4. 写 COLMAP sparse/0/{cameras,images,points3D}.bin + points3D.ply
     (相机位姿由渲染器直接给出, 精确已知, 无需跑 COLMAP 特征匹配)
  5. images_rgba/ 直接生成 (与 Ref-DGS prepare_rgba.py 输出格式一致)

输出结构 (每 link):
  {out}/<link_name>/
    images/frame_%05d.png         白底 RGB
    masks/frame_%05d.png          L 模式 0/255
    images_rgba/frame_%05d.png    RGBA (alpha=前景 mask)
    sparse/0/{cameras,images,points3D}.bin  COLMAP 二进制格式
    sparse/0/points3D.ply         SIBR 格式 (mesh 表面采样初始化)

运行环境: refdgs (trimesh + pyrender)
用法:
  python build_so101_link_3dgs.py [--links base_link ...] [--views 60] [--res 512]
"""
import argparse
import os
import struct
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
import pyrender
from PIL import Image
from trimesh.transformations import euler_matrix, quaternion_from_matrix

URDF = "/amax/home/fengshuangyu/relighting/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
ASSETS = os.path.join(os.path.dirname(URDF), "assets")
DEFAULT_OUT = "/amax/home/fengshuangyu/relighting/IRGS/dataset/so101_links"

COLORS = {
    "3d_printed": np.array([1.0, 0.82, 0.12]),   # 黄
    "sts3215":    np.array([0.1, 0.1, 0.1]),     # 舵机黑
}

FOV_Y = np.deg2rad(45.0)
ELEVATIONS = np.deg2rad([-60, -30, 0, 30, 60])
AZIMUTHS = np.linspace(0, 2 * np.pi, 12, endpoint=False)


def parse_urdf_visuals(path):
    """返回 {link_name: [(mesh_path, xyz, rpy, color_rgb)]}"""
    root = ET.parse(path).getroot()
    materials = {m.attrib["name"]: m for m in root.findall("material")}

    def color_of(material_name):
        mat = materials[material_name]
        rgba = mat.find("color").attrib["rgba"].split()
        return np.array([float(v) for v in rgba[:3]])

    out = {}
    for link in root.findall("link"):
        visuals = []
        for vis in link.findall("visual"):
            mat_name = vis.find("material").attrib["name"]
            origin = vis.find("origin")
            if origin is None:
                xyz, rpy = np.zeros(3), np.zeros(3)
            else:
                xyz = np.array([float(v) for v in origin.attrib["xyz"].split()])
                rpy = np.array([float(v) for v in origin.attrib["rpy"].split()])
            mesh_rel = vis.find("geometry/mesh").attrib["filename"]
            if mesh_rel.startswith("assets/"):
                mesh_rel = mesh_rel[len("assets/"):]
            mesh_path = os.path.join(ASSETS, mesh_rel)
            visuals.append((mesh_path, xyz, rpy, color_of(mat_name)))
        if visuals:
            out[link.attrib["name"]] = visuals
    return out


def load_link_meshes(visuals, scale=1.0):
    """应用 visual origin 变换, 返回 [(Trimesh with vertex_colors, color_rgb)]

    scale: 场景整体放大系数。diff-surfel-rasterization 硬编码近裁剪
    p_view.z <= 0.2 直接丢弃 (in_frustum)，相机距离必须 > 0.2m；
    真实相机场景 (米级) 天然满足，机械臂 link (~0.1m) 必须放大。
    渲染图像不变 (相机随几何一起缩放)，仅世界尺度变化。
    """
    parts = []
    for mesh_path, xyz, rpy, color in visuals:
        m = trimesh.load(mesh_path, force="mesh")
        T = np.eye(4)
        T[:3, 3] = xyz
        # URDF rpy = 固定轴 (world-fixed) roll-pitch-yaw: R = Rz(yaw)@Ry(pitch)@Rx(roll),
        # 必须用 trimesh 'sxyz'; 曾误用 'rxyz' (动轴) 导致多轴 rpy 零件错位 120-180°
        # (数据自洽所以单 link 训练/验证看不出, 但组装整臂会歪)
        T[:3, :3] = euler_matrix(rpy[0], rpy[1], rpy[2], "sxyz")[:3, :3]
        m.apply_transform(T)
        if scale != 1.0:
            m.vertices = m.vertices * scale
        # 顶点色 (8bit 或 float 都行, trimesh 内部转 float)
        vc = np.zeros((len(m.vertices), 4))
        vc[:, :3] = color
        vc[:, 3] = 1.0
        m.visual = trimesh.visual.ColorVisuals(vertex_colors=vc)
        parts.append((m, color))
    return parts


def orbit_camera(center, radius, elev, azim):
    """环绕相机: COLMAP 约定 (x右 y下 z前), 返回 (T_wc_pose, qvec, tvec)"""
    d = radius * 4.0  # fov=45° 下球体半径投影 ≈ 0.6 * 半高
    cam_center = center + np.array([
        d * np.cos(elev) * np.cos(azim),
        d * np.cos(elev) * np.sin(azim),
        d * np.sin(elev),
    ])
    fwd = cam_center - center  # 朝目标
    fwd = fwd / np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, fwd)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)  # 单位正交

    # pyrender 位姿: OpenGL 相机 -z 前向, 视线 = center - cam_center = -fwd,
    # 所以 T_wc 的 z 列 = fwd 对 pyrender 正确 (已验证渲染含物体)。
    R_wc = np.vstack([right, down, fwd])
    T_wc = np.eye(4)
    T_wc[:3, :3] = R_wc.T
    T_wc[:3, 3] = cam_center
    # COLMAP 位姿: 相机 +z 前向, 视线 = center - cam_center。
    # 三个硬性要求 (缺一即错):
    #   1. look 必须归一化 (否则行非单位长 -> tvec 缩放错 + 四元数非正交)
    #   2. (right, down, look) 必须构成右手系 det=+1 —— COLMAP 相机
    #      y 轴是图像下方 = -up, 而 down = cross(fwd,right) 是 pyrender
    #      (OpenGL y 朝上) 的 y 轴 = up; 用 cross(right, fwd) 才 det=+1。
    #      det=-1 的矩阵不是旋转矩阵, quaternion_from_matrix 提取出垃圾
    #      (此前 bug: qvec/tvec 全错, 训练位姿对不上渲染).
    #   3. qvec 约定 X_cam = R(q) @ X_world + t (w2c 四元数);
    #      quaternion_from_matrix 给 c2w 四元数 -> 取共轭 (w 不变, xyz 取反)。
    look = center - cam_center
    look = look / np.linalg.norm(look)
    down_img = np.cross(right, fwd)  # 图像下方 (COLMAP y 轴), 右手系 y
    R_colmap = np.vstack([right, down_img, look])  # 行 = 相机轴, det=+1
    tvec = -R_colmap @ cam_center
    T_colmap = np.eye(4)
    T_colmap[:3, :3] = R_colmap.T
    T_colmap[:3, 3] = cam_center
    qvec = quaternion_from_matrix(T_colmap) * np.array([1.0, -1.0, -1.0, -1.0])
    return T_wc, qvec, tvec


def render_views(parts, center, radius):
    """返回 [(rgb_uint8, alpha_uint8)] 白底合成"""
    scene = pyrender.Scene(ambient_light=np.array([0.35, 0.35, 0.35]))
    for m, _ in parts:
        pm = pyrender.Mesh.from_trimesh(m, smooth=False)
        scene.add(pm)
    cam = pyrender.PerspectiveCamera(yfov=FOV_Y, aspectRatio=1.0)
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.5)

    rnd = pyrender.OffscreenRenderer(512, 512)
    views = []
    for elev in ELEVATIONS:
        for azim in AZIMUTHS:
            T_wc, _, _ = orbit_camera(center, radius, elev, azim)
            cam_node = scene.add(cam, pose=T_wc)
            scene.add(light, parent_node=cam_node)  # 灯随相机 (头部灯, 明暗一致)
            color, depth = rnd.render(scene)
            scene.remove_node(cam_node)  # 连同子节点(灯)一起移除
            # 背景深度被清成 0 (非 far 平面), znear=0.05 → 真实几何 depth>0.08
            alpha = ((depth > 0.08) & (depth < 50.0)).astype(np.uint8) * 255
            rgb = color[:, :, :3]
            rgb = (rgb * (alpha[..., None] / 255.0) + 255.0 * (1.0 - alpha[..., None] / 255.0))
            views.append((rgb.astype(np.uint8), alpha))
    rnd.delete()
    return views


def write_colmap_bin(scene_dir, qvecs, tvecs, points_xyz, points_rgb):
    """写 COLMAP 二进制 sparse 模型 (Ref-DGS 用自带 reader 读 .bin)"""
    sparse_dir = os.path.join(scene_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)

    # cameras.bin: 按 Ref-DGS colmap_loader "iiQQ" 格式 (width/height 为 uint64)
    # PINHOLE(model_id=1), 512x512, fx fy cx cy
    with open(os.path.join(sparse_dir, "cameras.bin"), "wb") as f:
        f.write(struct.pack("<Q", 1))            # num_cameras
        f.write(struct.pack("<i", 1))            # camera_id
        f.write(struct.pack("<i", 1))            # model_id PINHOLE
        f.write(struct.pack("<QQ", 512, 512))
        f.write(struct.pack("<4d", 512 / (2 * np.tan(FOV_Y / 2)),
                            512 / (2 * np.tan(FOV_Y / 2)), 256.0, 256.0))

    # images.bin: "idddddddi" + null 结尾文件名 + 无 2D 特征点
    with open(os.path.join(sparse_dir, "images.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(qvecs)))
        for i, (q, t) in enumerate(zip(qvecs, tvecs)):
            f.write(struct.pack("<i", i + 1))
            f.write(struct.pack("<4d", *q))
            f.write(struct.pack("<3d", *t))
            f.write(struct.pack("<i", 1))        # camera_id
            name = f"frame_{i:05d}.png".encode() + b"\x00"
            f.write(struct.pack(f"<{len(name)}s", name))
            f.write(struct.pack("<Q", 0))        # num_points2D

    # points3D.bin: "QdddBBBd" + 空 track
    with open(os.path.join(sparse_dir, "points3D.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(points_xyz)))
        for i, (p, c) in enumerate(zip(points_xyz, points_rgb)):
            f.write(struct.pack("<Q", i + 1))
            f.write(struct.pack("<3d", *p))
            f.write(struct.pack("<3B", *c))
            f.write(struct.pack("<d", 0.0))      # error
            f.write(struct.pack("<Q", 0))        # track 长度

    # points3D.ply (Ref-DGS storePly/fetchPly 格式: red/green/blue u1)
    n = len(points_xyz)
    with open(os.path.join(sparse_dir, "points3D.ply"), "wb") as f:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property float nx\nproperty float ny\nproperty float nz\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode())
        for p, c in zip(points_xyz, points_rgb):
            f.write(struct.pack("<3f", *p))
            f.write(struct.pack("<3f", 0.0, 0.0, 0.0))
            f.write(struct.pack("<3B", *c))


def sample_init_points(parts, n_total=20000, seed=0):
    """按面积加权网格表面采样, 每个部件带自己的平色"""
    rng = np.random.default_rng(seed)
    areas = [m.area for m, _ in parts]
    total = sum(areas)
    pts, cols = [], []
    for (m, color), a in zip(parts, areas):
        n = int(round(n_total * a / total))
        if n < 100:
            n = 100
        face_idx = rng.choice(len(m.faces), size=n, p=m.area_faces / m.area)
        # barycentric 采样
        tri = m.vertices[m.faces[face_idx]]
        r1 = rng.random(n)[:, None]
        r2 = rng.random(n)[:, None]
        bary = np.hstack([1 - np.sqrt(r1), np.sqrt(r1) * (1 - r2), np.sqrt(r1) * r2])
        pts.append((tri * bary[..., None]).sum(1))
        cols.append(np.tile(color, (n, 1)))
    xyz = np.concatenate(pts)
    rgb = (np.concatenate(cols) * 255).astype(np.uint8)
    return xyz, rgb


def build_link(link_name, visuals, out_dir, seed=0, scale=1.0):
    print(f"=== {link_name}: {len(visuals)} visuals")
    parts = load_link_meshes(visuals, scale=scale)
    all_verts = np.concatenate([m.vertices for m, _ in parts])
    center = (all_verts.min(0) + all_verts.max(0)) / 2
    radius = float(np.linalg.norm(all_verts - center, axis=1).max())

    views = render_views(parts, center, radius)
    xyz, rgb = sample_init_points(parts, seed=seed)
    qvecs, tvecs = [], []
    for elev in ELEVATIONS:
        for azim in AZIMUTHS:
            _, q, t = orbit_camera(center, radius, elev, azim)
            qvecs.append(q)
            tvecs.append(t)

    scene_dir = os.path.join(out_dir, link_name)
    for sub in ("images", "masks", "images_rgba"):
        os.makedirs(os.path.join(scene_dir, sub), exist_ok=True)
    for i, (rgb_img, alpha) in enumerate(views):
        base = f"frame_{i:05d}"
        Image.fromarray(rgb_img, "RGB").save(os.path.join(scene_dir, "images", base + ".png"))
        Image.fromarray(alpha, "L").save(os.path.join(scene_dir, "masks", base + ".png"))
        rgba = np.concatenate([rgb_img, alpha[..., None]], axis=-1)
        Image.fromarray(rgba, "RGBA").save(os.path.join(scene_dir, "images_rgba", base + ".png"))

    write_colmap_bin(scene_dir, qvecs, tvecs, xyz, rgb)
    print(f"[OK] {link_name}: {len(views)} views, {len(xyz)} init pts -> {scene_dir}")
    return scene_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", nargs="*", default=None, help="指定 link, 默认全部")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=4.0,
                    help="场景放大系数 (默认 4.0: 相机距离 ~0.8m, 满足 rasterizer 近裁剪 0.2m)")
    opt = ap.parse_args()

    all_links = parse_urdf_visuals(URDF)
    print(f"URDF links with visuals: {list(all_links)}")
    links = opt.links if opt.links else list(all_links)
    for link in links:
        if link not in all_links:
            print(f"[WARN] {link} 无 visual (dummy link?), 跳过")
            continue
        build_link(link, all_links[link], opt.out, seed=opt.seed, scale=opt.scale)


if __name__ == "__main__":
    main()
