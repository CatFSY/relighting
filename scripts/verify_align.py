#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""决定性验证: 3DGS 点云 vs GT mesh 顶点是否落在同一坐标系 (双向最近邻距离)

mesh 路径:  STL → origin(sxyz) → link frame(米制) → ×SCALE → FK → 世界放大系
3dgs 路径:  训练好的高斯坐标 → FK → 世界放大系 (3DGS 训练无任何坐标归一化)
两者应重合: 双向 NN 距离应为 mm 级

用法 (irgs env):
  python verify_align.py [--links gripper_link moving_jaw_so101_v1_link]
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree
from trimesh.transformations import euler_matrix, rotation_matrix

sys.path.insert(0, "/amax/home/fengshuangyu/relighting/IRGS")
from scripts.assemble_so101 import parse_urdf, fk, load_link_ply, assemble, ALL_LINKS

URDF = "/amax/home/fengshuangyu/relighting/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
ASSETS = os.path.join(os.path.dirname(URDF), "assets")
OUT_ROOT = "/amax/home/fengshuangyu/relighting/IRGS/outputs/so101_links"
SCALE = 4.0


def gt_mesh_vertices(links, T_fk):
    """GT mesh 顶点: origin(sxyz) → link frame → ×SCALE → FK (与 gt_arm_render 一致)"""
    root = ET.parse(URDF).getroot()
    import trimesh
    verts = []
    for link in links:
        T_link = T_fk[link]
        for link_el in root.findall("link"):
            if link_el.attrib["name"] != link:
                continue
            for vis in link_el.findall("visual"):
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
                verts.append(m.vertices * SCALE)
    return np.concatenate(verts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", nargs="*", default=None)
    opt = ap.parse_args()
    links = opt.links or ALL_LINKS

    joints = parse_urdf(URDF)
    T_fk = fk(joints)

    merged = assemble(links, T_fk)
    gs = merged["xyz"]
    mesh_v = gt_mesh_vertices(links, T_fk)
    print(f"3DGS: {len(gs)} pts, bbox extent={np.ptp(gs, 0).round(3).tolist()}")
    print(f"mesh: {len(mesh_v)} verts, bbox extent={np.ptp(mesh_v, 0).round(3).tolist()}")

    # 双向最近邻 (单位: 米, 世界放大系; ×4 后 link 尺寸 ~0.1-0.5m)
    d1 = cKDTree(gs).query(mesh_v, k=1)[0]          # mesh 顶点 → 最近高斯
    d2 = cKDTree(mesh_v).query(gs, k=1)[0]          # 高斯 → 最近 mesh 顶点

    def stats(d, label):
        print(f"{label}: mean={d.mean()*100:.2f}cm  med={np.median(d)*100:.2f}cm  "
              f"p95={np.percentile(d,95)*100:.2f}cm  max={d.max()*100:.2f}cm  "
              f"<1cm: {(d < 0.01).mean()*100:.1f}%")

    stats(d1, "mesh→3dgs ")
    stats(d2, "3dgs→mesh ")
    ok = np.percentile(d1, 95) < 0.02 and np.percentile(d2, 95) < 0.02
    print(f"[{'PASS' if ok else 'FAIL'}] p95 双向距离 < 2cm (link 尺寸 ~10-40cm)")


if __name__ == "__main__":
    main()
