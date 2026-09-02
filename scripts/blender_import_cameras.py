# -*- coding: utf-8 -*-
"""Blender 内运行: 从 colmap_blender_poses.py 导出的 JSON 重建相机 (c2w)

用法:
  1. 外部先导出:
     python colmap_blender_poses.py colmap2json <sparse_dir> /path/to/cameras.json
  2. Blender 内 (两种方式):
     a. Scripting 面板粘贴运行, 末尾改 JSON 路径;
     b. 命令行 (无头):
        blender -b scene.blend -P blender_import_cameras.py -- /path/to/cameras.json
"""
import json
import sys

import bpy
import numpy as np


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def main(json_path=None):
    if json_path is None:
        argv = sys.argv
        json_path = argv[argv.index("--") + 1] if "--" in argv else argv[-1]
    data = json.load(open(json_path))

    # 删除已有相机
    for ob in list(bpy.data.objects):
        if ob.type == "CAMERA":
            bpy.data.objects.remove(ob, do_unlink=True)

    sw = data.get("sensor_width_mm", 36.0)
    frames = data["frames"]
    for i, fr in enumerate(frames):
        cam = bpy.data.cameras.new(f"cam_{i:04d}")
        ob = bpy.data.objects.new(cam.name, cam)
        bpy.context.collection.objects.link(ob)
        T = np.eye(4)
        T[:3, :3] = quat_to_mat(fr["rotation_quaternion_wxyz"])
        T[:3, 3] = fr["location"]
        ob.matrix_world = T.tolist()
        w, h = fr["width"], fr["height"]
        cam.sensor_width = sw
        cam.sensor_fit = "AUTO"
        cam.lens = fr["focal_px"] / w * sw
        cam.shift_x = (w / 2.0 - fr["cx"]) / w   # 主点偏移 (sensor 宽度的分数)
        cam.shift_y = (h / 2.0 - fr["cy"]) / h
        if i == 0:
            scn = bpy.context.scene
            scn.render.resolution_x = w
            scn.render.resolution_y = h
            scn.render.image_settings.file_format = "PNG"
    print(f"[OK] {len(frames)} 相机已创建: {json_path}")


if __name__ == "__main__":
    main()
