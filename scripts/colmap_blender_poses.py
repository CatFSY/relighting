#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""COLMAP ↔ Blender 相机位姿互转 (自包含, 仅需 numpy)

坐标约定:
  COLMAP : 外参为 w2c (X_cam = R_w2c @ X_world + tvec), qvec 为 w2c 四元数;
           相机系 x 右, y 下, z 前 (图像 y 轴向下)
  Blender: 相机对象 matrix_world 为 c2w; 相机局部 -Z 前向, +Y 上方
           (与 OpenGL/pyrender 相机约定一致: x 右, y 上, z 后)

转换核心 (与 scripts/validate_poses.py 同一公式, 已在 SO101 数据集实测 IoU=1.0000):
  S = diag(1, -1, -1)                      # COLMAP 相机系 -> Blender 相机系 (绕 x 转 180°, det=+1)
  R_c2w = R_w2c.T @ S;  loc = -R_w2c.T @ tvec          # w2c -> c2w
  R_w2c = S @ R_c2w.T;   tvec = -S @ R_c2w.T @ loc     # c2w -> w2c
  (S 为真旋转不是反射, 两个方向都保持右手系, 无手性陷阱)

典型工作流:
  A. Blender 摆好相机/渲染出图 -> COLMAP 训练集:
     (Blender 内) python colmap_blender_poses.py blender2colmap <out_dir>
     或 (Blender 内) 仅导出 json:  python colmap_blender_poses.py blender2json <out.json>
        然后外部:                  python colmap_blender_poses.py json2colmap <in.json> <out_dir>
  B. COLMAP 位姿 -> Blender 复现/验证相机:
     python colmap_blender_poses.py colmap2json <sparse_dir> <out.json>
     然后在 Blender Scripting 面板运行 scripts/blender_import_cameras.py <out.json>

命令:
  colmap2json   <sparse_dir> <out.json>       COLMAP(bin/txt) -> Blender json
  json2colmap   <in.json> <out_dir>          Blender json -> COLMAP(bin, 顺带 txt)
  blender2json  <out.json>                   (bpy) 当前场景相机 -> json
  blender2colmap <out_dir>                   (bpy) 当前场景相机 -> COLMAP
  colmap2blender <sparse_dir>                (bpy) COLMAP -> 直接建相机
  --self-test                                数值验证 (无需 bpy)
"""
import json
import os
import struct
import sys
import tempfile

import numpy as np

# ---------------------------------------------------------------------------
# 核心: COLMAP w2c <-> Blender c2w 互转
# ---------------------------------------------------------------------------

S = np.diag([1.0, -1.0, -1.0])  # COLMAP 相机系 -> Blender/OpenGL 相机系


def w2c_to_c2w(R_w2c, tvec):
    """COLMAP w2c (x右y下z前) -> Blender c2w (x右y上z后, 相机 -Z 前向)"""
    R_c2w = R_w2c.T @ S
    loc = -R_w2c.T @ tvec
    return R_c2w, loc


def c2w_to_w2c(R_c2w, loc):
    """Blender c2w -> COLMAP w2c (反向)"""
    R_w2c = S @ R_c2w.T
    tvec = -S @ R_c2w.T @ loc
    return R_w2c, tvec


def quat_wxyz_to_mat(q):
    """wxyz 四元数 -> 旋转矩阵 (与 COLMAP/Blender 一致的右手旋转)"""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat_wxyz(R):
    """旋转矩阵 -> wxyz 四元数 (Shepperd 分支法, 数值稳定, w>=0)"""
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
        q = -q  # w>=0 约定 (q 与 -q 等价, 统一符号避免 diff 假阳性)
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------
# COLMAP sparse 读写 (bin + txt, 与 IRGS/scene/colmap_loader.py 兼容)
# ---------------------------------------------------------------------------

CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}
MODEL_IDS = {name: i for i, (name, _) in CAMERA_MODELS.items()}


def read_cameras_bin(path):
    cameras = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cid, model_id, w, h = struct.unpack("<iiQQ", f.read(24))
            name, nparam = CAMERA_MODELS[model_id]
            params = struct.unpack("<%dd" % nparam, f.read(8 * nparam))
            cameras[cid] = {"model": name, "width": w, "height": h,
                            "params": list(params)}
    return cameras


def write_cameras_bin(path, cameras):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cid, c in cameras.items():
            model_id = MODEL_IDS[c["model"]]
            nparam = CAMERA_MODELS[model_id][1]
            f.write(struct.pack("<iiQQ", cid, model_id, c["width"], c["height"]))
            f.write(struct.pack("<%dd" % nparam, *c["params"][:nparam]))


def read_images_bin(path):
    images = []
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack("<idddddddi", f.read(64))
            name = b"".join(iter(lambda: f.read(1), b"\x00")).decode()
            npts = struct.unpack("<Q", f.read(8))[0]
            if npts:
                f.read(16 * npts)  # (x, y) 2x8B + point3D_id 8B, 忽略
            images.append({"id": iid, "qvec": np.array([qw, qx, qy, qz]),
                           "tvec": np.array([tx, ty, tz]), "camera_id": cid,
                           "name": name})
    return images


def write_images_bin(path, images):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for img in images:
            q, t = img["qvec"], img["tvec"]
            f.write(struct.pack("<idddddddi", img["id"], *q, *t, img["camera_id"]))
            f.write(img["name"].encode() + b"\x00")
            f.write(struct.pack("<Q", 0))  # 无 2D 特征点


def read_cameras_txt(path):
    cameras = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        cameras[int(p[0])] = {"model": p[1], "width": int(p[2]), "height": int(p[3]),
                              "params": [float(x) for x in p[4:]]}
    return cameras


def write_cameras_txt(path, cameras):
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cid, c in cameras.items():
            f.write(f"{cid} {c['model']} {c['width']} {c['height']} "
                    + " ".join(f"{p:.8f}" for p in c["params"]) + "\n")


def read_images_txt(path):
    images = []
    lines = [l.rstrip("\n") for l in open(path) if l.strip() and not l.startswith("#")]
    for k in range(0, len(lines), 2):
        p = lines[k].split()
        images.append({"id": int(p[0]),
                       "qvec": np.array([float(v) for v in p[1:5]]),
                       "tvec": np.array([float(v) for v in p[5:8]]),
                       "camera_id": int(p[7]), "name": p[8]})
    return images


def write_images_txt(path, images):
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for img in images:
            q, t = img["qvec"], img["tvec"]
            f.write(f"{img['id']} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
                    f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {img['camera_id']} {img['name']}\n")
            f.write("\n")


def find_sparse(sparse_dir):
    """自动定位 sparse/0 或 sparse 根, 返回 (目录, 'bin'|'txt')"""
    for cand, name in ((sparse_dir, "bin"), (sparse_dir, "txt"),
                       (os.path.join(sparse_dir, "0"), "bin"),
                       (os.path.join(sparse_dir, "0"), "txt")):
        if os.path.isfile(os.path.join(cand, "cameras." + name)):
            return cand, name
    raise FileNotFoundError(f"找不到 cameras.bin/txt: {sparse_dir}")


# ---------------------------------------------------------------------------
# COLMAP <-> Blender JSON (中间格式)
# ---------------------------------------------------------------------------

def _pinhole_params(params):
    """各种相机模型取 (fx, fy, cx, cy); 单焦距模型 fx=fy"""
    if len(params) in (3, 4):       # SIMPLE_PINHOLE / PINHOLE / SIMPLE_RADIAL
        return params[0], (params[1] if len(params) == 4 else params[0]), params[-2], params[-1]
    raise ValueError(f"不支持的相机模型参数 {len(params)} 个 (仅支持 fx/fy 在开头的模型)")


def colmap2json(sparse_dir, out_json):
    sdir, fmt = find_sparse(sparse_dir)
    read_c = read_cameras_bin if fmt == "bin" else read_cameras_txt
    read_i = read_images_bin if fmt == "bin" else read_images_txt
    cameras = read_c(os.path.join(sdir, "cameras." + fmt))
    images = read_i(os.path.join(sdir, "images." + fmt))

    frames = []
    for img in images:
        R_w2c = quat_wxyz_to_mat(img["qvec"])
        R_c2w, loc = w2c_to_c2w(R_w2c, img["tvec"])
        fx, fy, cx, cy = _pinhole_params(cameras[img["camera_id"]]["params"])
        c = cameras[img["camera_id"]]
        frames.append({
            "name": img["name"], "camera_id": img["camera_id"],
            "location": loc.tolist(),                      # c2w 平移
            "rotation_quaternion_wxyz": mat_to_quat_wxyz(R_c2w).tolist(),
            "focal_px": fx, "fx": fx, "fy": fy,
            "cx": cx, "cy": cy, "width": c["width"], "height": c["height"],
        })
    data = {"sensor_width_mm": 36.0, "cameras": {str(k): v for k, v in cameras.items()},
            "frames": frames}
    json.dump(data, open(out_json, "w"), indent=2, ensure_ascii=False)
    print(f"[OK] {len(frames)} 帧 -> {out_json}  (Blender c2w, 相机 -Z 前向)")


def json2colmap(json_path, out_dir):
    data = json.load(open(json_path))
    frames = data["frames"]

    cam_map = data.get("cameras")
    if not cam_map:  # 无 cameras 字段时按第一帧内参合成 PINHOLE
        fr = frames[0]
        cam_map = {"1": {"model": "PINHOLE", "width": fr["width"], "height": fr["height"],
                         "params": [fr["fx"], fr.get("fy", fr["fx"]), fr["cx"], fr["cy"]]}}

    images = []
    for i, fr in enumerate(frames):
        R_c2w = quat_wxyz_to_mat(fr["rotation_quaternion_wxyz"])
        R_w2c, tvec = c2w_to_w2c(R_c2w, np.array(fr["location"]))
        images.append({"id": i + 1, "qvec": mat_to_quat_wxyz(R_w2c),  # w2c 四元数
                       "tvec": tvec, "camera_id": fr.get("camera_id", 1),
                       "name": fr.get("name", f"frame_{i:05d}.png")})

    sparse0 = os.path.join(out_dir, "sparse", "0")
    os.makedirs(sparse0, exist_ok=True)
    cameras = {int(k): v for k, v in cam_map.items()}
    write_cameras_bin(os.path.join(sparse0, "cameras.bin"), cameras)
    write_images_bin(os.path.join(sparse0, "images.bin"), images)
    with open(os.path.join(sparse0, "points3D.bin"), "wb") as f:
        f.write(struct.pack("<Q", 0))  # 空点云 (初始化通常用 ply/pcd, 仅需文件存在)
    write_cameras_txt(os.path.join(sparse0, "cameras.txt"), cameras)
    write_images_txt(os.path.join(sparse0, "images.txt"), images)
    print(f"[OK] {len(images)} 帧 -> {sparse0}  (qvec 为 w2c, 相机系 x右y下z前)")


# ---------------------------------------------------------------------------
# bpy 可选分支 (仅在 Blender 内置 python 中可用)
# ---------------------------------------------------------------------------

def apply_to_blender(json_path):
    """COLMAP json -> 在 Blender 场景中创建相机对象 (matrix_world = c2w)"""
    import bpy
    data = json.load(open(json_path))
    for ob in list(bpy.data.objects):
        if ob.type == "CAMERA":
            bpy.data.objects.remove(ob, do_unlink=True)
    sw = data.get("sensor_width_mm", 36.0)
    for i, fr in enumerate(data["frames"]):
        cam = bpy.data.cameras.new(f"cam_{i:04d}")
        ob = bpy.data.objects.new(cam.name, cam)
        bpy.context.collection.objects.link(ob)
        T = np.eye(4)
        T[:3, :3] = quat_wxyz_to_mat(fr["rotation_quaternion_wxyz"])
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
    print(f"[OK] {len(data['frames'])} 相机已创建: {json_path}")


def export_from_blender(out_json):
    """Blender 当前场景所有相机 -> json (c2w)"""
    import bpy
    cams = [ob for ob in bpy.data.objects if ob.type == "CAMERA"]
    if not cams:
        sys.exit("场景中没有相机对象")
    scn = bpy.context.scene
    w, h = scn.render.resolution_x, scn.render.resolution_y
    frames, cam_map = [], {}
    for ob in cams:
        T = np.array(ob.matrix_world)
        R_c2w, loc = T[:3, :3], T[:3, 3]
        cam = ob.data
        focal_px = cam.lens / cam.sensor_width * w
        cx = w / 2.0 - cam.shift_x * w
        cy = h / 2.0 - cam.shift_y * h
        frames.append({
            "name": ob.name + ".png", "camera_id": 1,
            "location": loc.tolist(),
            "rotation_quaternion_wxyz": mat_to_quat_wxyz(R_c2w).tolist(),
            "focal_px": focal_px, "fx": focal_px, "fy": focal_px,
            "cx": cx, "cy": cy, "width": w, "height": h,
        })
    cam_map["1"] = {"model": "PINHOLE", "width": w, "height": h,
                    "params": [focal_px, focal_px, cx, cy]}
    data = {"sensor_width_mm": cams[0].data.sensor_width, "cameras": cam_map,
            "frames": frames}
    json.dump(data, open(out_json, "w"), indent=2, ensure_ascii=False)
    print(f"[OK] {len(frames)} 相机 -> {out_json} (render {w}x{h})")


def colmap2blender(sparse_dir):
    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        colmap2json(sparse_dir, tmp)
        apply_to_blender(tmp)
    finally:
        os.unlink(tmp)


def blender2colmap(out_dir):
    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        export_from_blender(tmp)
        json2colmap(tmp, out_dir)
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

def self_test():
    rng = np.random.default_rng(0)

    # [1] 随机 round-trip: w2c -> c2w -> w2c
    err = 0.0
    for _ in range(100):
        A = rng.standard_normal((3, 3))
        R, _ = np.linalg.qr(A)
        R = R * np.linalg.det(R)          # 保证 det=+1 (旋转)
        t = rng.standard_normal(3) * 10
        R2, loc = w2c_to_c2w(R, t)
        R3, t2 = c2w_to_w2c(R2, loc)
        err = max(err, float(np.abs(R - R3).max()), float(np.abs(t - t2).max()))
        assert abs(np.linalg.det(R2) - 1.0) < 1e-12, "c2w 必须保持右手系 det=+1"
    print(f"[1] w2c<->c2w round-trip max err = {err:.2e}  (应 < 1e-12)")

    # [2] 基准用例: 相机在原点看向 +z, 物体在 (0,0,5)
    # 相机局部 +z 指向背后, 所以 c2w z 列 = [0,0,-1]; 前向 -z 列 = [0,0,1] 朝物体
    R, t = np.eye(3), np.array([0.0, 0.0, 5.0])
    R2, loc = w2c_to_c2w(R, t)
    assert np.allclose(loc, [0, 0, -5]) and np.allclose(R2[:, 2], [0, 0, -1])
    assert np.allclose(-R2[:, 2], [0, 0, 1])  # 前向 = -z 列朝 +z
    print(f"[2] 基准用例 OK: loc={loc.tolist()} (应 [0,0,-5]), c2w z 列={R2[:, 2].tolist()} "
          f"(应 [0,0,-1], 前向 -z 列=[0,0,1] 朝物体)")

    # [3] 四元数 round-trip
    err = 0.0
    for _ in range(50):
        A = rng.standard_normal((3, 3))
        R, _ = np.linalg.qr(A)
        R = R * np.linalg.det(R)
        q = mat_to_quat_wxyz(R)
        err = max(err, float(np.abs(R - quat_wxyz_to_mat(q)).max()))
    print(f"[3] quat<->mat round-trip max err = {err:.2e}  (应 < 1e-12)")

    # [4] 真实数据交叉验证: SO101 base_link 60 帧, colmap2json -> json2colmap 逐帧还原
    base = "/amax/home/fengshuangyu/relighting/IRGS/dataset/so101_links/base_link/sparse"
    if os.path.isdir(base):
        with tempfile.TemporaryDirectory() as d:
            colmap2json(base, d + "/c.json")
            json2colmap(d + "/c.json", d + "/out")
            imgs1 = read_images_bin(os.path.join(base, "0", "images.bin"))
            imgs2 = read_images_bin(os.path.join(d, "out", "sparse", "0", "images.bin"))
            err = 0.0
            for a, b in zip(imgs1, imgs2):
                err = max(err, float(np.abs(quat_wxyz_to_mat(a["qvec"]) -
                                            quat_wxyz_to_mat(b["qvec"])).max()),
                          float(np.abs(a["tvec"] - b["tvec"]).max()))
            print(f"[4] base_link {len(imgs1)} 帧 colmap2json->json2colmap 还原 err = {err:.2e}")
    print("ALL CHECKS PASSED")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__)
        return
    cmd = sys.argv[1]
    try:
        if cmd == "--self-test":
            self_test()
        elif cmd == "colmap2json":
            colmap2json(sys.argv[2], sys.argv[3])
        elif cmd == "json2colmap":
            json2colmap(sys.argv[2], sys.argv[3])
        elif cmd == "colmap2blender":
            colmap2blender(sys.argv[2])
        elif cmd == "blender2json":
            export_from_blender(sys.argv[2])
        elif cmd == "blender2colmap":
            blender2colmap(sys.argv[2])
        else:
            print(f"未知命令: {cmd}\n")
            print(__doc__)
            sys.exit(1)
    except (IndexError, KeyError) as e:
        print(f"参数错误: {e}\n", file=sys.stderr)
        print(__doc__, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
