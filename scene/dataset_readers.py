#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import math
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json, cv2
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import pyexr
import imageio as imageio
from utils.graphics_utils import srgb_to_rgb, rgb_to_srgb
from tqdm import tqdm


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    K: np.array # #
    FovY: np.array
    FovX: np.array
    image: np.array
    mask: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    # GT material maps (arb_render_data): PIL images, None for other datasets
    albedo: np.array = None   # sRGB-encoded albedo (RGB)
    orm: np.array = None      # linear [occlusion, roughness, metallic] (RGB)
    normal: np.array = None   # camera-space normal EXR (RGB), loaded as linear float

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder,
                      mask_dir=None, pred_material_dir=None, white_background=False):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            # #
            K = np.array([
                [focal_length_x, 0, intr.params[1]],
                [0, focal_length_x, intr.params[2]],
                [0, 0, 1],
            ])
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            # #
            K = np.array([
                [focal_length_x, 0, intr.params[2]],
                [0, focal_length_y, intr.params[3]],
                [0, 0, 1],
            ])
        # #
        elif intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            K = np.array([
                [focal_length_x, 0, intr.params[1]],
                [0, focal_length_x, intr.params[2]],
                [0, 0, 1],
            ])
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        if not os.path.exists(image_path):
            image_path = image_path.replace('.JPG', '.jpg')
        image = Image.open(image_path)

        #if intr.model=="SIMPLE_RADIAL":
        #    image = cv2.undistort(np.array(image), K, np.array([intr.params[3], 0,0,0]))
        #    image = Image.fromarray(image.astype('uint8')).convert('RGB')
        # #
        real_im_scale = image.size[0] / width
        K[:2] *=  real_im_scale

        # ---- 物体 mask（SAM3 多视角分割输出，RGBA PNG，alpha 通道 = 物体区域）----
        # mask_dir 设置时：mask 外区域按 white_background 合成到背景（训练用 mask 后图像），
        # 同时 cam.mask 参与 mask 熵正则与各材质 loss 的像素加权
        mask = None
        mask_f = None
        if mask_dir is not None:
            mask_path = os.path.join(mask_dir, image_name + ".png")
            if os.path.exists(mask_path):
                m_arr = np.asarray(Image.open(mask_path))
                m_alpha = m_arr[..., 3] if (m_arr.ndim == 3 and m_arr.shape[-1] == 4) else m_arr
                # SAM3 mask 分辨率可能与图像不同（如 288x512 vs 512x960），先对齐到图像尺寸
                if m_alpha.shape[:2] != image.size[::-1]:
                    m_alpha = cv2.resize(m_alpha.astype(np.float32), image.size,
                                         interpolation=cv2.INTER_LINEAR)
                mask = m_alpha > 127.5
                mask_f = mask[..., None].astype(np.float32)
                bg = np.ones(3) if white_background else np.zeros(3)
                arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
                arr = arr * mask_f + bg * (1.0 - mask_f)
                image = Image.fromarray((arr * 255.0).astype(np.byte), "RGB")
            else:
                print(f"\n[Warning] object mask not found, view without mask/material supervision: {mask_path}")

        # ---- 预测材质（DiffusionRenderer 输出），本身无 mask，用物体 mask 裁掉背景 ----
        #   basecolor: sRGB 编码（loadCam 里 srgb_to_rgb 转线性，与 arb 预测路径一致）
        #   roughness/metallic: 单通道灰度 linear [0,1]，拼成伪 ORM（loadCam 只取 G/B）
        #   normal: 相机空间 n*0.5+0.5 编码，转世界系（COLMAP 存储的 R = W2C_rot^T，
        #           故 C2W_rot = R，行向量变换为 n @ R.T）
        albedo = None
        orm = None
        normal = None
        target_size = image.size  # (W, H)

        def _match_size(arr):
            # 预测材质分辨率可能与图像不同，对齐到图像尺寸
            if arr.shape[:2] != target_size[::-1]:
                arr = cv2.resize(arr.astype(np.float32), target_size,
                                 interpolation=cv2.INTER_LINEAR)
            return arr

        if pred_material_dir is not None:
            alb_path = os.path.join(pred_material_dir, "basecolor", image_name + ".png")
            if os.path.exists(alb_path):
                a = _match_size(np.asarray(Image.open(alb_path).convert("RGB"))).astype(np.float32)
                if mask_f is not None:
                    a = a * mask_f
                albedo = Image.fromarray(a.astype(np.uint8), "RGB")

            rough_path = os.path.join(pred_material_dir, "roughness", image_name + ".png")
            metal_path = os.path.join(pred_material_dir, "metallic", image_name + ".png")
            if os.path.exists(rough_path) and os.path.exists(metal_path):
                r_arr = _match_size(np.asarray(Image.open(rough_path).convert("RGB"))).astype(np.float32)
                mt_arr = _match_size(np.asarray(Image.open(metal_path).convert("RGB"))).astype(np.float32)
                occ = np.full(r_arr.shape[:2] + (1,), 255.0, dtype=np.float32)
                if mask_f is not None:
                    r_arr = r_arr * mask_f
                    mt_arr = mt_arr * mask_f
                    occ = occ * mask_f[..., :1]
                orm = Image.fromarray(np.concatenate(
                    [occ, r_arr[..., :1], mt_arr[..., :1]], axis=-1).astype(np.uint8))

            normal_path = os.path.join(pred_material_dir, "normal", image_name + ".png")
            if os.path.exists(normal_path):
                n = _match_size(np.asarray(Image.open(normal_path).convert("RGB"))).astype(np.float32) / 255.0 * 2.0 - 1.0
                if mask_f is not None:
                    n = n * mask_f
                normal = n @ R.T

        cam_info = CameraInfo(uid=uid, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              mask=mask, albedo=albedo, orm=orm, normal=normal)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')

    # 伪真值监督加载统计：确认每帧的 GT 材质（albedo/orm/normal）都读到了
    if pred_material_dir is not None:
        n_alb = sum(1 for c in cam_infos if c.albedo is not None)
        n_orm = sum(1 for c in cam_infos if c.orm is not None)
        n_nrm = sum(1 for c in cam_infos if c.normal is not None)
        n_msk = sum(1 for c in cam_infos if c.mask is not None)
        print(f"[监督] GT 加载统计: albedo {n_alb}/{len(cam_infos)} | "
              f"orm(rough+metal) {n_orm}/{len(cam_infos)} | "
              f"normal {n_nrm}/{len(cam_infos)} | mask {n_msk}/{len(cam_infos)} "
              f"(pred_material_dir={pred_material_dir})", flush=True)
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    # #
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        print('Load Ply color and normals failed, random init')
        colors = np.random.rand(*positions.shape) / 255.0
        normals = np.random.rand(*positions.shape)
        normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8,
                        mask_dir=None, pred_material_dir=None, white_background=False):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir),
                                           mask_dir=mask_dir, pred_material_dir=pred_material_dir,
                                           white_background=white_background)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    # #
    spc_ply_path = os.path.join(path, "sparse/0/points_spc.ply")
    if os.path.exists(spc_ply_path):
        ply_path = spc_ply_path
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        # fovx = contents["camera_angle_x"]
        fovx = contents.get("camera_angle_x", None)
        if fovx is None:
            w = contents["w"]
            fl_x = contents["fl_x"]
            fovx = 2 * math.atan(w / (2 * fl_x))
        frames = contents["frames"]
        for idx, frame in enumerate(tqdm(frames)):
            # if not idx % 10 == 0:
            #     continue
            file_path = frame["file_path"]
            if '.png' not in file_path:
                file_path = file_path + extension
            cam_name = os.path.join(path, file_path)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            ### NOTE !!!!!
            # Here R has been transposed, R = w2c.T
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            if norm_data.shape[-1] == 4:
                mask = norm_data[:, :, 3] > 0.5
            else:
                mask = None
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            # #
            fo = fov2focal(fovx, image.size[0])

            W,H = image.size[0], image.size[1]
            K = np.array([
                [fo, 0, W/2],
                [0, fo, H/2],
                [0, 0, 1],
            ])

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx
            # #
            # For blender datasets, we consider its camera center offset is zero (ideal camera)
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image, mask=mask,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    # print("Reading Training Transforms")
    # train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    # print("Reading Test Transforms")
    # test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []
    
    if eval:
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
        print("Reading Test Transforms")
        test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    else:
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
        print("Reading Test Transforms")
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def load_img_rgb(path):
    
    if path.endswith(".exr"):
        exr_file = pyexr.open(path)
        img = exr_file.get()
        img[..., 0:3] = rgb_to_srgb(img[..., 0:3])
        # img[..., 0:3] = rgb_to_srgb(img[..., 0:3], clip=False)
    else:
        img = imageio.imread(path)
        img = img / 255
        # img[..., 0:3] = srgb_to_rgb(img[..., 0:3])
    return img

def load_mask_bool(mask_file):
    mask = imageio.imread(mask_file, mode='L')
    mask = mask.astype(np.float32)
    mask[mask > 0.5] = 1.0

    return mask

def readCamerasFromTransforms3(path, transformsfile, white_background, extension=".png", debug=False):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(tqdm(frames)):
            # if not idx % 10 == 0:
            #     continue
            # if idx > 1:
            #     break
            image_path = os.path.join(path, frame["file_path"] + extension)
            mask_path = image_path.replace("_rgb.exr", "_mask.png")
            image_name = Path(image_path).stem

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            bg = 1 if white_background else 0
            
            image = load_img_rgb(image_path)
            mask = load_mask_bool(mask_path).astype(np.float32)

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            arr = image[..., :3] * mask[..., None] + bg * (1 - mask[..., None])
            
            # H, W = image.shape[:2]
            # fo = fov2focal(fovx, W)
            # K = np.array([
            #     [fo, 0, W/2],
            #     [0, fo, H/2],
            #     [0, 0, 1],
            # ])
            
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            fo = fov2focal(fovx, image.size[0])

            W,H = image.size[0], image.size[1]
            K = np.array([
                [fo, 0, W/2],
                [0, fo, H/2],
                [0, 0, 1],
            ])

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image, mask=mask,
                                        image_path=image_path, image_name=image_name,
                                        width=image.size[0], height=image.size[1]))
    return cam_infos

def readSynthetic4RelightInfo(path, white_background, eval, debug=False):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms3(path, "transforms_train.json", white_background, "_rgb.exr", debug=debug)
    if eval:
        print("Reading Test Transforms")
        test_cam_infos = readCamerasFromTransforms3(path, "transforms_test.json", white_background, "_rgba.png", debug=debug)
    else:
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0

        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)

    return scene_info

def readCamerasFromTransforms2(path, transformsfile, white_background, 
                               extension=".png", benchmark_size = 512, debug=False):
    cam_infos = []
    
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(tqdm(frames, leave=False)):
            if os.path.exists(os.path.join(path, frame["file_path"] + '.png')):
                image_path = os.path.join(path, frame["file_path"] + '.png')
            else:
                image_path = os.path.join(path, frame["file_path"] + '.exr')
                
            mask_item = frame["file_path"].replace("test", "test_mask").replace("train", "train_mask")
            if os.path.exists(os.path.join(path, mask_item + '.png')):
                mask_path = os.path.join(path, mask_item + '.png')
            else:
                mask_path = os.path.join(path, mask_item + '.exr')
            
            image_name = Path(image_path).stem

            c2w = np.array(frame["transform_matrix"])
            c2w[:3, 1:3] *= -1

            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]

            image = load_img_rgb(image_path)
            mask = load_mask_bool(mask_path).astype(np.float32)
            image = cv2.resize(image, (benchmark_size, benchmark_size), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (benchmark_size, benchmark_size), interpolation=cv2.INTER_AREA)

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            image = image * mask[..., None] + bg * (1 - mask[..., None])

            image = Image.fromarray(np.array(image*255.0, dtype=np.byte), "RGB")
            fo = fov2focal(fovx, image.size[0])

            W,H = image.size[0], image.size[1]
            K = np.array([
                [fo, 0, W/2],
                [0, fo, H/2],
                [0, 0, 1],
            ])

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image, mask=mask,
                                        image_path=image_path, image_name=image_name,
                                        width=image.size[0], height=image.size[1]))

            if debug and idx >= 5:
                break

    return cam_infos

def readStanfordORBInfo(path, white_background, eval, extension=".exr", benchmark_size = 512, debug=False):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms2(path, "transforms_train.json", white_background, 
                                                 extension, benchmark_size, debug=debug)
    if eval:
        print("Reading Test Transforms")
        test_cam_infos = readCamerasFromTransforms2(path, "transforms_test.json", white_background, 
                                                    extension, benchmark_size, debug=debug)
    else:
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0

        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)

    return scene_info

def readArbCameras(path, view_ids, env_idx, white_background, transform_matrices, pred_material_dir=None):
    """Read a set of arb_render_data views under one lighting condition.

    Data layout (per object dir):
      color_{env}_{view:02d}.png : RGBA, straight-alpha, sRGB (mask = alpha)
      camera.json                : {"transform_matrix": [V, 4, 4]}  Blender c2w
    Intrinsics follow the Blender physical camera used to render the set
    (lens=50mm, sensor width=36mm, square 512/256 frames), so fx == fy and
    the principal point is centered (an ideal pinhole).
    """
    cam_infos = []
    # Blender physical camera -> horizontal FoV (square frame, fit=HORIZONTAL)
    LENS_MM = 50.0000114440918
    SENSOR_WIDTH_MM = 36.0
    fovx = 2.0 * math.atan(0.5 * SENSOR_WIDTH_MM / LENS_MM)

    for view_id in tqdm(view_ids):
        # NeRF/Blender 'transform_matrix' is a camera-to-world transform
        c2w = np.array(transform_matrices[view_id])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_path = os.path.join(path, f"color_{env_idx}_{view_id:02d}.png")
        image_name = Path(image_path).stem
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
        norm_data = im_data / 255.0
        if norm_data.shape[-1] == 4:
            mask = norm_data[:, :, 3] > 0.5
        else:
            mask = None
        arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

        # 材质图加载：
        #   pred_material_dir 设置时 -> 读 DiffusionRenderer 预测结果
        #     albedo 读 basecolor/{env_idx}/{view:02d}.png（sRGB，与 GT 同编码）；
        #     normal 为相机空间 n*0.5+0.5 编码，与 GT EXR 同约定
        #   否则 -> 读 $SRC 下的 GT 材质图（albedo_*.png / orm_*.png / normal_*.exr）
        if pred_material_dir is not None:
            albedo_path = os.path.join(pred_material_dir, 'basecolor',
                                       f'basecolor_{env_idx}_{view_id:02d}.png')
            albedo = Image.open(albedo_path).convert('RGB') if os.path.exists(albedo_path) else None

            # roughness / metallic 各为单通道灰度 PNG（linear [0,1]），拼成伪 ORM：
            # R=occlusion(=1) G=roughness B=metallic（loadCam 只取 G/B 两通道）
            rough_path = os.path.join(pred_material_dir, 'roughness',
                                      f'roughness_{env_idx}_{view_id:02d}.png')
            metal_path = os.path.join(pred_material_dir, 'metallic',
                                      f'metallic_{env_idx}_{view_id:02d}.png')
            if os.path.exists(rough_path) and os.path.exists(metal_path):
                r_arr = np.asarray(Image.open(rough_path).convert('RGB'))
                m_arr = np.asarray(Image.open(metal_path).convert('RGB'))
                occ = np.full(r_arr.shape[:2] + (1,), 255, dtype=np.uint8)
                orm = Image.fromarray(np.concatenate(
                    [occ, r_arr[..., :1], m_arr[..., :1]], axis=-1))
            else:
                orm = None

            normal_path = os.path.join(pred_material_dir, 'normal',
                                       f'normal_{env_idx}_{view_id:02d}.png')
            if os.path.exists(normal_path):
                n = np.asarray(Image.open(normal_path).convert('RGB')).astype(np.float32) / 255.0 * 2.0 - 1.0
                n = n @ np.array(transform_matrices[view_id], dtype=np.float32)[:3, :3].T
                normal = n
            else:
                normal = None
        else:
            # GT material maps (lighting-independent, shared by all env_idx):
            #   albedo_{view:02d}.png : sRGB-encoded albedo (RGBA, alpha = mask)
            #   orm_{view:02d}.png    : rendered Raw => linear, R=occlusion,
            #                           G=roughness, B=metallic
            #   normal_{view:02d}.exr : camera-space normal, BGR, [-1, 1]
            albedo_path = os.path.join(path, f"albedo_{view_id:02d}.png")
            albedo = Image.open(albedo_path).convert("RGB") if os.path.exists(albedo_path) else None
            orm_path = os.path.join(path, f"orm_{view_id:02d}.png")
            orm = Image.open(orm_path).convert("RGB") if os.path.exists(orm_path) else None
            normal_path = os.path.join(path, f"normal_{view_id:02d}.exr")
            if os.path.exists(normal_path):
                n = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
                if n is not None:
                    n = cv2.cvtColor(n[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32) * 2.0 - 1.0
                    n = n @ np.array(transform_matrices[view_id], dtype=np.float32)[:3, :3].T
                    # keep float32 array in [-1, 1]; do NOT quantize to uint8/PIL
                    normal = n
                else:
                    normal = None
            else:
                normal = None

        W, H = image.size[0], image.size[1]
        fo = fov2focal(fovx, W)
        K = np.array([
            [fo, 0, W / 2],
            [0, fo, H / 2],
            [0, 0, 1],
        ])
        fovy = focal2fov(fov2focal(fovx, W), H)

        cam_infos.append(CameraInfo(uid=view_id, R=R, T=T, K=K, FovY=fovy, FovX=fovx,
                                    image=image, mask=mask,
                                    image_path=image_path, image_name=image_name,
                                    width=W, height=H, albedo=albedo, orm=orm, normal=normal))
    return cam_infos

def readArbInfo(path, white_background, eval, env_idx=0, num_test=4, pred_material_dir=None):
    """Scene reader for a single arb_render_data object dir.

    All training images come from ONE lighting condition (``env_idx``), which
    is what IRGS expects (it recovers a single unknown environment map).
    ``num_test`` views are held out (evenly spaced) for NVS evaluation.
    """
    with open(os.path.join(path, "camera.json")) as f:
        camdata = json.load(f)
    transform_matrices = camdata["transform_matrix"]
    num_views = len(transform_matrices)

    # evenly-spaced held-out test views (e.g. 36 views, num_test=4 -> 0,9,18,27)
    if eval and num_views > num_test:
        step = num_views // num_test
        test_ids = [i * step for i in range(num_test)]
    else:
        test_ids = []
    train_ids = [i for i in range(num_views) if i not in test_ids]
    print(f"[Arb] path={path}\n[Arb] env_idx={env_idx}, num_views={num_views}, "
          f"train={len(train_ids)}, test={len(test_ids)} (ids={test_ids})")

    print("Reading Arb Training Cameras")
    train_cam_infos = readArbCameras(path, train_ids, env_idx, white_background, transform_matrices, pred_material_dir)
    if test_ids:
        print("Reading Arb Test Cameras")
        test_cam_infos = readArbCameras(path, test_ids, env_idx, white_background, transform_matrices, pred_material_dir)
    else:
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # No colmap data: start from a random point cloud inside the scene bounds
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Synthetic4Relight": readSynthetic4RelightInfo,
    "StanfordORB": readStanfordORBInfo,
    "Arb": readArbInfo,
}