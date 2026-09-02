# Relighting with IRGS

本仓库用于训练可重光照的 IRGS 模型，并支持单物体重光照、SO101 机械臂组装以及多物体轨迹渲染。

```text
多视角图像与相机位姿
  -> RefGS 几何训练
  -> IRGS 材质与光照训练
  -> 单物体重光照 / SO101 与场景资产组装重光照
```

本文档只保留环境安装、数据准备、训练、重光照和组装说明。

## 1. 克隆仓库和 SO-ARM100

SO101 的 URDF 与网格来自 [`TheRobotStudio/SO-ARM100`](https://github.com/TheRobotStudio/SO-ARM100)。本仓库将它登记为 `SO-ARM100` 子模块，地址明确为：

```text
https://github.com/TheRobotStudio/SO-ARM100.git
```

首次下载时使用递归 clone：

```bash
git clone --recurse-submodules https://github.com/CatFSY/relighting.git
cd relighting
```

如果已经 clone 了主仓库，再初始化子模块：

```bash
git submodule update --init --recursive
```

检查子模块来源：

```bash
git config --file .gitmodules --get submodule.SO-ARM100.url
```

应输出：

```text
https://github.com/TheRobotStudio/SO-ARM100.git
```

## 2. 环境安装

### 2.1 基本要求

- Linux；
- NVIDIA GPU 和可用的 CUDA 驱动；
- Conda；
- Git、CMake 和 C++ 编译器。

### 2.2 创建 Conda 环境

```bash
conda env create -f environment.yml
conda activate irgs
```

如果 `irgs` 环境已经存在：

```bash
conda env update -n irgs -f environment.yml
conda activate irgs
```

### 2.3 安装 CUDA 扩展

```bash
pip install submodules/diff-surfel-rasterization
pip install submodules/simple-knn
pip install submodules/raytracing
```

编译并安装 2D Gaussian ray tracer：

```bash
cd submodules/surfel_tracer
rm -rf build
mkdir build
cd build
cmake ..
make -j"$(nproc)"
cd ../../..
pip install submodules/surfel_tracer
```

SO101 网格处理和静态组装还需要：

```bash
pip install pyrender trimesh scipy plyfile
```

检查 PyTorch 和 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 3. 数据准备

### 3.1 单物体数据

`train.sh` 默认读取：

```text
dataset/input_video_frames_<scene>/
├── images/
├── sparse/0/
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin 或 points3D.ply
├── masks/<object_name>/
├── basecolor/
├── roughness/
├── metallic/
└── normal/
```

必需内容：

- `images/`：训练图像；
- `sparse/0/`：COLMAP 相机和稀疏模型；
- `masks/<object_name>/`：与图像对应的物体 mask。

材质监督使用 `basecolor/`、`roughness/`、`metallic/` 和 `normal/`。

### 3.2 SO101 轨迹场景

动态组装需要 SO-ARM100 子模块、训练好的 link 模型和轨迹包：

```text
SO-ARM100/Simulation/SO101/so101_new_calib.urdf

outputs/so101_links/<link_name>/irgs_full/
└── point_cloud/iteration_20000/point_cloud.ply

dataset/guiji1/ 或 dataset/guiji2/
├── trajectories/*.csv
├── configs/execution_contract.json
└── provenance/complete_coordinate_transforms.json
```

## 4. 训练

### 4.1 两阶段统一训练

[`train.sh`](train.sh) 会依次运行：

1. `train_refgaussian.py`：RefGS 几何训练；
2. `train.py`：IRGS 材质和光照训练。

训练场景 `basev8`：

```bash
conda activate irgs
bash train.sh basev8
```

默认输入与输出：

```text
输入：dataset/input_video_frames_basev8/
输出：outputs/input_video_frames_basev8/
```

关键 checkpoint：

```text
outputs/input_video_frames_basev8/refgs/chkpnt50000.pth
outputs/input_video_frames_basev8/irgs/chkpnt20000.pth
```

常用训练命令：

```bash
# 指定 GPU
CUDA_VISIBLE_DEVICES=3 bash train.sh basev8

# 跳过 Stage 1，复用已有 RefGS checkpoint
RUN_REFGS=0 bash train.sh basev8

# 快速检查训练流程
REFGS_ITERS=500 IRGS_ITERS=200 bash train.sh basev8

# 自定义数据和输出目录
SCENE_DIR=/path/to/dataset \
OUTPUT_DIR=/path/to/output \
bash train.sh my_scene
```

强制重新训练会删除该场景已有的 `refgs/` 和 `irgs/` 输出：

```bash
CUDA_VISIBLE_DEVICES=3 FORCE_RETRAIN=1 bash train.sh basev8
```

### 4.2 手动训练

Stage 1：

```bash
CUDA_VISIBLE_DEVICES=0 python train_refgaussian.py \
  -s dataset/input_video_frames_basev8 \
  -m outputs/input_video_frames_basev8/refgs \
  --eval -w \
  --iterations 50000 \
  --mask_dir dataset/input_video_frames_basev8/masks/<object_name> \
  --pred_material_dir dataset/input_video_frames_basev8 \
  --lambda_mask_entropy 0.05 \
  --normal_supervision \
  --lambda_normal_gt 0.1
```

Stage 2：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  -s dataset/input_video_frames_basev8 \
  -m outputs/input_video_frames_basev8/irgs \
  --eval \
  --iterations 20000 \
  --start_checkpoint_refgs outputs/input_video_frames_basev8/refgs/chkpnt50000.pth \
  --mask_dir dataset/input_video_frames_basev8/masks/<object_name> \
  --pred_material_dir dataset/input_video_frames_basev8 \
  --envmap_resolution 128 \
  --diffuse_sample_num 32 \
  --train_ray
```

通常建议直接使用 `train.sh`，它会检查目录、自动选择 mask，并复用已经完成的 checkpoint。

### 4.3 SO101 link 训练

从 SO101 URDF 生成每个 link 的训练数据：

```bash
python scripts/build_so101_link_3dgs.py --out dataset/so101_links
```

并行训练全部 7 个 link：

```bash
bash scripts/run_full_training.sh
```

该脚本默认使用多张 GPU；运行前请按机器配置修改脚本中的 `GPUS`。

已有 RefGS checkpoint、只训练 IRGS Stage 2 时：

```bash
bash scripts/train_so101_links_unified_t005.sh
```

## 5. 单物体重光照

### 5.1 统一入口

内置环境名称为 `env3`、`env6`、`env12`，也可以传入自定义 EXR/HDR 路径。

固定相机和固定环境：

```bash
MODEL_DIR=outputs/input_video_frames_basev8/irgs \
bash relight.sh single basev8 env3 --pose-mode fixed
```

固定相机并旋转环境：

```bash
MODEL_DIR=outputs/input_video_frames_basev8/irgs \
bash relight.sh single basev8 env3 \
  --pose-mode fixed \
  --env-rotate \
  --env-rotate-axis z
```

相机插值并同时旋转环境：

```bash
MODEL_DIR=outputs/input_video_frames_basev8/irgs \
bash relight.sh single basev8 env3 \
  --pose-mode interpolate \
  --steps-per-pair 6 \
  --env-rotate
```

### 5.2 Python 入口

```bash
CUDA_VISIBLE_DEVICES=0 python relight_single.py \
  -m outputs/input_video_frames_basev8/irgs \
  --iteration 20000 \
  --envmap assets/env_map/envmap3.exr \
  --output outputs/input_video_frames_basev8/relight_env3.mp4 \
  --pose-mode fixed \
  --view-id 0
```

常用参数：

- `--pose-mode fixed`：固定相机；
- `--pose-mode interpolate`：在数据集相机之间插值；
- `--env-rotate`：旋转环境贴图；
- `--env-rotations`：指定旋转圈数；
- `--env-rotate-deg-per-sec`：指定环境角速度；
- `--diffuse-samples`、`--light-samples`：指定采样数；
- `--output-buffer`：输出 RGB、diffuse、specular、normal 等 buffer。

## 6. SO101 静态组装

[`scripts/assemble_so101.py`](scripts/assemble_so101.py) 按 URDF 关节树组装独立训练的 link：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/assemble_so101.py \
  --links base_link shoulder_link upper_arm_link lower_arm_link \
          wrist_link gripper_link moving_jaw_so101_v1_link \
  --joints "shoulder_lift=0.3,elbow_flex=-0.8" \
  --views 4 \
  --save-ply \
  --out outputs/so101_links/assembled
```

输出包括组装后的 `merged.ply` 和各视角渲染图。

## 7. 动态轨迹组装与重光照

[`relight_assembled.py`](relight_assembled.py) 是机械臂、桌子和物体动态组装的公开入口。默认采用完整轨迹、刚体 IAS、DS32、LS32 和 `light_t_min=0.10`。

`guiji1` 完整轨迹：

```bash
CUDA_VISIBLE_DEVICES=0 python relight_assembled.py \
  --trajectory-set guiji1 \
  --out outputs/guiji1_relight
```

`guiji2` 完整轨迹：

```bash
CUDA_VISIBLE_DEVICES=0 python relight_assembled.py \
  --trajectory-set guiji2 \
  --out outputs/guiji2_relight
```

轨迹运动时让环境持续旋转：

```bash
CUDA_VISIBLE_DEVICES=0 python relight_assembled.py \
  --trajectory-set guiji2 \
  --trajectory-env-deg-per-sec 90 \
  --env-rotate-axis z \
  --out outputs/guiji2_relight_envrotate
```

也可以使用 shell 入口：

```bash
bash relight.sh assembled guiji1
bash relight.sh assembled guiji2 --trajectory-env-deg-per-sec 90
```

### 自定义轨迹和资产

自定义轨迹包需要包含：

```text
custom_trajectory/
├── trajectories/*.csv
├── configs/execution_contract.json
└── provenance/complete_coordinate_transforms.json
```

运行：

```bash
CUDA_VISIBLE_DEVICES=0 python relight_assembled.py \
  --trajectory-dir /path/to/custom_trajectory \
  --table-ply /path/to/table_model \
  --object-ply /path/to/object_model \
  --object-name cup \
  --out outputs/custom_trajectory_relight
```

`--table-ply` 和 `--object-ply` 可以指向具体 PLY，也可以指向 IRGS 输出目录；程序会自动寻找最新的 `point_cloud.ply`。

## 8. 输出目录

```text
outputs/
├── input_video_frames_<scene>/
│   ├── refgs/
│   ├── irgs/
│   └── relight_*.mp4
├── so101_links/<link_name>/
└── <trajectory>_relight/
```

数据集、checkpoint、PLY、日志和生成视频默认不提交到 Git。

## 9. 常见问题

### 子模块没有下载

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

### 检查 SO-ARM100 指向

```bash
git config --file .gitmodules --get submodule.SO-ARM100.url
git submodule status
```

### 找不到训练数据

检查：

```text
dataset/input_video_frames_<scene>/images/
dataset/input_video_frames_<scene>/sparse/0/cameras.bin
dataset/input_video_frames_<scene>/sparse/0/images.bin
dataset/input_video_frames_<scene>/masks/<object_name>/
```

### 找不到 RefGS checkpoint

```bash
bash train.sh <scene>
```

使用 `RUN_REFGS=0` 时，必须已有对应的 `refgs/chkpnt50000.pth`。

### 查看命令参数

```bash
bash relight.sh
python relight_single.py --help
python relight_assembled.py --help
```
