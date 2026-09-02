# IRGS 当前工作流

本文档是当前本地流程的使用摘要。详细的 SO101 坐标审计和历史轨迹记录仍见
[`so101_guiji_irgs_pipeline.md`](so101_guiji_irgs_pipeline.md)。

## 1. 流程总览

```text
视频/图像
  → 抽帧与相机位姿（VGGT 或 COLMAP）
  → DiffusionRenderer 材质分解
  → SAM3 mask 与材质整理
  → RefGS 第一阶段：几何/高斯
  → IRGS 第二阶段：材质/光照
  → 单体重光照 或 多物体轨迹组装重光照
```

训练和推理使用同一个 IRGS 坐标约定；重光照阶段不会重新估计相机或物体尺寸。

## 2. 数据目录约定

场景名为 `basev8` 时，默认目录是：

```text
数据集：IRGS/dataset/input_video_frames_basev8/
输出：  IRGS/outputs/input_video_frames_basev8/
```

数据集至少需要：

```text
images/
sparse/0/cameras.bin
sparse/0/images.bin
```

若使用材质监督，还需要：

```text
basecolor/
roughness/
metallic/
normal/
masks/<object>/
```

完整的视频预处理入口在仓库上级目录：

```text
/amax/home/fengshuangyu/relighting/run_pipeline_irgs.sh
```

## 3. 重新训练

统一训练入口：[`train.sh`](../train.sh)。

### 默认训练

```bash
cd /amax/home/fengshuangyu/relighting/IRGS
bash train.sh basev8
```

它依次执行：

1. `train_refgaussian.py`：生成 RefGS 几何 checkpoint；
2. `train.py`：读取 RefGS checkpoint，训练 IRGS 材质和光照。

输出：

```text
outputs/input_video_frames_basev8/refgs/chkpnt50000.pth
outputs/input_video_frames_basev8/irgs/chkpnt20000.pth
```

已有 checkpoint 时自动跳过对应阶段。

### 常用训练命令

```bash
# 指定 GPU
CUDA_VISIBLE_DEVICES=3 bash train.sh basev8

# 强制重训两个阶段（会删除该场景 refgs/ 和 irgs/）
CUDA_VISIBLE_DEVICES=3 FORCE_RETRAIN=1 bash train.sh basev8

# 跳过第一阶段，复用已有 chkpnt50000.pth
RUN_REFGS=0 bash train.sh basev8

# 快速排错
REFGS_ITERS=500 IRGS_ITERS=200 bash train.sh basev8

# 自定义数据和输出目录
SCENE_DIR=/path/to/dataset OUTPUT_DIR=/path/to/output bash train.sh my_scene
```

第一阶段包含几何重建、normal supervision 和 distortion loss；第二阶段读取
base color、roughness、normal 及 mask 监督，并训练环境光和间接光相关参数。

## 4. 当前默认推理配置

组合轨迹和单体入口默认采用：

```text
Gaussian 代理：icosphere320（每个 Gaussian 由 320 面代理表示）
漫反射采样 DS：32
环境重要性采样 LS：32
light_t_min：0.10
环境：fill=0.35，key=2500
组合 BVH：全刚体 IAS
```

默认环境贴图：

```text
assets/env_map/pointlike_camera_key_light_fill035_key2500.exr
```

## 5. 单个物体重光照

入口：[`relight_single.py`](../relight_single.py)。

该脚本加载一个已经训练好的 IRGS 模型和一个 EXR/HDR 环境，支持：

- 固定相机；
- 相机在数据集相机之间做平移线性插值和旋转 SLERP；
- 环境固定；
- 环境连续旋转；
- 位姿插值和环境旋转同时开启。

### 固定相机、环境旋转

```bash
CUDA_VISIBLE_DEVICES=0 python relight_single.py \
  -m outputs/input_video_frames_basev8/irgs_stage2_with_pseudo \
  --iteration 20000 \
  --envmap assets/env_map/envmap3.exr \
  --output outputs/input_video_frames_basev8/relight_envrotate.mp4 \
  --pose-mode fixed --view-id 0 --env-rotate
```

### 位姿插值和环境同时旋转

```bash
CUDA_VISIBLE_DEVICES=0 python relight_single.py \
  -m outputs/input_video_frames_basev8/irgs_stage2_with_pseudo \
  --iteration 20000 \
  --envmap assets/env_map/envmap3.exr \
  --output outputs/input_video_frames_basev8/relight_pose_envrotate.mp4 \
  --pose-mode interpolate --steps-per-pair 6 --env-rotate \
  --env-rotate-axis z --env-rotations 1
```

`pose-mode interpolate` 改变的是相机位姿，物体仍保持训练坐标系中的固定位置。

## 6. 多物体和机械臂轨迹组装

公开入口：[`relight_assembled.py`](../relight_assembled.py)。

底层组装器：[`scripts/render_guiji_irgs.py`](../scripts/render_guiji_irgs.py)。

它可以组合：

- SO101 七个 link GS；
- 桌子 GS；
- 杯子、Lego 或其他刚体 GS；
- URDF 关节树和 Isaac 轨迹 CSV。

每帧流程是：

```text
读取轨迹状态
  → URDF FK 得到各 link 世界位姿
  → 更新动态 GS 的 xyz/scale/rotation
  → 更新 IAS 实例矩阵
  → rasterization/G-buffer
  → DS/LS 采样、BVH visibility、BRDF 与间接光
  → 输出 RGB
```

静态物体和每个刚体组件的 GAS 只构建一次；轨迹过程中只更新 IAS 实例矩阵，避免每帧重建数千万代理三角形。

### guiji2 完整轨迹，环境固定

```bash
CUDA_VISIBLE_DEVICES=0 python relight_assembled.py \
  --trajectory-set guiji2 \
  --out outputs/guiji2_relight
```

### guiji2 完整轨迹，环境持续旋转

```bash
CUDA_VISIBLE_DEVICES=0 python relight_assembled.py \
  --trajectory-set guiji2 \
  --trajectory-env-deg-per-sec 90 \
  --env-rotate-axis z \
  --out outputs/guiji2_relight_envrotate
```

组合入口默认输出 640×360、30 FPS 的轨迹视频；可用底层脚本的 `--width`、
`--height`、`--fps` 覆盖。

## 7. 坐标系和组装约定

- Isaac 世界坐标：米制、`+Z up`；
- URDF `origin` 使用 `sxyz` 语义；
- link 局部坐标由 URDF link frame 定义；
- link GS 在各自 link frame 中训练和保存；
- 场景内部统一使用 `SCENE_SCALE=4`；
- 机器人 link 已在相应的放大坐标中，桌子/杯子先完成源坐标到 Isaac 世界的审计变换，再乘场景缩放；
- 四元数统一使用 `wxyz`；
- 环境旋转默认绕 Isaac world-Z 轴；
- mesh 只用于坐标审计、尺寸检查和漂浮点过滤，最终视频由 GS 渲染。

详细矩阵、杯子/桌子转换和 URDF 资料见
[`so101_guiji_irgs_pipeline.md`](so101_guiji_irgs_pipeline.md)。

## 8. 诊断和测速

常用工具：

```text
benchmark_relighting_fps.py       单帧 FPS
benchmark_relighting_stages.py    分阶段耗时
benchmark_trace_modes.py          BVH/OptiX trace 对比
benchmark_env_switch_latency.py   环境切换延迟
verify_refgs_normal_frame.py      第一阶段法线验证
my_render_pose_normal_depth_videos.py  法线/深度视频
my_render_gt_vs_render_grid.py    GT 与渲染材质对比
```

训练和渲染产生的 checkpoint、PLY、视频和 JSON 报告位于 `outputs/` 下；这些是结果数据，不是代码依赖。历史测速日志删除不会影响训练或推理。

## 9. 常见问题

### 找不到数据集

确认 `dataset/input_video_frames_<scene>` 存在，并且包含 `images/` 与
`sparse/0/cameras.bin`、`images.bin`。

### 找不到 RefGS checkpoint

先运行：

```bash
bash train.sh <scene>
```

或确认 `RUN_REFGS=0` 时对应的 `refgs/chkpnt50000.pth` 已存在。

### 视频分辨率不对

单体入口使用模型相机分辨率；组合入口由 `--width` 和 `--height` 指定。视频编码前会裁成偶数尺寸，这是 FFmpeg/YUV420 的要求。

### env 旋转和轨迹是否可以同时存在

可以。单体入口使用 `--pose-mode interpolate --env-rotate`；组合入口使用
`--full-video`（默认）并加 `--trajectory-env-deg-per-sec` 或
`--trajectory-env-rotations`。
