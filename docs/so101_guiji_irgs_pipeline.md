# SO101 轨迹场景全 3DGS 表达与重光照流程

## 1. 目标与状态

本流程将 Isaac Sim 生成的 SO101 抓取轨迹，转换为由 IRGS 资产完整表达的动态场景。最终渲染包含：

- SO101 的 7 个 link 3DGS；
- 桌子 3DGS；
- 杯子 3DGS；
- HDR 环境光、旋转环境光或程序化单光源；
- 全场景统一 BVH，用于遮挡、阴影和 IRGS 近似间接光。

最终画面不使用 mesh。Mesh 只用于坐标审计、尺寸验证和过滤漂浮点。

当前已完成独立 link 训练、机械臂 mesh/3DGS 对齐、`guiji1` Isaac 轨迹解析、newbase/机器人/杯子全 3DGS 动态场景以及 375 帧完整轨迹视频。最终 80 mm 杯子 USD、leveled/scaled newbase USD、源到 V32 的完整变换资料包和两个 Isaac 相机均已接入。主相机环境旋转、主相机环绕和第二相机完整轨迹三种模式已全部实测通过。

## 2. 数据与代码

机械臂：

```text
URDF: SO-ARM100/Simulation/SO101/so101_new_calib.urdf
数据生成: IRGS/scripts/build_so101_link_3dgs.py
位姿验证: IRGS/scripts/validate_poses.py
静态组装: IRGS/scripts/assemble_so101.py
mesh 参考: IRGS/scripts/gt_arm_render.py
数值对齐: IRGS/scripts/verify_align.py
link IRGS: IRGS/outputs/so101_links/<link>/irgs_full/
```

参与渲染的 link：

```text
base_link
shoulder_link
upper_arm_link
lower_arm_link
wrist_link
gripper_link
moving_jaw_so101_v1_link
```

桌子：

```text
IRGS: IRGS/outputs/input_video_frames_newbase/irgs/
训练数据: IRGS/dataset/input_video_frames_newbase/
源 mesh: IRGS/outputs/input_video_frames_newbase/refgs/mesh_stage1/stage1_mesh_clean_largest.glb
最终 USD: IRGS/dataset/guiji1/assets/table/newbase_table_leveled_scaled.usd
```

杯子：

```text
IRGS: IRGS/outputs/input_video_frames_beizi/irgs/
训练数据: IRGS/dataset/input_video_frames_beizi/
粗 mesh: IRGS/outputs/input_video_frames_beizi/refgs/mesh_stage1/stage1_mesh_clean_largest.glb
```

轨迹：

```text
根目录: IRGS/dataset/guiji1/
实际轨迹: trajectories/formal_task_lift_trace.csv
命令轨迹: trajectories/task_lift_command_schedule.csv
执行合同: configs/execution_contract.json
接触记录: contacts/formal_task_lift_contacts.csv
参考视频: videos/base_mesh_cup_pick_lift_v0.mp4
最终杯子 USD: assets/cup/cup_canonical_80mm_kraft_black_lid_converted.usd
实际机器人 USD: assets/robot/SO-ARM101-USD-NO-CAMERA.usd
```

当前动态代码：

```text
IRGS/scripts/render_guiji_irgs.py
IRGS/scripts/make_single_source_env.py
```

组装时 PLY 阶段选择错误及其修复记录见：

```text
IRGS/docs/guiji3_model_selection_bug.md
```

## 3. 坐标系约定

### 3.1 URDF 与 link frame

URDF 没有独立定义绝对世界坐标。机械臂以 `base_link` 为根：

\[
T_{base\leftarrow base}=I
\]

各 child link 通过 joint tree 递推：

\[
T_{parent\leftarrow child}(q)=T_{origin}(xyz,rpy)T_{axis}(q)
\]

URDF RPY 使用固定轴 XYZ，对应：

```python
euler_matrix(roll, pitch, yaw, "sxyz")
```

### 3.2 link 3DGS 局部坐标

每个 STL 先通过 `<visual><origin>` 放入所属 link frame，然后整体放大 4 倍：

\[
p_{link,GS}=4(R_{visual}p_{STL}+t_{visual})
\]

所以 link PLY 的语义是：URDF link 局部坐标，方向与 URDF 一致，长度数值是米制坐标的 4 倍。

### 3.3 最终场景

最终场景采用 Isaac world 的轴向与原点，但统一放大 4 倍：

\[
p_{scene}=4p_{IsaacWorld,m}
\]

这样既兼容已经按 4 倍尺度训练的机械臂 link，也能避免小尺寸机械臂在 rasterizer/光追中的近裁剪问题。

### 3.4 相机

IRGS 使用 COLMAP/GS 风格相机：X 向右、Y 向下、Z 向前，输入 `c2w`，内部求逆得到 `w2c`。

合同提供两个相机。主相机 `/World/TaskLiftCamera` 用于整臂构图：

```text
eye    = [0.55, -0.56, 0.48] m
target = [0.00,  0.16, 0.14] m
up     = [0.00,  0.00, 1.00]
resolution = 960 × 540
```

第二相机 `/World/CupContactCamera` 是杯子接触区域的桌面侧视近景：

```text
eye    = [0.45, 0.20, 0.14] m
target = [0.10, 0.20, 0.05] m
up     = [0.00, 0.00, 1.00]
focal_length = 42 mm
resolution = 960 × 540
```

主相机合同没有 FOV。根据 Isaac 关键帧构图反推，当前统一使用垂直 FOV `27°`，约等于 16:9 下水平 FOV `46°`；这也与第二相机 42 mm 焦距的近似垂直视场一致。FOV 只影响取景，不影响资产对齐。渲染器通过 `--camera-name main|contact` 选择固定相机。

## 4. 轨迹字段

`formal_task_lift_trace.csv` 包含 1496 个 physics step，频率 120 Hz，总时长 12.4667 秒。

关节映射：

| CSV/Isaac | URDF joint |
|---|---|
| `Rotation` | `shoulder_pan` |
| `Pitch` | `shoulder_lift` |
| `Elbow` | `elbow_flex` |
| `Wrist_Pitch` | `wrist_flex` |
| `Wrist_Roll` | `wrist_roll` |
| `Jaw` | `gripper` |

渲染使用实际执行角 `<joint>__actual_q_rad`，而不是命令角。杯子直接使用：

```text
object_x_m, object_y_m, object_z_m
object_qw, object_qx, object_qy, object_qz
```

杯子四元数顺序为 `wxyz`。

完整视频按 `stride=4` 抽样并强制保留最终 step 1495，因此输出 30 FPS、共 375 帧、12.5 秒。

## 5. 机械臂变换

合同指出实际 NVIDIA USD root 与 URDF `base_link` 不重合，并提供：

```text
active_usd_root_from_urdf_base_link
```

加上 USD root 的世界位姿：

\[
T_{world\leftarrow base}=T_{world\leftarrow activeRoot}T_{activeRoot\leftarrow base}
\]

每帧用 `actual_q_rad` 做 URDF FK：

\[
T_{world\leftarrow link}(k)=T_{world\leftarrow base}T_{base\leftarrow link}(q_k)
\]

link PLY 已是 4 倍局部坐标，所以：

\[
p'_{scene}=R_{world\leftarrow link}p_{link,GS}+4t_{world\leftarrow link}
\]

高斯旋转同步更新：

\[
q'_{GS}=q_{world\leftarrow link}\otimes q_{GS}
\]

刚体运动不改变 link 高斯自身 scale。

## 6. 桌子对齐

newbase 源 GLB 与 newbase 3DGS 位于同一重建坐标系。最终 leveled/scaled USD 与源 GLB 顶点一一对应；用全部 59,846 个顶点拟合源坐标到 Isaac world 的仿射变换，平均/最大残差为 `8.1e-9 / 3.3e-8 m`。线性部分的三个奇异值均为 `1.7511800185`，即旋转加合同规定的统一缩放。

最终 USD 世界范围为：

```text
X = [-0.321974, 0.321974] m
Y = [-0.025512, 0.607557] m
Z = [-0.468015, 0.009461] m
```

所以 newbase 变换已达到顶点级数值闭环：

\[
p_{table,scene}=4(A_{table}p_{table,GS}+t_{table})
\]

当前依据源 cleaned GLB 包围盒并放宽 2 cm 过滤：

```text
原始：76845
保留：76787
过滤：58
```

## 7. 全场景合并与动态 BVH

第 `k` 帧：

\[
G_{scene}(k)=G_{table}\cup G_{cup}(k)\cup\bigcup_{i=1}^{7}G_{link_i}(k)
\]

当前点数：

```text
桌子：  76787
机械臂：140235
杯子：  77515
总计： 294537
```

材质、SH 和 opacity 在轨迹中不变。每帧只更新 `xyz/scale/rotation`。第一帧调用 `build_bvh()`，后续保持拓扑不变，调用 `update_bvh()`。BVH 更新通常约 7–25 ms。

这使桌子、杯子和不同 link 进入同一可见性查询，支持跨物体遮挡、阴影和 IRGS 近似间接光。

早期验证设置（仅用于复现旧输出，不再作为当前 relighting 默认值）：

```text
base_color_min = 0.0
light_t_min = 0.15（4 倍场景单位）
diffuse_sample_num = 64
```

IRGS 不是完整 path tracer。它提供环境直接光、Gaussian visibility、跨物体阴影和近似一次间接光。Metallic 只有在 `use_metallic_brdf=True` 时进入最终 BRDF。

## 8. 环境旋转与单光源

普通 HDR 已验证 `envmap3/6/12.exr`。旧版对比视频使用 `envmap3.exr` 与 `light_t_min=0.15`；当前轨迹 relighting 配置见下方“当前采用配置”。

固定轨迹帧时，机械臂、桌子、杯子和相机保持不动，只更新 `EnvLight.transform`。

IRGS lat-long 方向定义为：

```text
d = (sin(theta)sin(phi), cos(theta), -sin(theta)cos(phi))
```

环境贴图参数化以 Y 为极轴，但机械臂场景继承 Isaac 的 Z-up。当前“光源绕场景一周”统一对 `EnvLight.transform` 施加 world-Z 旋转，即命令行使用 `--env-rotate-axis z`。

相机环绕是独立模式：环境变换保持单位阵，固定某一轨迹帧的全部物体，相机保持半径与高度差不变并绕 Isaac world Z 轴旋转 360°。环境旋转与相机环绕不会在同一次任务中同时启用。

程序化单光源由 `make_single_source_env.py` 生成：均匀环境填充加一个有限角尺寸的 Gaussian 高亮区域。早期测试参数为：

```text
HDR: 1024 × 512
颜色: [1.0, 0.86, 0.68]
仰角: 35°
光斑 FWHM: 6°
峰值辐亮度: 35
环境填充: 0.025
```

它表现为单个摄影棚面积光绕场景移动，但没有真实局部点光源的距离平方衰减。

### 8.1 当前采用的 relighting 配置（2026-08-23）

当前默认运行配置（轨迹组装与重光照）为：

```text
GS 代理：icosphere320（每个 Gaussian 使用 320 面代理）
漫反射采样 DS：32
环境重要性采样 LS：32
light_t_min：0.10
环境光：fill=0.35，key=2500
BVH 布局：全刚体 IAS（静态对象/各 link/刚体对象分别建 GAS，逐帧更新实例矩阵）
```

除非命令行显式覆盖，否则以上参数视为默认值；旧实验中的 256/128 采样配置不再代表默认设置。

当前 `guiji1/guiji2` 轨迹视频采用以下配置：

```text
环境贴图: assets/env_map/pointlike_camera_key_light_fill035_key2500.exr
分辨率: 1024 × 512（lat-long EXR）
光源颜色: [1.0, 1.0, 1.0]（中性白）
光源方向: elevation=-38°, azimuth=131°
光斑 FWHM: 1.5°
主光峰值 key: 2500
均匀环境填充 fill: 0.35

diffuse_samples: 256
light_samples: 128
light_t_min: 0.10（4 倍场景单位）
base_color_min: 0.03
输出: 960 × 540
```

每个参与着色的 Gaussian 使用 256 个法线上半球 Fibonacci 样本和 128 个环境贴图重要性样本。后者按环境亮度与 lat-long 球面面积权重抽样，对 1.5° 的高亮光斑非常重要。`light_samples=0` 无法稳定捕获该尖锐光源，会产生严重亮点、黑点和缺光；128 是当前速度与噪声的折中。高质量单帧可使用 `diffuse_samples=512, light_samples=256`。

`light_t_min=0.10` 是当前自遮挡与漏光之间的折中：`0.05` 接触阴影更强但局部黑点更多，`0.15` 黑点较少但漏光和阴影减弱更明显。

### 8.2 当前环境旋转模式

固定轨迹帧模式：物体和相机不动，环境沿 world Z 旋转 360°。当前使用 72 帧、18 FPS、每帧 5°，视频长 4 秒。

动态轨迹模式：物体按 Isaac 轨迹运动，同时环境以固定角速度连续旋转。使用：

```text
--trajectory-env-deg-per-sec 90 --env-rotate-axis z
```

在 30 FPS 下每帧旋转 3°。完整约 12.5 秒轨迹会旋转约 1119°（约 3.1 圈），不要求首尾环境方向闭合。`--trajectory-env-rotations N` 仍保留用于必须在整段轨迹中恰好旋转 N 圈的情况。

## 9. 杯子最终 USD 对齐

### 9.1 最终资产

`guiji1` 已提供实际渲染使用的最终资产：

```text
assets/cup/cup_canonical_80mm_kraft_black_lid_converted.usd
SHA256 = 7541aa0a1a8c03902825d0a0e087f9e4336ba3da07be0c26d19606e695201ce0
```

USD 可见表面的最终范围为 52.788 × 52.974 × 80.000 mm，杯底位于 canonical Z=0，杯盖位于 +Z。

```text
[0.052788, 0.052974, 0.080000] m
```

### 9.2 源 3DGS 到最终杯子局部系

最终流程不再使用旧版 AABB 轴交换或逐轴非均匀缩放。渲染器直接读取：

```text
dataset/guiji1/provenance/complete_coordinate_transforms.json
cup.T_v32_visual_local_from_source
```

列向量约定下的精确矩阵为：

```text
[-0.162297863773,  0.120307361044, -0.151638543871, 0.085704670877]
[ 0.193389473967,  0.109249054600, -0.120307361044, 0.110538035694]
[ 0.008283747166, -0.193389473967, -0.162297863773, 0.125459895688]
[ 0,                 0,                 0,                1]
```

该矩阵包含 PCA 杯盖朝上、统一缩放到 80 mm、XY 居中、杯底对齐 canonical Z=0，以及 V32 visual local 的 `[0,0,-0.04] m` 平移。轨迹首帧杯子刚体根节点约为 `z=0.04 m`，因此名义杯底落在世界 Z=0。

杯子训练源 GLB SHA256 为：

```text
800233b0d18f54db2faf1703cd866a1cffd69870fe6993d25f7f51a6ee5c701c
```

通过 `cup_original_to_split_vertex_mapping.csv` 对全部 33,905 个拆分后顶点复核，源 GLB 经上述矩阵到最终 CupBody/CupLid 的最大误差为 `3.8345e-9 m`。增加的 608 个顶点只来自杯身/杯盖边界共享顶点复制，66,031 个三角形保持不变。

一般 affine 变换会同步作用于每个 2D Gaussian 的 tangent basis，代码使用 SVD 重分解两轴 scale 与 rotation，并使用 inverse-transpose 维护法线方向和右手性。六个阶段边界关键帧以及完整视频均已验证杯盖朝上、夹爪闭合关系连续和抬升阶段逐帧跟随。

### 9.3 接触面显示限制

mesh/刚体名义接触正确不意味着两个独立训练的 3DGS 在接触处具有零厚度。首帧数值检查结果：

```text
杯子最低 Gaussian 中心：约 +0.225 mm
杯子 Gaussian 向下支撑：最低约 -0.769 mm
杯子下方桌面 Gaussian 向上支撑：最高约 +3.680 mm
```

因此杯底与桌面 Gaussian 的透明椭球支撑域存在毫米级重叠，alpha 混合后可能看起来略微嵌入。所有物体先合并为一个 Gaussian 集合并由同一次 `render_ir()` 统一深度排序，所以这不是桌子/杯子的拼接先后顺序造成的。`light_t_min=0.15` 在 4 倍场景中对应现实约 3.75 cm，可能削弱近距离接触阴影，但不会改变主光栅化几何位置。

## 10. 可复现命令

环境：

```bash
cd /amax/home/fengshuangyu/relighting/IRGS
export CUDA_VISIBLE_DEVICES=5
export PATH="/amax/home/fengshuangyu/miniconda3/envs/irgs/bin:$PATH"
export LIBRARY_PATH="/amax/home/fengshuangyu/miniconda3/envs/irgs/lib:/amax/home/fengshuangyu/miniconda3/envs/irgs/lib64:/amax/home/fengshuangyu/miniconda3/envs/irgs/lib/stubs:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/amax/home/fengshuangyu/miniconda3/envs/irgs/lib:/amax/home/fengshuangyu/miniconda3/envs/irgs/lib64:${LD_LIBRARY_PATH:-}"
export OPEN3D_CPU_RENDERING=true
export OPENCV_IO_ENABLE_OPENEXR=1
```

主相机完整轨迹（精确杯子变换）：

```bash
python scripts/render_guiji_irgs.py --scene full --full-video --stride 4 \
  --camera-name main \
  --width 960 --height 540 --fovy-deg 27 --fps 30 \
  --diffuse-samples 64 --light-t-min 0.15 \
  --envmap assets/env_map/envmap3.exr \
  --out outputs/guiji1_irgs_full_exact_cup
```

固定轨迹中间状态 `step=748` 与主相机，旋转 `envmap3` 360°：

```bash
python scripts/render_guiji_irgs.py --scene full --steps 748 \
  --camera-name main --env-rotate-count 120 --env-rotate-axis y \
  --width 960 --height 540 --fovy-deg 27 --fps 30 \
  --diffuse-samples 64 --light-t-min 0.15 \
  --envmap assets/env_map/envmap3.exr \
  --out outputs/guiji1_middle_envrotate_exact
```

固定相同中间状态与 `envmap3`，主相机绕世界 Z 轴旋转 360°：

```bash
python scripts/render_guiji_irgs.py --scene full --steps 748 \
  --camera-name main --camera-orbit-count 120 \
  --width 960 --height 540 --fovy-deg 27 --fps 30 \
  --diffuse-samples 64 --light-t-min 0.15 \
  --envmap assets/env_map/envmap3.exr \
  --out outputs/guiji1_middle_camera_orbit_exact
```

第二相机完整轨迹：

```bash
python scripts/render_guiji_irgs.py --scene full --full-video --stride 4 \
  --camera-name contact \
  --width 960 --height 540 --fovy-deg 27 --fps 30 \
  --diffuse-samples 64 --light-t-min 0.15 \
  --envmap assets/env_map/envmap3.exr \
  --out outputs/guiji1_contact_camera_full_exact
```

生成当前采用的白色窄光源（fill 0.35 / key 2500）：

```bash
python scripts/make_single_source_env.py \
  --out assets/env_map/pointlike_camera_key_light_fill035_key2500.exr \
  --height 512 --width 1024 \
  --elevation-deg=-38 --azimuth-deg 131 --radius-deg 1.5 \
  --intensity 2500 --ambient 0.35 --color 1.0,1.0,1.0
```

`guiji1` 固定中间帧，物体/相机不动、环境旋转一周：

```bash
python scripts/render_guiji_irgs.py --trajectory-set guiji1 --scene full --steps 748 \
  --env-rotate-count 72 --env-rotate-axis z \
  --camera-name main \
  --width 960 --height 540 --fovy-deg 27 --fps 18 \
  --diffuse-samples 256 --light-samples 128 --light-t-min 0.10 \
  --base-color-min 0.03 --table-z-offset-m 0 \
  --envmap assets/env_map/pointlike_camera_key_light_fill035_key2500.exr \
  --out outputs/guiji1_step748_fill035_key2500_ds256_ls128_t010_envrotate_z360
```

`guiji2` 完整动态轨迹，同时让环境以 90°/秒持续旋转：

```bash
python scripts/render_guiji_irgs.py --trajectory-set guiji2 --scene full \
  --full-video --stride 4 --fps 30 \
  --trajectory-env-deg-per-sec 90 --env-rotate-axis z \
  --camera-name main --width 960 --height 540 --fovy-deg 27 \
  --diffuse-samples 256 --light-samples 128 --light-t-min 0.10 \
  --base-color-min 0.03 --table-z-offset-m -0.006 \
  --envmap assets/env_map/pointlike_camera_key_light_fill035_key2500.exr \
  --out outputs/guiji2_full_trajectory_fill035_key2500_ds256_ls128_t010_envrotate90dps
```

### 10.1 有限点光源与矩形面光源

`render_guiji_irgs.py` 现在支持三种互斥光源：

- `--light-type env`：原有 EXR/HDR 无限远环境光，默认模式，原命令保持兼容。
- `--light-type point`：有三维位置、平方反比衰减和硬阴影的有限点光源。
- `--light-type area`：有限矩形面光源，在发光面均匀采样，包含面积、发光面余弦和平方反比衰减，产生软阴影。

点光源和面光源的位置、目标与尺寸使用 Isaac 世界米，脚本会自动转换到内部 `SCENE_SCALE=4` 坐标。颜色和强度均在线性 RGB 中计算。点光源的强度是 RGB radiant intensity；矩形面光源的强度是恒定 emitted radiance。因此两种模式的强度数值不能直接横向等同。默认值分别是 `0.5` 和 `5.0`。

点光源单帧：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/render_guiji_irgs.py \
  --trajectory-set guiji1 --scene full --steps 748 \
  --camera-name main --width 960 --height 540 --fovy-deg 27 \
  --light-type point \
  --point-light-position-m 0.30 -0.20 0.60 \
  --light-color 1.0 0.95 0.90 --light-intensity 0.5 \
  --light-t-min 0.10 \
  --out outputs/guiji1_step748_point_light
```

矩形面光源单帧：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/render_guiji_irgs.py \
  --trajectory-set guiji1 --scene full --steps 748 \
  --camera-name main --width 960 --height 540 --fovy-deg 27 \
  --light-type area \
  --area-light-center-m 0.30 -0.20 0.60 \
  --area-light-target-m 0.00 0.00 0.10 \
  --area-light-up 0 1 0 --area-light-size-m 0.30 0.20 \
  --light-color 1.0 0.95 0.90 --light-intensity 5.0 \
  --analytic-light-samples 64 --light-t-min 0.10 \
  --out outputs/guiji1_step748_area_light
```

在点光源或面光源模式下使用原有 `--env-rotate-count` 或轨迹光源旋转参数，会让解析光源绕世界原点旋转，而不是旋转 EXR。解析光源当前实现直接光照与有限距离阴影，不伪造环境贴图间接光；未命中 GS 的背景仍由渲染器 `bg_color` 决定，当前轨迹脚本为白色。

## 11. 主要输出

```text
主相机完整轨迹（375 帧，30 FPS，12.5 秒）:
IRGS/outputs/guiji1_irgs_full_exact_cup/so101_newbase_cup_trajectory_irgs.mp4
IRGS/outputs/guiji1_irgs_full_exact_cup/video_report.json

中间帧 envmap3 旋转（120 帧，30 FPS，4 秒）:
IRGS/outputs/guiji1_middle_envrotate_exact/step_0748_envrotate_irgs.mp4
IRGS/outputs/guiji1_middle_envrotate_exact/envrotate_report.json

中间帧主相机环绕（120 帧，30 FPS，4 秒）:
IRGS/outputs/guiji1_middle_camera_orbit_exact/step_0748_camera_orbit_irgs.mp4
IRGS/outputs/guiji1_middle_camera_orbit_exact/camera_orbit_report.json

第二相机完整轨迹（375 帧，30 FPS，12.5 秒）:
IRGS/outputs/guiji1_contact_camera_full_exact/so101_newbase_cup_trajectory_contact_camera_irgs.mp4
IRGS/outputs/guiji1_contact_camera_full_exact/video_report.json

单光源:
IRGS/assets/env_map/pointlike_camera_key_light_fill035_key2500.exr
IRGS/outputs/guiji_irgs_singlelight_step0748/step_0748_envrotate_irgs.mp4
IRGS/outputs/guiji_irgs_singlelight_step0748/envrotate_report.json
```

## 12. 结论

当前流程能够用同一 URDF 驱动 7 个 link 3DGS，使用 Isaac 实际关节角与杯子位姿，将机器人、桌子和杯子放入同一个 4 倍 Isaac world，每帧更新全局 BVH，并使用 IRGS 完成动态重光照、固定视角环境旋转、固定环境相机环绕、主相机轨迹和第二相机轨迹。

newbase 坐标已通过 59,846 个对应顶点数值闭环。机器人使用 URDF 加 `guiji1` 合同 root correction。杯子使用 V32 资料包精确矩阵并通过 33,905 个映射顶点验证，最大误差 `3.8345e-9 m`。最终三类新增视频的帧数、帧率、分辨率、时长和报告模式均已使用 `ffprobe` 与 JSON 回读验证。剩余可见限制主要来自独立训练 3DGS 在接触面的有限椭球厚度，而不是坐标链或物体拼接顺序。
