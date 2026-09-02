# guiji3 组装时 PLY 阶段选择错误

## 问题概述

在 `guiji3` 的桌子+机械臂+杯子组装视频中，杯子出现严重的彩色/彩虹伪影。单独渲染
`beizi2` 的 IRGS 模型时，多视角均正常，因此问题不在杯子训练结果或相机轨迹。

## 根因

调用组装脚本时传入了一个同时包含两个阶段的输出目录：

```text
outputs/input_video_frames_beizi2/
├── irgs/point_cloud/iteration_20000/point_cloud.ply
└── refgs/point_cloud/iteration_50000/point_cloud.ply
```

旧版 `resolve_ply_path()` 会递归搜集所有 `point_cloud.ply`，再按迭代号选择最大值。因此它错误地选择了
`refgs/iteration_50000`，而不是用于重光照的 `irgs/iteration_20000`。

两者的 GS 数量都为 37,825，单看数量无法发现问题；但材质 logits 不同：

| 模型 | base color 均值（raw） | roughness 均值（raw） | metallic 均值（raw） |
|---|---:|---:|---:|
| 正确 `irgs/20000` | `[-4.460, -4.644, -4.697]` | `-0.0113` | `-1.3863` |
| 错误 `refgs/50000` | `[-1.172, -1.167, -0.943]` | `-1.0110` | `-3.8249` |

这些材质字段被送入 IRGS 的 BRDF/间接光计算后，造成组装杯子外观异常。

## 修复

`IRGS/scripts/render_guiji_irgs.py` 的目录解析现在按以下顺序选择：

1. `irgs/point_cloud/iteration_*/point_cloud.ply`
2. `point_cloud/iteration_*/point_cloud.ply`
3. `refgs/point_cloud/iteration_*/point_cloud.ply`
4. 递归搜索作为最后回退

同时，渲染命令应尽量显式传入阶段目录或具体 PLY，例如：

```bash
--object-ply /amax/home/fengshuangyu/relighting/IRGS/outputs/input_video_frames_beizi2/irgs
--table-ply  /amax/home/fengshuangyu/relighting/IRGS/outputs/input_video_frames_basev8/irgs
```

## 验证

修复后目录解析结果为：

```text
/amax/home/fengshuangyu/relighting/IRGS/outputs/input_video_frames_beizi2/irgs/point_cloud/iteration_20000/point_cloud.ply
```

单杯的组装测试帧不再出现彩色伪影；随后使用 basev8 桌子、beizi2 杯子和 SO101 link 重新生成了完整
`guiji3` 轨迹视频：

```text
outputs/guiji3_corrected_basev8_beizi2_full_960x540_default/
└── so101_table_cup_trajectory_irgs.mp4
```

视频共 384 帧，分辨率 960×540。检查首帧、中间帧和末帧，杯子外观正常，轨迹运动正常。

## 预防措施

- `refgs` 是第一阶段几何模型，`irgs` 是用于材质和重光照的模型；组装重光照时必须使用 `irgs`。
- 报告中应记录每个组件最终解析到的 PLY 完整路径，而不只记录 GS 数量。
- 不要仅根据迭代号在 `irgs` 与 `refgs` 之间选择模型。
