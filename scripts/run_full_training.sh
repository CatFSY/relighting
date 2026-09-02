#!/bin/bash
# 全量训练: 7 个 SO101 link × (4a Ref-GS 50000 → 4b IRGS 20000, 官方 run.sh 配置), 每 link 一卡并行
# 无伪监督: 不传 --pred_material_dir / --normal_supervision, lambda_albedo/roughness/metallic/normal_gt 全 0
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate irgs
export OPEN3D_CPU_RENDERING=true OPENCV_IO_ENABLE_OPENEXR=1

LINKS=(base_link shoulder_link upper_arm_link lower_arm_link wrist_link gripper_link moving_jaw_so101_v1_link)
GPUS=(0 1 2 4 5 6 7)

cd /amax/home/fengshuangyu/relighting/IRGS
for i in "${!LINKS[@]}"; do
  link="${LINKS[$i]}"
  gpu="${GPUS[$i]}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    D=dataset/so101_links/$link
    M=outputs/so101_links/$link
    mkdir -p $M
    echo "== $link on GPU $gpu: 4a start $(date +%H:%M:%S)"
    python train_refgaussian.py -s $D -m $M/refgs_full --eval -w \
      --mask_dir $D/masks --lambda_mask_entropy 0.05 \
      --prune_opacity_threshold 0.005 --lambda_opacity_reg 0 --iterations 50000 \
      > $M/refgs_full_train.log 2>&1
    echo "== $link: 4a done $(date +%H:%M:%S), 4b start"
    python train.py -s $D -m $M/irgs_full --eval --iterations 20000 \
      --start_checkpoint_refgs $M/refgs_full/chkpnt50000.pth \
      --mask_dir $D/masks --envmap_resolution 128 \
      --lambda_base_color_smooth 2 --lambda_roughness_smooth 2 \
      --diffuse_sample_num 256 --envmap_cubemap_lr 0.01 \
      --lambda_light_smooth 0.0005 --init_roughness_value 0.6 \
      --lambda_light 0.1 --train_ray \
      --lambda_albedo 0 --lambda_roughness 0 --lambda_metallic 0 \
      --base_color_min 0.0 --lambda_normal_gt 0 --lambda_opacity_reg 0 \
      > $M/irgs_full_train.log 2>&1
    echo "== $link: all done $(date +%H:%M:%S)"
  ) &
done
wait
echo "ALL LINKS TRAINED $(date +%H:%M:%S)"
