#!/bin/bash
# 4b-only: 7 link 并行, 各 20000 iter (官方 run.sh 配置), 从 4a checkpoint 恢复
# base/shoulder/upper_arm/gripper 用 bridge chkpnt40000 (4a 跑到 40000 被杀, ply 全状态)
# moving_jaw/lower_arm/wrist 用完整 chkpnt50000
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate irgs
export OPEN3D_CPU_RENDERING=true OPENCV_IO_ENABLE_OPENEXR=1
cd /amax/home/fengshuangyu/relighting/IRGS
declare -A CKPT=(
  [base_link]=chkpnt40000.pth [shoulder_link]=chkpnt40000.pth
  [upper_arm_link]=chkpnt40000.pth [gripper_link]=chkpnt40000.pth
  [moving_jaw_so101_v1_link]=chkpnt50000.pth [lower_arm_link]=chkpnt50000.pth
  [wrist_link]=chkpnt50000.pth)
LINKS=(base_link shoulder_link upper_arm_link lower_arm_link wrist_link gripper_link moving_jaw_so101_v1_link)
GPUS=(0 1 2 4 5 6 7)
for i in "${!LINKS[@]}"; do
  link="${LINKS[$i]}"; gpu="${GPUS[$i]}"
  ( export CUDA_VISIBLE_DEVICES="$gpu"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    D=dataset/so101_links/$link; M=outputs/so101_links/$link
    mkdir -p $M
    python train.py -s $D -m $M/irgs_full --eval --iterations 20000 \
      --start_checkpoint_refgs $M/refgs_full/${CKPT[$link]} \
      --mask_dir $D/masks --envmap_resolution 128 \
      --lambda_base_color_smooth 2 --lambda_roughness_smooth 2 \
      --diffuse_sample_num 256 --envmap_cubemap_lr 0.01 \
      --lambda_light_smooth 0.0005 --init_roughness_value 0.6 \
      --lambda_light 0.1 --train_ray \
      --lambda_albedo 0 --lambda_roughness 0 --lambda_metallic 0 \
      --base_color_min 0.0 --lambda_normal_gt 0 --lambda_opacity_reg 0 \
      > $M/irgs_full_train.log 2>&1
    echo "== $link: 4b done $(date +%H:%M:%S)" ) &
done
wait
echo "ALL IRGS DONE $(date +%H:%M:%S)"
