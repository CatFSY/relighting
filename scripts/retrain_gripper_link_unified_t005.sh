#!/bin/bash
# Clean recovery path for gripper_link. Its legacy 40k checkpoint was rebuilt
# from a PLY and cannot resume because optimizer/env-map states are absent.
set -euo pipefail

source /amax/home/fengshuangyu/miniconda3/etc/profile.d/conda.sh
set +u
conda activate irgs
set -u

cd /amax/home/fengshuangyu/relighting/IRGS
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export OPEN3D_CPU_RENDERING=true
export OPENCV_IO_ENABLE_OPENEXR=1
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$CONDA_PREFIX/lib/stubs:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LD_LIBRARY_PATH:-}"

data="dataset/so101_links/gripper_link"
root="outputs/so101_links/gripper_link"
refgs="$root/refgs_unified_t005"
irgs="$root/irgs_full"

echo "[$(date -Is)] gripper_link: clean Stage1 start"
python train_refgaussian.py -s "$data" -m "$refgs" \
  --eval -w --iterations 50000 \
  --mask_dir "$data/masks" \
  --lambda_mask_entropy 0.05 \
  --prune_opacity_threshold 0.005 \
  --lambda_opacity_reg 0 \
  --light_t_min 0.05

test -f "$refgs/chkpnt50000.pth"
echo "[$(date -Is)] gripper_link: clean Stage1 complete; unified Stage2 start"
mkdir -p "$irgs"
python train.py -s "$data" -m "$irgs" \
  --eval --iterations 20000 \
  --start_checkpoint_refgs "$refgs/chkpnt50000.pth" \
  --mask_dir "$data/masks" \
  --envmap_resolution 128 \
  --lambda_base_color_smooth 2 \
  --lambda_roughness_smooth 2 \
  --diffuse_sample_num 256 \
  --envmap_cubemap_lr 0.01 \
  --lambda_light_smooth 0.0005 \
  --init_roughness_value 0.6 \
  --lambda_light 0.01 \
  --train_ray \
  --light_t_min 0.05 \
  --base_color_min 0.03 \
  --lambda_albedo 0 \
  --lambda_roughness 0 \
  --lambda_metallic 0 \
  --lambda_normal_gt 0 \
  --lambda_opacity_reg 0

echo "[$(date -Is)] gripper_link: unified Stage2 complete"
