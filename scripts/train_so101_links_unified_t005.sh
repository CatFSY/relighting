#!/bin/bash
# Train all SO101 link IRGS models with the same Stage2 configuration used by
# the Synthetic4Relight table and Lego assets.  Six links reuse their finished
# RefGaussian-50k checkpoints.  gripper_link resumes RefGaussian 40k -> 50k
# first, then automatically starts its unified Stage2 job.
set -euo pipefail

source /amax/home/fengshuangyu/miniconda3/etc/profile.d/conda.sh
set +u
conda activate irgs
set -u

cd /amax/home/fengshuangyu/relighting/IRGS
export OPEN3D_CPU_RENDERING=true
export OPENCV_IO_ENABLE_OPENEXR=1
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$CONDA_PREFIX/lib/stubs:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LD_LIBRARY_PATH:-}"

LINKS=(
  base_link
  shoulder_link
  upper_arm_link
  lower_arm_link
  wrist_link
  gripper_link
  moving_jaw_so101_v1_link
)
GPUS=(0 1 2 3 4 5 6)
PIDS=()

train_stage2() {
  local link="$1"
  local data="dataset/so101_links/$link"
  local model="outputs/so101_links/$link"
  local checkpoint="$model/refgs_full/chkpnt50000.pth"

  test -f "$checkpoint"
  mkdir -p "$model/irgs_full"
  python train.py -s "$data" -m "$model/irgs_full" \
    --eval --iterations 20000 \
    --start_checkpoint_refgs "$checkpoint" \
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
}

for index in "${!LINKS[@]}"; do
  link="${LINKS[$index]}"
  gpu="${GPUS[$index]}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    data="dataset/so101_links/$link"
    model="outputs/so101_links/$link"
    mkdir -p "$model"

    if [ "$link" = "gripper_link" ] && [ ! -f "$model/refgs_full/chkpnt50000.pth" ]; then
      echo "[$(date -Is)] $link GPU=$gpu: resume Stage1 40000 -> 50000"
      python train_refgaussian.py -s "$data" -m "$model/refgs_full" \
        --eval -w --iterations 50000 \
        --start_checkpoint "$model/refgs_full/chkpnt40000.pth" \
        --mask_dir "$data/masks" \
        --lambda_mask_entropy 0.05 \
        --prune_opacity_threshold 0.005 \
        --lambda_opacity_reg 0
      echo "[$(date -Is)] $link GPU=$gpu: Stage1 complete"
    fi

    echo "[$(date -Is)] $link GPU=$gpu: unified Stage2 start"
    train_stage2 "$link"
    echo "[$(date -Is)] $link GPU=$gpu: unified Stage2 complete"
  ) > "outputs/so101_links/$link/unified_t005_train.log" 2>&1 &
  PIDS+=("$!")
  echo "$link GPU=$gpu PID=${PIDS[-1]}"
done

status=0
for index in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$index]}"; then
    echo "FAILED: ${LINKS[$index]} PID=${PIDS[$index]}" >&2
    status=1
  fi
done

if [ "$status" -ne 0 ]; then
  exit "$status"
fi
echo "[$(date -Is)] ALL SO101 LINKS UNIFIED STAGE2 COMPLETE"
