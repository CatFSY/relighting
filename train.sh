#!/usr/bin/env bash
# Train one local IRGS scene: RefGS geometry (stage 1) followed by IRGS
# material/illumination decomposition (stage 2).
#
# =========================== 常用命令 ===========================
# 1. 默认训练 basev8（使用 GPU 0；已有 checkpoint 的阶段会自动跳过）
#    bash train.sh basev8
#
# 2. 指定 GPU 运行
#    CUDA_VISIBLE_DEVICES=3 bash train.sh basev8
#
# 3. 强制从头重训两个阶段（会删除该场景的 refgs/ 和 irgs/ 输出）
#    CUDA_VISIBLE_DEVICES=3 FORCE_RETRAIN=1 bash train.sh basev8
#
# 4. 跳过第一阶段，复用已有 refgs/chkpnt50000.pth
#    RUN_REFGS=0 bash train.sh basev8
#
# 5. 临时缩短训练用于排错（例如第一阶段 500 步、第二阶段 200 步）
#    REFGS_ITERS=500 IRGS_ITERS=200 bash train.sh basev8
#
# 6. 指定自定义数据集和输出目录
#    SCENE_DIR=/path/to/dataset OUTPUT_DIR=/path/to/output bash train.sh my_scene
#
# 注意：本脚本只执行 RefGS + IRGS 训练，不执行 mesh 提取或重光照视频。
# ================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_NAME="${1:-}"
if [[ -z "$SCENE_NAME" || "$SCENE_NAME" == -* ]]; then
    echo "Usage: bash train.sh <scene-name>" >&2
    echo "Example: bash train.sh basev8" >&2
    exit 2
fi

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
else
    CONDA_BASE="${CONDA_BASE:-/amax/home/fengshuangyu/miniconda3}"
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${IRGS_CONDA_ENV:-irgs}"
cd "$PROJECT_DIR"

export OPEN3D_CPU_RENDERING=true
export OPENCV_IO_ENABLE_OPENEXR=1
export OPENCV_IO_MAX_IMAGE_PIXELS=17179869184
export LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64:${CONDA_PREFIX}/lib/stubs:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCENE="${SCENE_DIR:-$PROJECT_DIR/dataset/input_video_frames_${SCENE_NAME}}"
OUT="${OUTPUT_DIR:-$PROJECT_DIR/outputs/input_video_frames_${SCENE_NAME}}"
PY="${PYTHON:-python}"
REFGS_ITERS="${REFGS_ITERS:-50000}"
IRGS_ITERS="${IRGS_ITERS:-20000}"
DIST_WEIGHT="${REFGS_DIST_WEIGHT:-1000}"
DIST_START="${REFGS_DIST_LOSS_START:-3000}"
OPACITY_REG="${OPACITY_REG:-0}"
RUN_REFGS="${RUN_REFGS:-1}"   # 0=跳过第一阶段，复用已有 RefGS checkpoint

# 每次训练单独保存一份完整日志；日志不放在 refgs/irgs 子目录中，
# 因此 FORCE_RETRAIN=1 清理 checkpoint 时不会删除历史记录。
mkdir -p "$OUT"
LOG_FILE="${TRAIN_LOG:-$OUT/train_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log: $LOG_FILE"

[[ -d "$SCENE/images" ]] || { echo "Missing dataset images: $SCENE" >&2; exit 1; }
[[ -f "$SCENE/sparse/0/cameras.bin" && -f "$SCENE/sparse/0/images.bin" ]] || {
    echo "Missing COLMAP sparse model under $SCENE/sparse/0" >&2; exit 1;
}

# Object masks are scene-specific. Prefer the known table name, then the scene
# name, then the first available mask directory.
if [[ -d "$SCENE/masks/table" ]]; then
    MASK_DIR="$SCENE/masks/table"
elif [[ -d "$SCENE/masks/$SCENE_NAME" ]]; then
    MASK_DIR="$SCENE/masks/$SCENE_NAME"
else
    MASK_DIR="$(find "$SCENE/masks" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | head -1 || true)"
fi
[[ -n "$MASK_DIR" && -d "$MASK_DIR" ]] || { echo "No mask directory under $SCENE/masks" >&2; exit 1; }

REFGS_CKPT="$OUT/refgs/chkpnt${REFGS_ITERS}.pth"
IRGS_CKPT="$OUT/irgs/chkpnt${IRGS_ITERS}.pth"

echo "Scene: $SCENE_NAME"
echo "Dataset: $SCENE"
echo "Output: $OUT"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Mask: $MASK_DIR"

if [[ "$RUN_REFGS" != "0" && "$RUN_REFGS" != "1" ]]; then
    echo "RUN_REFGS must be 0 or 1" >&2
    exit 2
fi
if [[ "${FORCE_RETRAIN:-0}" == "1" && "$RUN_REFGS" == "0" ]]; then
    echo "FORCE_RETRAIN=1 与 RUN_REFGS=0 冲突：不能删除后跳过第一阶段" >&2
    exit 2
fi
if [[ "${FORCE_RETRAIN:-0}" == "1" ]]; then
    rm -rf "$OUT/refgs" "$OUT/irgs"
fi

if [[ "$RUN_REFGS" == "0" ]]; then
    echo "[1/2] 跳过 RefGS，复用: $REFGS_CKPT"
elif [[ -f "$REFGS_CKPT" && "${FORCE_RETRAIN:-0}" != "1" ]]; then
    echo "[1/2] RefGS checkpoint exists, skip: $REFGS_CKPT"
else
    echo "[1/2] Training RefGS geometry (${REFGS_ITERS} iterations)"
    "$PY" -u train_refgaussian.py \
        -s "$SCENE" -m "$OUT/refgs" --eval -w \
        --iterations "$REFGS_ITERS" \
        --mask_dir "$MASK_DIR" \
        --pred_material_dir "$SCENE" \
        --lambda_mask_entropy 0.05 \
        --prune_opacity_threshold 0.005 \
        --lambda_dist "$DIST_WEIGHT" \
        --dist_loss_start "$DIST_START" \
        --lambda_opacity_reg "$OPACITY_REG" \
        --normal_supervision --lambda_normal_gt 0.1
fi
[[ -f "$REFGS_CKPT" ]] || { echo "RefGS failed: $REFGS_CKPT" >&2; exit 1; }

if [[ -f "$IRGS_CKPT" && "${FORCE_RETRAIN:-0}" != "1" ]]; then
    echo "[2/2] IRGS checkpoint exists, skip: $IRGS_CKPT"
else
    echo "[2/2] Training IRGS materials/lighting (${IRGS_ITERS} iterations)"
    "$PY" -u train.py \
        -s "$SCENE" -m "$OUT/irgs" --eval \
        --iterations "$IRGS_ITERS" \
        --start_checkpoint_refgs "$REFGS_CKPT" \
        --mask_dir "$MASK_DIR" --pred_material_dir "$SCENE" \
        --envmap_resolution 128 \
        --lambda_base_color_smooth 2 --lambda_roughness_smooth 2 \
        --diffuse_sample_num 32 \
        --envmap_cubemap_lr 0.01 --lambda_light_smooth 0.0005 \
        --init_roughness_value 0.6 --lambda_light 0.1 --train_ray \
        --lambda_albedo 1.0 --albedo_loss_space srgb \
        --lambda_roughness 0.5 --base_color_min 0.0 \
        --lambda_normal_gt 0.1 --lambda_opacity_reg "$OPACITY_REG"
fi
[[ -f "$IRGS_CKPT" ]] || { echo "IRGS failed: $IRGS_CKPT" >&2; exit 1; }
echo "Training complete: $OUT"
