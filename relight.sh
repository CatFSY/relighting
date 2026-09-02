#!/usr/bin/env bash
# 统一重光照入口：单体模型或多物体轨迹组装。
#
# =========================== 常用命令 ===========================
# 1. 单个物体，固定相机、env3 旋转
#    bash relight.sh single basev8 env3 --pose-mode fixed --env-rotate
#
# 2. 单个物体，位姿插值 + env3 同时旋转
#    bash relight.sh single basev8 env3 --pose-mode interpolate --env-rotate
#
# 3. 单个物体，固定相机和固定环境（不加 --env-rotate）
#    bash relight.sh single basev8 envmap3.exr --pose-mode fixed
#
# 4. 机械臂/桌子/Lego 组合，完整 guiji2 轨迹，环境固定
#    bash relight.sh assembled guiji2
#
# 5. 组合轨迹 + 环境持续旋转（90 度/秒）
#    bash relight.sh assembled guiji2 --trajectory-env-deg-per-sec 90
#
# 6. 指定 GPU
#    CUDA_VISIBLE_DEVICES=3 bash relight.sh assembled guiji1
#
# 7. 自定义轨迹 + 自定义桌子/物体
#    轨迹目录应包含：trajectories/*.csv、configs/execution_contract.json、
#    provenance/complete_coordinate_transforms.json。
#    桌子和物体参数既可以写 PLY 文件，也可以写 IRGS output 文件夹，
#    脚本会自动寻找其中 iteration_* 最高的 point_cloud.ply。
#    CUDA_VISIBLE_DEVICES=1 bash relight.sh assembled \
#      /path/to/new_guiji \
#      --table-ply /path/to/table_output \
#      --object-ply /path/to/object_output \
#      --object-name cup \
#      --output-dir /path/to/result
#
#    例如：
#    CUDA_VISIBLE_DEVICES=1 bash relight.sh assembled \
#      /amax/home/fengshuangyu/relighting/IRGS/dataset/guiji_new \
#      --table-ply /amax/home/fengshuangyu/relighting/IRGS/outputs/new_table \
#      --object-ply /amax/home/fengshuangyu/relighting/IRGS/outputs/new_object \
#      --object-name lego \
#      --output-dir /amax/home/fengshuangyu/relighting/IRGS/outputs/guiji_new_relight
#    如果 execution_contract.json 已声明 table_irgs_output/object_irgs_output，
#    则可省略 --table-ply、--object-ply、--object-name，脚本会自动推断。
#
# 环境名可写 env3、env6、env12，也可直接写 EXR 的路径。
# 单体默认模型：outputs/input_video_frames_<场景名>/irgs_stage2_with_pseudo
# 组合默认输出：outputs/<场景名>_relight
# =================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-}"
SCENE_NAME="${2:-}"

usage() {
    sed -n '1,31p' "$0"
    echo
    echo "用法："
    echo "  bash relight.sh single <场景名> <env3|env6|env12|EXR路径> [额外参数...]"
    echo "  bash relight.sh assembled <guiji1|guiji2|轨迹目录> [额外参数...]"
}

if [[ "$MODE" != "single" && "$MODE" != "assembled" ]] || [[ -z "$SCENE_NAME" ]]; then
    usage >&2
    exit 2
fi

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
else
    # 非交互/后台 shell 可能没有初始化 conda，使用本机默认安装位置。
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
PY="${PYTHON:-python}"

if [[ "$MODE" == "single" ]]; then
    if [[ $# -ge 3 ]]; then
        ENV_NAME="$3"
        shift 3
    else
        ENV_NAME="env3"
        shift 2
    fi
    case "$ENV_NAME" in
        env3)  ENV_PATH="$PROJECT_DIR/assets/env_map/envmap3.exr" ;;
        env6)  ENV_PATH="$PROJECT_DIR/assets/env_map/envmap6.exr" ;;
        env12) ENV_PATH="$PROJECT_DIR/assets/env_map/envmap12.exr" ;;
        *)     ENV_PATH="$ENV_NAME" ;;
    esac
    MODEL="${MODEL_DIR:-$PROJECT_DIR/outputs/input_video_frames_${SCENE_NAME}/irgs_stage2_with_pseudo}"
    OUTPUT="${OUTPUT_MP4:-$PROJECT_DIR/outputs/input_video_frames_${SCENE_NAME}/relight_${ENV_NAME}.mp4}"
    mkdir -p "$(dirname "$OUTPUT")"
    LOG_FILE="${RELIGHT_LOG:-$(dirname "$OUTPUT")/relight_$(date +%Y%m%d_%H%M%S).log}"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "Log: $LOG_FILE"
    [[ -d "$MODEL" ]] || { echo "模型目录不存在: $MODEL" >&2; exit 1; }
    [[ -f "$ENV_PATH" ]] || { echo "环境贴图不存在: $ENV_PATH" >&2; exit 1; }
    exec "$PY" -u relight_single.py \
        -m "$MODEL" \
        --iteration "${ITERATION:-20000}" \
        --envmap "$ENV_PATH" \
        --output "$OUTPUT" \
        --diffuse-samples "${DIFFUSE_SAMPLES:-32}" \
        --light-samples "${LIGHT_SAMPLES:-32}" \
        --light-t-min "${LIGHT_T_MIN:-0.10}" \
        "$@"
fi

shift 2
OUTPUT="${OUTPUT_DIR:-$PROJECT_DIR/outputs/${SCENE_NAME}_relight}"
FORWARD_ARGS=("$@")
for ((i=0; i<${#FORWARD_ARGS[@]}; i++)); do
    if [[ "${FORWARD_ARGS[$i]}" == "--out" && $((i + 1)) -lt ${#FORWARD_ARGS[@]} ]]; then
        OUTPUT="${FORWARD_ARGS[$((i + 1))]}"
        break
    fi
done
mkdir -p "$OUTPUT"
LOG_FILE="${RELIGHT_LOG:-$OUTPUT/relight_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log: $LOG_FILE"
ASSEMBLED_ARGS=(--out "$OUTPUT" \
    --diffuse-samples "${DIFFUSE_SAMPLES:-32}" \
    --light-samples "${LIGHT_SAMPLES:-32}" \
    --light-t-min "${LIGHT_T_MIN:-0.10}" \
    --bvh-layout "${BVH_LAYOUT:-component-ias-rigid}")
if [[ "$SCENE_NAME" == "guiji1" || "$SCENE_NAME" == "guiji2" ]]; then
    ASSEMBLED_ARGS+=(--trajectory-set "$SCENE_NAME")
else
    # 自定义轨迹包：目录内自动寻找 trajectories/*.csv，其他配置按约定目录读取。
    [[ -d "$SCENE_NAME" ]] || { echo "轨迹目录不存在: $SCENE_NAME" >&2; exit 1; }
    ASSEMBLED_ARGS+=(--trajectory-dir "$SCENE_NAME")
fi
exec "$PY" -u relight_assembled.py "${ASSEMBLED_ARGS[@]}" "${FORWARD_ARGS[@]}"
