#!/usr/bin/env bash
set -euo pipefail

output="${1:-outputs/guiji1_full_trajectory_stratified_ds32_ls32_icosphere320_t010_envrotate90dps}"
mkdir -p "$output"

runtime_path="/amax/home/fengshuangyu/miniconda3/envs/irgs/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
python_bin="/amax/home/fengshuangyu/miniconda3/envs/irgs/bin/python"

run_render() {
    local selected_gpu="$1"
    scripts/run_with_gpu_memory_monitor.sh "$selected_gpu" "$output" \
    env \
    PYTHONUNBUFFERED=1 \
    PATH="$runtime_path" \
    CUDA_VISIBLE_DEVICES="$selected_gpu" \
    IRGS_GS_BOUNDING_POLYHEDRON=icosphere320 \
    "$python_bin" scripts/render_guiji_irgs.py \
    --trajectory-set guiji1 \
    --scene full \
    --full-video \
    --stride 4 \
    --width 960 \
    --height 540 \
    --fps 30 \
    --diffuse-samples 32 \
    --light-samples 32 \
    --diffuse-sampling-mode cosine \
    --light-sampling-mode stratified_shared \
    --light-t-min 0.10 \
    --render-ray-budget 16777216 \
    --fg-lut-query-layout tiled \
    --fg-lut-tile-width 2048 \
    --base-color-min 0.03 \
    --envmap assets/env_map/pointlike_camera_key_light_fill035_key2500.exr \
    --trajectory-env-deg-per-sec 90 \
    --env-rotate-axis z \
    --table-z-offset-m 0 \
    --material-variant-name stratified_ds32_ls32 \
    --out "$output"
}

attempt=0
while true; do
    selected_gpu=""
    stable_gpu=""
    stable_count=0
    # Require the same GPU to stay below the threshold for three consecutive
    # polls. This avoids selecting a card during a brief gap between jobs.
    while [[ "$stable_count" -lt 3 ]]; do
        selected_gpu=""
        while IFS=',' read -r index memory_used utilization; do
            index="${index// /}"
            memory_used="${memory_used// /}"
            utilization="${utilization// /}"
            if (( memory_used <= 10000 && utilization <= 10 )); then
                selected_gpu="$index"
                break
            fi
        done < <(nvidia-smi \
            --query-gpu=index,memory.used,utilization.gpu \
            --format=csv,noheader,nounits)

        if [[ -n "$selected_gpu" && "$selected_gpu" == "$stable_gpu" ]]; then
            stable_count=$((stable_count + 1))
        elif [[ -n "$selected_gpu" ]]; then
            stable_gpu="$selected_gpu"
            stable_count=1
        else
            stable_gpu=""
            stable_count=0
        fi
        sleep 5
    done

    selected_gpu="$stable_gpu"
    attempt=$((attempt + 1))
    echo "$selected_gpu" > "$output/selected_gpu.txt"
    printf '%s,%s,%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$attempt" "$selected_gpu" >> "$output/selected_gpu_history.csv"

    if run_render "$selected_gpu"; then
        break
    fi
    if ! grep -qE 'OutOfMemoryError|CUDA out of memory|cudaMalloc' \
        "$output/background_render.log"; then
        exit 1
    fi
    sleep 30
done

"$python_bin" scripts/summarize_guiji_timing.py \
    "$output/video_report.json" \
    --discard 10 \
    --output "$output/hot_timing_summary.json"
