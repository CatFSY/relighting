#!/usr/bin/env bash
set -uo pipefail

output="$1"
mkdir -p "$output"
python_bin="/amax/home/fengshuangyu/miniconda3/envs/irgs/bin/python"
runtime_path="/amax/home/fengshuangyu/miniconda3/envs/irgs/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

attempt=0
while true; do
    attempt=$((attempt + 1))
    printf '%s,%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$attempt" \
        >> "$output/attempt_history.csv"

    if scripts/run_with_gpu_memory_monitor.sh 2 "$output" \
        env PYTHONUNBUFFERED=1 PATH="$runtime_path" CUDA_VISIBLE_DEVICES=2 \
        IRGS_GS_BOUNDING_POLYHEDRON=icosphere320 \
        "$python_bin" scripts/render_guiji_irgs.py \
        --trajectory-set guiji1 --scene full --full-video --stride 4 \
        --width 960 --height 540 --fps 30 \
        --diffuse-samples 32 --light-samples 32 \
        --diffuse-sampling-mode cosine \
        --light-sampling-mode stratified_shared \
        --light-t-min 0.10 --render-ray-budget 16777216 \
        --fg-lut-query-layout tiled --fg-lut-tile-width 2048 \
        --base-color-min 0.03 \
        --envmap assets/env_map/pointlike_camera_key_light_fill035_key2500.exr \
        --trajectory-env-deg-per-sec 90 --env-rotate-axis z \
        --table-z-offset-m 0 --material-variant-name stratified_ds32_ls32 \
        --out "$output"; then
        "$python_bin" scripts/summarize_guiji_timing.py \
            "$output/video_report.json" --discard 10 \
            --output "$output/hot_timing_summary.json"
        exit 0
    fi

    if ! grep -qE 'OutOfMemoryError|CUDA out of memory|cudaMalloc' \
        "$output/background_render.log"; then
        exit 1
    fi
    cp "$output/background_render.log" \
        "$output/attempt_${attempt}_oom.log"
    cp "$output/gpu_memory_summary.json" \
        "$output/attempt_${attempt}_memory_summary.json"

    # Retry only on GPU 2 after it has remained below 12 GiB for 15 seconds.
    stable=0
    while (( stable < 3 )); do
        values="$(nvidia-smi --id=2 \
            --query-gpu=memory.used,utilization.gpu \
            --format=csv,noheader,nounits | tr -d ' ')"
        memory_used="${values%%,*}"
        utilization="${values##*,}"
        if (( memory_used <= 12000 && utilization <= 20 )); then
            stable=$((stable + 1))
        else
            stable=0
        fi
        sleep 5
    done
done
