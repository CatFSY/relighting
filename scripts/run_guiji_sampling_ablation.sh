#!/usr/bin/env bash
set -euo pipefail

gpu_index="${1:-3}"
wait_pid="${2:-0}"
output_root="${3:-outputs/guiji1_sampling_ablation_step748_icosphere320}"

while [[ "$wait_pid" != "0" ]] && kill -0 "$wait_pid" 2>/dev/null; do
    sleep 10
done

mkdir -p "$output_root"
envmap="assets/env_map/pointlike_camera_key_light_fill035_key2500.exr"
python_bin="/amax/home/fengshuangyu/miniconda3/envs/irgs/bin/python"
runtime_path="/amax/home/fengshuangyu/miniconda3/envs/irgs/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

run_variant() {
    local name="$1"
    local ds="$2"
    local ls="$3"
    local diffuse_mode="$4"
    local light_mode="$5"
    local output="$output_root/$name"

    env \
        PYTHONUNBUFFERED=1 \
        PATH="$runtime_path" \
        CUDA_VISIBLE_DEVICES="$gpu_index" \
        IRGS_GS_BOUNDING_POLYHEDRON=icosphere320 \
        "$python_bin" scripts/render_guiji_irgs.py \
        --trajectory-set guiji1 \
        --scene full \
        --steps 748 \
        --width 960 \
        --height 540 \
        --diffuse-samples "$ds" \
        --light-samples "$ls" \
        --diffuse-sampling-mode "$diffuse_mode" \
        --light-sampling-mode "$light_mode" \
        --light-t-min 0.10 \
        --render-ray-budget 67108864 \
        --fg-lut-query-layout tiled \
        --fg-lut-tile-width 2048 \
        --base-color-min 0.03 \
        --envmap "$envmap" \
        --material-variant-name "$name" \
        --out "$output" \
        > "$output_root/$name.log" 2>&1
}

run_variant baseline_ds128_ls256_iid 128 256 uniform iid
run_variant stratified_ds64_ls64 64 64 cosine stratified_shared
run_variant stratified_ds32_ls32 32 32 cosine stratified_shared

"$python_bin" scripts/analyze_sampling_ablation.py "$output_root"
