#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 PID GPU_INDEX OUTPUT_CSV" >&2
    exit 2
fi

target_pid="$1"
gpu_index="$2"
output_csv="$3"

echo "timestamp_utc,pid,gpu_index,process_used_memory_mib,gpu_total_used_memory_mib,gpu_utilization_percent" > "$output_csv"

while kill -0 "$target_pid" 2>/dev/null; do
    process_memory="$({
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true
    } | awk -F',' -v pid="$target_pid" '$1 + 0 == pid {gsub(/ /, "", $2); print $2; found=1} END {if (!found) print 0}')"

    gpu_values="$(nvidia-smi --id="$gpu_index" \
        --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"

    printf '%s,%s,%s,%s,%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)" \
        "$target_pid" "$gpu_index" "$process_memory" "$gpu_values" \
        >> "$output_csv"
    sleep 1
done
