#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 GPU_INDEX OUTPUT_DIR COMMAND [ARGS...]" >&2
    exit 2
fi

gpu_index="$1"
output_dir="$2"
shift 2

mkdir -p "$output_dir"
log_file="$output_dir/background_render.log"
samples_file="$output_dir/gpu_memory_samples.csv"

"$@" > "$log_file" 2>&1 &
render_pid="$!"
echo "$render_pid" > "$output_dir/background_render.pid"

"$(dirname "$0")/monitor_process_gpu_memory.sh" \
    "$render_pid" "$gpu_index" "$samples_file" \
    > "$output_dir/gpu_memory_monitor.log" 2>&1 &
monitor_pid="$!"
echo "$monitor_pid" > "$output_dir/gpu_memory_monitor.pid"

wait "$render_pid"
exit_code="$?"
wait "$monitor_pid" 2>/dev/null || true

process_peak="$(awk -F',' 'NR > 1 && $4 + 0 > peak {peak=$4+0} END {print peak+0}' "$samples_file")"
gpu_peak="$(awk -F',' 'NR > 1 && $5 + 0 > peak {peak=$5+0} END {print peak+0}' "$samples_file")"
sample_count="$(awk 'END {print (NR > 0 ? NR - 1 : 0)}' "$samples_file")"

cat > "$output_dir/gpu_memory_summary.json" <<EOF
{
  "gpu_index": $gpu_index,
  "render_pid": $render_pid,
  "sample_interval_seconds": 1,
  "sample_count": $sample_count,
  "process_peak_memory_mib": $process_peak,
  "gpu_peak_total_memory_mib": $gpu_peak,
  "render_exit_code": $exit_code
}
EOF

exit "$exit_code"
