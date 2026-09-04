#!/usr/bin/env bash
# Render a SO101 handoff across multiple GPUs with completion-based resume.

set -u
set -o pipefail

usage() {
    cat <<'EOF'
Usage:
  render_so101_irgs_multi_gpu.sh HANDOFF_ROOT [GPU_LIST] [renderer options...]

Examples:
  ./render_so101_irgs_multi_gpu.sh /path/to/handoff 0,1,2,3
  ./render_so101_irgs_multi_gpu.sh /path/to/handoff 4,5,6,7 --camera main
  GPUS=0,2 ./render_so101_irgs_multi_gpu.sh /path/to/handoff --batch 1

GPU_LIST is a comma-separated list and defaults to $GPUS or 0.
Outputs default to HANDOFF_ROOT/batch_*/episodes/episode_*/video. Run the same command again to resume:
completed camera videos (video + video_report.json) are skipped automatically.
Pass --rerender only when completed outputs should be replaced.
On normal completion, worker failure, or Ctrl-C, render_metrics.json is written
next to the worker logs with per-video/per-GPU frame counts and elapsed times.
EOF
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    [[ $# -ge 1 ]] && exit 0 || exit 2
fi

handoff_root=$1
shift
gpu_list=${GPUS:-0}
if [[ $# -gt 0 && $1 != --* ]]; then
    gpu_list=$1
    shift
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
launcher=${IRGS_LAUNCH_PYTHON:-python3}
log_root="${handoff_root%/}/render_logs"
scan_root=""
extra_args=("$@")

# Honor an explicit --output-root when choosing the worker-log directory.
for ((arg_index = 0; arg_index < ${#extra_args[@]}; arg_index++)); do
    if [[ ${extra_args[arg_index]} == "--output-root" ]]; then
        if ((arg_index + 1 >= ${#extra_args[@]})); then
            echo "error: --output-root requires a value" >&2
            exit 2
        fi
        log_root=${extra_args[arg_index + 1]}/logs
        scan_root=${extra_args[arg_index + 1]}
    elif [[ ${extra_args[arg_index]} == --output-root=* ]]; then
        log_root=${extra_args[arg_index]#--output-root=}/logs
        scan_root=${extra_args[arg_index]#--output-root=}
    fi
done

IFS=',' read -r -a gpu_ids <<< "$gpu_list"
if [[ ${#gpu_ids[@]} -eq 0 ]]; then
    echo "error: GPU_LIST is empty" >&2
    exit 2
fi

declare -A seen_gpus=()
for gpu_id in "${gpu_ids[@]}"; do
    if [[ ! $gpu_id =~ ^[0-9]+$ ]]; then
        echo "error: invalid GPU id: $gpu_id" >&2
        exit 2
    fi
    if [[ -n ${seen_gpus[$gpu_id]:-} ]]; then
        echo "error: duplicate GPU id: $gpu_id" >&2
        exit 2
    fi
    seen_gpus[$gpu_id]=1
done

mkdir -p -- "$log_root"
worker_count=${#gpu_ids[@]}
pids=()
worker_logs=()

collect_metrics() {
    local collection_status=$1
    local collector_args=(
        "$handoff_root"
        --output "$log_root/render_metrics.json"
        --status "$collection_status"
    )
    if [[ -n $scan_root ]]; then
        collector_args+=(--scan-root "$scan_root")
    fi
    "$launcher" "$script_dir/scripts/summarize_so101_render_metrics.py" \
        "${collector_args[@]}" || true
}

stop_workers() {
    trap - INT TERM
    echo "Stopping all renderer workers..." >&2
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            # Each worker is started in its own session. Killing the process
            # group also stops the active CUDA renderer subprocess.
            kill -TERM -- "-$pid" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
    collect_metrics interrupted
    echo "Stopped. Re-run the same command to resume completed work." >&2
    exit 130
}
trap stop_workers INT TERM

for ((worker_index = 0; worker_index < worker_count; worker_index++)); do
    gpu_id=${gpu_ids[worker_index]}
    worker_log="$log_root/worker_${worker_index}_gpu_${gpu_id}.log"
    command=(
        "$launcher" "$script_dir/render_so101_irgs_handoff.py" "$handoff_root"
        --gpu "$gpu_id"
        --worker-index "$worker_index"
        --worker-count "$worker_count"
        "${extra_args[@]}"
    )
    {
        printf '\n[%s] ' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf '%q ' "${command[@]}"
        printf '\n'
    } >>"$worker_log"
    # Keep the worker and tee in one session so Ctrl-C/TERM can stop the
    # complete process group. Renderer progress is visible in the terminal
    # and appended to the per-worker log at the same time.
    setsid bash -c '
        log_file=$1
        shift
        set -o pipefail
        "$@" 2>&1 | tee -a "$log_file"
        exit "${PIPESTATUS[0]}"
    ' _ "$worker_log" "${command[@]}" &
    pid=$!
    pids+=("$pid")
    worker_logs+=("$worker_log")
    echo "worker $worker_index/$worker_count: GPU $gpu_id, pid $pid"
    echo "  log: $worker_log"
done

failed=0
for ((worker_index = 0; worker_index < worker_count; worker_index++)); do
    pid=${pids[worker_index]}
    if wait "$pid"; then
        echo "worker $worker_index finished: ${worker_logs[worker_index]}"
    else
        status=$?
        echo "worker $worker_index failed (exit $status): ${worker_logs[worker_index]}" >&2
        failed=1
    fi
done

if [[ $failed -ne 0 ]]; then
    collect_metrics failed
else
    collect_metrics completed
fi

if [[ $failed -ne 0 ]]; then
    echo "One or more workers failed. Fix the error and re-run to resume." >&2
    exit 1
fi
echo "All workers finished. Videos are under HANDOFF_ROOT/batch_*/episodes/*/video"
echo "Worker logs: $log_root"
