#!/usr/bin/env bash
set -Eeuo pipefail

OUTPUT="${1:-system_samples.csv}"
INTERVAL="${MONITOR_INTERVAL_SECONDS:-1}"
PID="${LLAMA_SERVER_PID:-}"
STOP_FILE="${MONITOR_STOP_FILE:-}"

mkdir -p "$(dirname "$OUTPUT")"

echo "timestamp,gpu_util,gpu_mem_used_mb,gpu_mem_free_mb,gpu_power_w,gpu_pstate,ram_used_mb,ram_available_mb,swap_used_mb,llama_server_rss_mb,llama_server_vms_mb,num_llama_threads,num_open_fds" > "$OUTPUT"

read_mem_field_kb() {
  local key="$1"
  awk -v key="$key" '$1 == key ":" { print int($2 / 1024) }' /proc/meminfo
}

while true; do
  if [[ -n "$STOP_FILE" && -f "$STOP_FILE" ]]; then
    break
  fi

  ts="$(date -Is)"
  gpu_line="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free,power.draw,pstate --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || true)"
  if [[ -z "$gpu_line" ]]; then
    gpu_line=",,,,"
  fi

  mem_total="$(read_mem_field_kb MemTotal)"
  mem_available="$(read_mem_field_kb MemAvailable)"
  swap_total="$(read_mem_field_kb SwapTotal)"
  swap_free="$(read_mem_field_kb SwapFree)"
  ram_used=$((mem_total - mem_available))
  swap_used=$((swap_total - swap_free))

  rss=0
  vms=0
  threads=0
  fds=0
  if [[ -n "$PID" && -d "/proc/$PID" ]]; then
    rss="$(awk '/VmRSS:/ { print int($2 / 1024) }' "/proc/$PID/status" 2>/dev/null || echo 0)"
    vms="$(awk '/VmSize:/ { print int($2 / 1024) }' "/proc/$PID/status" 2>/dev/null || echo 0)"
    threads="$(awk '/Threads:/ { print $2 }' "/proc/$PID/status" 2>/dev/null || echo 0)"
    fds="$(ls "/proc/$PID/fd" 2>/dev/null | wc -l || echo 0)"
  fi

  echo "$ts,$gpu_line,$ram_used,$mem_available,$swap_used,$rss,$vms,$threads,$fds" >> "$OUTPUT"
  sleep "$INTERVAL"
done
