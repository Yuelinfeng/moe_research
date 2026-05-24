#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
usage: run_stage_b_100_sweep.sh --output-dir DIR

Environment overrides:
  MODEL_PATH
  MODEL_EXPECTED_SIZE_BYTES=49626320288
  DATASET_DIR=/root/autodl-tmp/datasets/lmsys-chat-1m-stage-b-100
  SERVER_PORT=18080
  PROMPT_LIMIT=100
  MAX_TOKENS=64
  METRICS_INTERVAL_SECONDS=1
  MONITOR_INTERVAL_SECONDS=1
  USE_PROXY=1 PROXY_URL=http://127.0.0.1:7890
  CONFIGS_FILE=/path/to/configs.txt

Config file format:
  name|moe_placement|kv_offload|cache_type_k|cache_type_v|ctx|parallel|batch|ubatch|concurrency|kv_bytes
EOF
}

OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/llama-models/mradermacher-mixtral-8x7b-instruct-v0.1/Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf}"
MODEL_EXPECTED_SIZE_BYTES="${MODEL_EXPECTED_SIZE_BYTES:-49626320288}"
DATASET_DIR="${DATASET_DIR:-/root/autodl-tmp/datasets/lmsys-chat-1m-stage-b-100}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-18080}"
PROMPT_LIMIT="${PROMPT_LIMIT:-100}"
MAX_TOKENS="${MAX_TOKENS:-64}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
METRICS_INTERVAL_SECONDS="${METRICS_INTERVAL_SECONDS:-1}"
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-1}"
LLAMA_CPP_CONDA_ENV="${LLAMA_CPP_CONDA_ENV:-/root/autodl-tmp/conda-envs/llama-cpp}"

mkdir -p "$OUTPUT_DIR"/{configs,summaries,logs}

if [[ "${USE_PROXY:-0}" == "1" && -n "${PROXY_URL:-}" ]]; then
  export http_proxy="$PROXY_URL"
  export https_proxy="$PROXY_URL"
  export HTTP_PROXY="$PROXY_URL"
  export HTTPS_PROXY="$PROXY_URL"
fi

if [[ -f /root/miniconda3/bin/activate && -d "$LLAMA_CPP_CONDA_ENV" ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/bin/activate "$LLAMA_CPP_CONDA_ENV"
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "missing model file: $MODEL_PATH" >&2
  exit 20
fi

actual_size="$(stat -c '%s' "$MODEL_PATH")"
if [[ "$actual_size" -lt "$MODEL_EXPECTED_SIZE_BYTES" ]]; then
  echo "model file is smaller than expected: actual=$actual_size expected=$MODEL_EXPECTED_SIZE_BYTES" >&2
  exit 21
fi

python -m pip install -q -U datasets pyarrow >/dev/null

SHARD_DIR="$DATASET_DIR/shards"
if ! find "$SHARD_DIR" -maxdepth 1 -name '*.jsonl' -print -quit 2>/dev/null | grep -q .; then
  python "$SCRIPT_DIR/prepare_lmsys_chat1m.py" \
    --output-dir "$DATASET_DIR" \
    --limit "$PROMPT_LIMIT" \
    --shard-size "$PROMPT_LIMIT" | tee "$OUTPUT_DIR/logs/prepare_dataset.log"
fi

DEFAULT_CONFIGS=$(cat <<'EOF'
s2_gpu_kv_f16_np1|-cmoe|gpu|f16|f16|2048|1|512|256|1|2
s2_cpu_kv_f16_np1|-cmoe|cpu|f16|f16|2048|1|512|256|1|2
s2_gpu_kv_q8_np1|-cmoe|gpu|q8_0|q8_0|2048|1|512|256|1|1
s2_gpu_kv_f16_np2|-cmoe|gpu|f16|f16|2048|2|512|256|2|2
s2_cpu_kv_f16_np2|-cmoe|cpu|f16|f16|2048|2|512|256|2|2
EOF
)

if [[ -n "${CONFIGS_FILE:-}" ]]; then
  CONFIG_LINES="$(cat "$CONFIGS_FILE")"
else
  CONFIG_LINES="$DEFAULT_CONFIGS"
fi

append_summary_index() {
  local config_name="$1"
  local config_dir="$2"
  local status="$3"
  python - "$config_name" "$config_dir" "$status" "$OUTPUT_DIR/summaries/stage_b_summary.jsonl" <<'PY'
import json
import sys
from pathlib import Path

name, config_dir, status, index_path = sys.argv[1:5]
config_dir = Path(config_dir)
row = {"config_name": name, "status": status, "config_dir": str(config_dir)}
for rel in ("run_config.json", "summaries/summary.json"):
    path = config_dir / rel
    if path.exists():
        key = "run_config" if rel == "run_config.json" else "summary"
        row[key] = json.loads(path.read_text(encoding="utf-8"))
Path(index_path).parent.mkdir(parents=True, exist_ok=True)
with Path(index_path).open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

cleanup_config() {
  local stop_file="${1:-}"
  local server_pid="${2:-}"
  if [[ -n "$stop_file" ]]; then
    touch "$stop_file" 2>/dev/null || true
  fi
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    sleep 2
    kill -9 "$server_pid" 2>/dev/null || true
  fi
}

while IFS='|' read -r CONFIG_NAME MOE_PLACEMENT KV_OFFLOAD CACHE_TYPE_K CACHE_TYPE_V CTX_SIZE PARALLEL BATCH_SIZE UBATCH_SIZE CONCURRENCY KV_BYTES; do
  if [[ -z "${CONFIG_NAME:-}" || "${CONFIG_NAME:0:1}" == "#" ]]; then
    continue
  fi

  CONFIG_DIR="$OUTPUT_DIR/configs/$CONFIG_NAME"
  STOP_FILE="$CONFIG_DIR/monitor/stop"
  mkdir -p "$CONFIG_DIR"/{server,monitor,outputs,summaries}
  rm -f "$STOP_FILE"

  cat > "$CONFIG_DIR/run_config.json" <<EOF
{
  "stage": "B_100_sweep",
  "config_name": "$CONFIG_NAME",
  "model_path": "$MODEL_PATH",
  "dataset_dir": "$DATASET_DIR",
  "server_url": "http://$SERVER_HOST:$SERVER_PORT",
  "moe_placement": "$MOE_PLACEMENT",
  "kv_offload": "$KV_OFFLOAD",
  "cache_type_k": "$CACHE_TYPE_K",
  "cache_type_v": "$CACHE_TYPE_V",
  "ctx_size": $CTX_SIZE,
  "parallel_slots": $PARALLEL,
  "batch_size": $BATCH_SIZE,
  "ubatch_size": $UBATCH_SIZE,
  "client_concurrency": $CONCURRENCY,
  "prompt_limit": $PROMPT_LIMIT,
  "max_tokens": $MAX_TOKENS
}
EOF

  echo "stage_b_config_start=$CONFIG_NAME"
  SERVER_PID=""

  set +e
  bash "$SCRIPT_DIR/run_llama_server.sh" \
    --output-dir "$CONFIG_DIR" \
    --model "$MODEL_PATH" \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    --ctx-size "$CTX_SIZE" \
    --parallel "$PARALLEL" \
    --batch-size "$BATCH_SIZE" \
    --ubatch-size "$UBATCH_SIZE" \
    --gpu-layers 99 \
    --moe-placement "$MOE_PLACEMENT" \
    --kv-offload "$KV_OFFLOAD" \
    --cache-type-k "$CACHE_TYPE_K" \
    --cache-type-v "$CACHE_TYPE_V" \
    --flash-attn auto > "$CONFIG_DIR/server/launcher.log" 2>&1
  LAUNCH_EXIT=$?
  set -e

  if [[ "$LAUNCH_EXIT" -ne 0 ]]; then
    echo "stage_b_config_failed=$CONFIG_NAME phase=server_launch exit=$LAUNCH_EXIT" >&2
    append_summary_index "$CONFIG_NAME" "$CONFIG_DIR" "server_launch_failed"
    cleanup_config "$STOP_FILE" "$SERVER_PID"
    sleep 10
    continue
  fi

  SERVER_PID="$(cat "$CONFIG_DIR/server/server.pid")"
  export LLAMA_SERVER_PID="$SERVER_PID"
  export MONITOR_STOP_FILE="$STOP_FILE"
  export MONITOR_INTERVAL_SECONDS

  bash "$SCRIPT_DIR/monitor_system.sh" "$CONFIG_DIR/monitor/system_samples.csv" &
  MONITOR_PID=$!

  python "$SCRIPT_DIR/collect_llama_metrics.py" \
    --base-url "http://$SERVER_HOST:$SERVER_PORT" \
    --output-dir "$CONFIG_DIR/monitor" \
    --interval "$METRICS_INTERVAL_SECONDS" \
    --stop-file "$STOP_FILE" \
    --kv-bytes "$KV_BYTES" &
  METRICS_PID=$!

  set +e
  python "$SCRIPT_DIR/run_lmsys_1m_client.py" \
    --base-url "http://$SERVER_HOST:$SERVER_PORT" \
    --shard-dir "$SHARD_DIR" \
    --output-dir "$CONFIG_DIR/outputs" \
    --output-name "stage_b_100.results.jsonl.gz" \
    --limit "$PROMPT_LIMIT" \
    --concurrency "$CONCURRENCY" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P"
  CLIENT_EXIT=$?

  touch "$STOP_FILE"
  wait "$MONITOR_PID" 2>/dev/null
  MONITOR_EXIT=$?
  wait "$METRICS_PID" 2>/dev/null
  METRICS_EXIT=$?

  curl -fsS "http://$SERVER_HOST:$SERVER_PORT/metrics" > "$CONFIG_DIR/server/final_metrics.txt" 2>/dev/null
  curl -fsS "http://$SERVER_HOST:$SERVER_PORT/slots" > "$CONFIG_DIR/server/final_slots.json" 2>/dev/null
  nvidia-smi > "$CONFIG_DIR/monitor/nvidia_smi_after.txt" 2>/dev/null

  python "$SCRIPT_DIR/summarize_results.py" \
    --run-dir "$CONFIG_DIR" \
    --kv-bytes "$KV_BYTES" > "$CONFIG_DIR/summaries/summary.stdout.json"
  SUMMARY_EXIT=$?
  set -e

  cleanup_config "$STOP_FILE" "$SERVER_PID"

  if [[ "$CLIENT_EXIT" -eq 0 && "$MONITOR_EXIT" -eq 0 && "$METRICS_EXIT" -eq 0 && "$SUMMARY_EXIT" -eq 0 ]]; then
    STATUS="ok"
  else
    STATUS="failed_client_${CLIENT_EXIT}_monitor_${MONITOR_EXIT}_metrics_${METRICS_EXIT}_summary_${SUMMARY_EXIT}"
  fi
  append_summary_index "$CONFIG_NAME" "$CONFIG_DIR" "$STATUS"
  echo "stage_b_config_done=$CONFIG_NAME status=$STATUS"
  sleep 10
done <<< "$CONFIG_LINES"

echo "Stage B 100-prompt sweep completed: $OUTPUT_DIR"
