#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
usage: run_stage_a_smoke.sh --output-dir DIR

Environment overrides:
  MODEL_REPO_ID
  MODEL_FILENAME
  MODEL_DIR
  MODEL_EXPECTED_SIZE_BYTES
  SERVER_PORT
  USE_PROXY=1 PROXY_URL=http://127.0.0.1:7890
  SKIP_MODEL_DOWNLOAD=1
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

MODEL_REPO_ID="${MODEL_REPO_ID:-TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF}"
MODEL_FILENAME="${MODEL_FILENAME:-mixtral-8x7b-instruct-v0.1.Q8_0.gguf}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/llama-models/mixtral-8x7b-instruct-v0.1}"
MODEL_EXPECTED_SIZE_BYTES="${MODEL_EXPECTED_SIZE_BYTES:-49624262592}"
MODEL_PATH="$MODEL_DIR/$MODEL_FILENAME"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-18080}"
BASE_URL="http://$SERVER_HOST:$SERVER_PORT"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUTPUT_DIR"/{model,server,monitor,outputs,summaries}

if [[ "${USE_PROXY:-0}" == "1" && -n "${PROXY_URL:-}" ]]; then
  export http_proxy="$PROXY_URL"
  export https_proxy="$PROXY_URL"
  export HTTP_PROXY="$PROXY_URL"
  export HTTPS_PROXY="$PROXY_URL"
fi

LLAMA_CPP_CONDA_ENV="${LLAMA_CPP_CONDA_ENV:-/root/autodl-tmp/conda-envs/llama-cpp}"
if [[ -f /root/miniconda3/bin/activate && -d "$LLAMA_CPP_CONDA_ENV" ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/bin/activate "$LLAMA_CPP_CONDA_ENV"
fi

python -m pip install -q -U huggingface_hub hf_xet >/dev/null

cat > "$OUTPUT_DIR/run_config.json" <<EOF
{
  "stage": "A_smoke",
  "model_repo_id": "$MODEL_REPO_ID",
  "model_filename": "$MODEL_FILENAME",
  "model_path": "$MODEL_PATH",
  "model_expected_size_bytes": $MODEL_EXPECTED_SIZE_BYTES,
  "server_url": "$BASE_URL",
  "moe_placement": "-cmoe",
  "kv_offload": "gpu",
  "cache_type_k": "f16",
  "cache_type_v": "f16",
  "ctx_size": 2048,
  "parallel_slots": 1,
  "batch_size": 512,
  "ubatch_size": 256
}
EOF

download_model() {
  mkdir -p "$MODEL_DIR"
  python - "$MODEL_REPO_ID" "$MODEL_FILENAME" "$MODEL_DIR" <<'PY'
import os
import sys
from huggingface_hub import hf_hub_download

repo_id, filename, local_dir = sys.argv[1:4]
path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    token=os.environ.get("HF_TOKEN"),
)
print(path)
PY
}

if [[ ! -f "$MODEL_PATH" ]]; then
  if [[ "${SKIP_MODEL_DOWNLOAD:-0}" == "1" ]]; then
    echo "model missing and SKIP_MODEL_DOWNLOAD=1: $MODEL_PATH" >&2
    exit 20
  fi
  echo "Downloading model $MODEL_REPO_ID/$MODEL_FILENAME"
  download_model
fi

actual_size="$(stat -c '%s' "$MODEL_PATH")"
echo "$MODEL_PATH" > "$OUTPUT_DIR/model/model_path.txt"
echo "$actual_size" > "$OUTPUT_DIR/model/model_file_size.txt"
if [[ "$actual_size" -lt "$MODEL_EXPECTED_SIZE_BYTES" ]]; then
  echo "model file is smaller than expected: actual=$actual_size expected=$MODEL_EXPECTED_SIZE_BYTES" >&2
  exit 21
fi

SERVER_PID=""
STOP_FILE="$OUTPUT_DIR/monitor/stop"
cleanup() {
  touch "$STOP_FILE" || true
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

bash "$SCRIPT_DIR/run_llama_server.sh" \
  --output-dir "$OUTPUT_DIR" \
  --model "$MODEL_PATH" \
  --host "$SERVER_HOST" \
  --port "$SERVER_PORT" \
  --ctx-size 2048 \
  --parallel 1 \
  --batch-size 512 \
  --ubatch-size 256 \
  --gpu-layers 99 \
  --moe-placement "-cmoe" \
  --kv-offload gpu \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --flash-attn auto | tee "$OUTPUT_DIR/server/launcher.log"

SERVER_PID="$(cat "$OUTPUT_DIR/server/server.pid")"
export LLAMA_SERVER_PID="$SERVER_PID"
export MONITOR_STOP_FILE="$STOP_FILE"
export MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-1}"

bash "$SCRIPT_DIR/monitor_system.sh" "$OUTPUT_DIR/monitor/system_samples.csv" &
MONITOR_PID=$!

python "$SCRIPT_DIR/collect_llama_metrics.py" \
  --base-url "$BASE_URL" \
  --output-dir "$OUTPUT_DIR/monitor" \
  --interval "${METRICS_INTERVAL_SECONDS:-1}" \
  --stop-file "$STOP_FILE" &
METRICS_PID=$!

python "$SCRIPT_DIR/run_lmsys_1m_client.py" \
  --base-url "$BASE_URL" \
  --output-dir "$OUTPUT_DIR/outputs" \
  --output-name smoke.results.jsonl.gz \
  --smoke \
  --concurrency 1 \
  --max-tokens 64 \
  --temperature 0.0 \
  --top-p 1.0 \
  --store-output-text

touch "$STOP_FILE"
wait "$MONITOR_PID" 2>/dev/null || true
wait "$METRICS_PID" 2>/dev/null || true

python "$SCRIPT_DIR/summarize_results.py" --run-dir "$OUTPUT_DIR"

curl -fsS "$BASE_URL/metrics" > "$OUTPUT_DIR/server/final_metrics.txt" || true
curl -fsS "$BASE_URL/slots" > "$OUTPUT_DIR/server/final_slots.json" || true
nvidia-smi > "$OUTPUT_DIR/monitor/nvidia_smi_after.txt" || true

echo "Stage A smoke completed: $OUTPUT_DIR"
