#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
usage: run_llama_server.sh --output-dir DIR --model PATH [options]

Options:
  --host HOST             Default: 127.0.0.1
  --port PORT             Default: 18080
  --ctx-size N            Default: 2048
  --parallel N            Default: 1
  --batch-size N          Default: 512
  --ubatch-size N         Default: 256
  --gpu-layers N          Default: 99
  --moe-placement ARGS    Default: -cmoe
  --kv-offload MODE       gpu or cpu. Default: gpu
  --cache-type-k TYPE     Default: f16
  --cache-type-v TYPE     Default: f16
  --flash-attn MODE       Default: auto
EOF
}

OUTPUT_DIR=""
MODEL=""
HOST="127.0.0.1"
PORT="18080"
CTX_SIZE="2048"
PARALLEL="1"
BATCH_SIZE="512"
UBATCH_SIZE="256"
GPU_LAYERS="99"
MOE_PLACEMENT="-cmoe"
KV_OFFLOAD="gpu"
CACHE_TYPE_K="f16"
CACHE_TYPE_V="f16"
FLASH_ATTN="auto"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --ctx-size) CTX_SIZE="$2"; shift 2 ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --ubatch-size) UBATCH_SIZE="$2"; shift 2 ;;
    --gpu-layers) GPU_LAYERS="$2"; shift 2 ;;
    --moe-placement) MOE_PLACEMENT="$2"; shift 2 ;;
    --kv-offload) KV_OFFLOAD="$2"; shift 2 ;;
    --cache-type-k) CACHE_TYPE_K="$2"; shift 2 ;;
    --cache-type-v) CACHE_TYPE_V="$2"; shift 2 ;;
    --flash-attn) FLASH_ATTN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$OUTPUT_DIR" || -z "$MODEL" ]]; then
  usage >&2
  exit 2
fi

LLAMA_CPP_BUILD_DIR="${LLAMA_CPP_BUILD_DIR:-/root/autodl-tmp/llama.cpp/build-cuda-env}"
LLAMA_CPP_CONDA_ENV="${LLAMA_CPP_CONDA_ENV:-/root/autodl-tmp/conda-envs/llama-cpp}"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-/usr/local/cuda-12.4/lib64}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_CPP_BUILD_DIR/bin/llama-server}"

mkdir -p "$OUTPUT_DIR/server"

if [[ -f /root/miniconda3/bin/activate && -d "$LLAMA_CPP_CONDA_ENV" ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/bin/activate "$LLAMA_CPP_CONDA_ENV"
fi

export LD_LIBRARY_PATH="$LLAMA_CPP_BUILD_DIR/bin:$CUDA_LIB_DIR:${LD_LIBRARY_PATH:-}"

if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "missing llama-server binary: $LLAMA_SERVER" >&2
  exit 10
fi

if [[ ! -f "$MODEL" ]]; then
  echo "missing model file: $MODEL" >&2
  exit 11
fi

KV_ARGS=(--kv-offload)
if [[ "$KV_OFFLOAD" == "cpu" || "$KV_OFFLOAD" == "false" || "$KV_OFFLOAD" == "0" ]]; then
  KV_ARGS=(--no-kv-offload)
fi

read -r -a MOE_ARGS <<< "$MOE_PLACEMENT"

SERVER_CMD=(
  "$LLAMA_SERVER"
  -m "$MODEL"
  -ngl "$GPU_LAYERS"
  "${MOE_ARGS[@]}"
  "${KV_ARGS[@]}"
  -ctk "$CACHE_TYPE_K"
  -ctv "$CACHE_TYPE_V"
  -fa "$FLASH_ATTN"
  -c "$CTX_SIZE"
  -np "$PARALLEL"
  -b "$BATCH_SIZE"
  -ub "$UBATCH_SIZE"
  --metrics
  --slots
  --host "$HOST"
  --port "$PORT"
  --log-file "$OUTPUT_DIR/server/server.log"
)

{
  printf '%q ' "${SERVER_CMD[@]}"
  printf '\n'
} > "$OUTPUT_DIR/server/server_start_command.sh"
chmod +x "$OUTPUT_DIR/server/server_start_command.sh"

"${SERVER_CMD[@]}" > "$OUTPUT_DIR/server/stdout.log" 2> "$OUTPUT_DIR/server/stderr.log" &
SERVER_PID=$!
echo "$SERVER_PID" > "$OUTPUT_DIR/server/server.pid"

for _ in $(seq 1 900); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "llama-server exited during startup" >&2
    cat "$OUTPUT_DIR/server/stderr.log" >&2 || true
    exit 12
  fi
  if curl -fsS "http://$HOST:$PORT/health" > "$OUTPUT_DIR/server/server_health.json" 2>/dev/null; then
    echo "server_ready=1"
    echo "server_pid=$SERVER_PID"
    echo "server_url=http://$HOST:$PORT"
    exit 0
  fi
  sleep 1
done

echo "timed out waiting for llama-server health endpoint" >&2
exit 13
