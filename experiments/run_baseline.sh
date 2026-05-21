#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  echo "usage: run_baseline.sh --output-dir <dir>" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

ROOT="${REPO_PATH:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x /root/miniconda3/envs/deepseek_moe/bin/python ]]; then
    PYTHON_BIN="/root/miniconda3/envs/deepseek_moe/bin/python"
  elif [[ -x /root/miniconda3/bin/python ]]; then
    PYTHON_BIN="/root/miniconda3/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "no usable Python found" | tee "$OUTPUT_DIR/error.txt" >&2
    exit 40
  fi
fi

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/pip-cache}"
export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$PIP_CACHE_DIR" "$TMPDIR"

{
  echo "python=$PYTHON_BIN"
  echo "baseline_mode=${BASELINE_MODE:-hf}"
  echo "model_id=${MODEL_ID:-Qwen/Qwen1.5-MoE-A2.7B-Chat}"
  echo "hf_endpoint=${HF_ENDPOINT:-https://hf-mirror.com}"
  echo "require_cuda=${MOE_REQUIRE_CUDA:-1}"
  echo "hf_home=$HF_HOME"
  echo "tmpdir=$TMPDIR"
} > "$OUTPUT_DIR/baseline_env.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$OUTPUT_DIR/nvidia_smi.txt" 2>&1 || true
else
  echo "nvidia-smi not found" > "$OUTPUT_DIR/nvidia_smi.txt"
fi

"$PYTHON_BIN" "$ROOT/experiments/moe_inference_baseline.py" \
  --output-dir "$OUTPUT_DIR" \
  --mode "${BASELINE_MODE:-hf}" \
  --model-id "${MODEL_ID:-Qwen/Qwen1.5-MoE-A2.7B-Chat}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-32}" \
  --warmup-iters "${WARMUP_ITERS:-1}" \
  --benchmark-iters "${BENCHMARK_ITERS:-3}" \
  --dtype "${MODEL_DTYPE:-float16}" \
  --hf-placement "${HF_PLACEMENT:-auto}" \
  --cuda-max-memory "${CUDA_MAX_MEMORY:-12GiB}" \
  --cpu-max-memory "${CPU_MAX_MEMORY:-38GiB}" \
  --offload-dir "${OFFLOAD_DIR:-${OUTPUT_DIR}/offload}" \
  --require-cuda "${MOE_REQUIRE_CUDA:-1}" \
  --synthetic-batch-size "${SYNTHETIC_BATCH_SIZE:-1}" \
  --synthetic-seq-len "${SYNTHETIC_SEQ_LEN:-64}" \
  --synthetic-vocab-size "${SYNTHETIC_VOCAB_SIZE:-2048}" \
  --synthetic-hidden-size "${SYNTHETIC_HIDDEN_SIZE:-256}" \
  --synthetic-intermediate-size "${SYNTHETIC_INTERMEDIATE_SIZE:-512}" \
  --synthetic-num-layers "${SYNTHETIC_NUM_LAYERS:-4}" \
  --synthetic-num-experts "${SYNTHETIC_NUM_EXPERTS:-8}" \
  --synthetic-top-k "${SYNTHETIC_TOP_K:-2}"
