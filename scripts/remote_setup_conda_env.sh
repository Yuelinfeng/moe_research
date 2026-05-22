#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="${MOE_CONDA_PREFIX:-/root/autodl-tmp/conda-envs/moe-research}"
CONDA_BIN="${CONDA_BIN:-/root/miniconda3/bin/conda}"
BASE_PREFIX="${MOE_CONDA_BASE_PREFIX:-/root/miniconda3}"
PROXY_URL="${PROXY_URL:-http://127.0.0.1:7890}"

if [[ "${USE_PROXY:-0}" == "1" ]]; then
  export http_proxy="$PROXY_URL"
  export https_proxy="$PROXY_URL"
  export HTTP_PROXY="$PROXY_URL"
  export HTTPS_PROXY="$PROXY_URL"
fi

if [[ "${MOE_RECREATE_ENV:-0}" == "1" && -d "$ENV_PREFIX" ]]; then
  case "$ENV_PREFIX" in
    /root/autodl-tmp/conda-envs/moe-research)
      rm -rf "$ENV_PREFIX"
      ;;
    *)
      echo "refusing to remove unexpected env path: $ENV_PREFIX" >&2
      exit 61
      ;;
  esac
fi

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "conda not found: $CONDA_BIN" >&2
  exit 60
fi

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  mkdir -p "$(dirname "$ENV_PREFIX")"
  "$CONDA_BIN" create -y -p "$ENV_PREFIX" --clone "$BASE_PREFIX"
fi

PYTHON_BIN="$ENV_PREFIX/bin/python"
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install \
  "transformers>=4.41,<4.42" \
  "accelerate>=0.30" \
  "sentencepiece" \
  "safetensors" \
  "huggingface_hub" \
  "protobuf"

"$PYTHON_BIN" - <<'PY'
import importlib.util as u
import sys

required = ["torch", "transformers", "accelerate", "sentencepiece", "safetensors", "huggingface_hub"]
missing = [name for name in required if u.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing packages: {missing}")

import torch

print("python", sys.executable)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))
PY
