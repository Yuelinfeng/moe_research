# llama.cpp Mixtral Q8_0 + LMSYS-Chat-1M Experiment

This directory contains the experiment harness for measuring coupled MoE expert-weight and KV Cache memory pressure on a single 24 GB GPU.

Primary entry points:

```bash
bash experiments/llama_cpp_mixtral_lmsys_1m/run_stage_a_smoke.sh --output-dir "$RUN_DIR"
bash experiments/llama_cpp_mixtral_lmsys_1m/run_stage_b_100_sweep.sh --output-dir "$RUN_DIR"
python experiments/llama_cpp_mixtral_lmsys_1m/prepare_lmsys_chat1m.py --output-dir /root/autodl-tmp/datasets/lmsys-chat-1m-prepared
python experiments/llama_cpp_mixtral_lmsys_1m/run_lmsys_1m_client.py --shard-dir /root/autodl-tmp/datasets/lmsys-chat-1m-prepared/shards --output-dir "$RUN_DIR/outputs"
```

Stage A downloads or reuses the Mixtral Q8_0 GGUF, starts `llama-server`, polls `/metrics` and `/slots`, sends a few smoke requests, and writes a local summary.

Stage B prepares a fixed 100-prompt LMSYS sample and runs the default Expert/KV sweep across GPU KV, CPU KV, KV q8, and the first `-np 2` pressure points.

For KV pressure rather than batch-one baseline, run Stage B with the pressure config and longest-prompt selection:

```bash
env DATASET_DIR=/root/autodl-tmp/datasets/lmsys-chat-1m-stage-b-longest-100 \
  DATASET_SELECT_MODE=longest \
  DATASET_SCAN_LIMIT=20000 \
  DATASET_FORCE_REBUILD=1 \
  CONFIGS_FILE=experiments/llama_cpp_mixtral_lmsys_1m/stage_b_kv_pressure.config \
  bash experiments/llama_cpp_mixtral_lmsys_1m/run_stage_b_100_sweep.sh --output-dir "$RUN_DIR"
```

The full 1M run should only be started after Stage A and the 100/1k/10k calibration stages are stable.
