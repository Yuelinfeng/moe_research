# llama.cpp Mixtral Q8_0 + LMSYS-Chat-1M Experiment

This directory contains the experiment harness for measuring coupled MoE expert-weight and KV Cache memory pressure on a single 24 GB GPU.

Primary entry points:

```bash
bash experiments/llama_cpp_mixtral_lmsys_1m/run_stage_a_smoke.sh --output-dir "$RUN_DIR"
python experiments/llama_cpp_mixtral_lmsys_1m/prepare_lmsys_chat1m.py --output-dir /root/autodl-tmp/datasets/lmsys-chat-1m-prepared
python experiments/llama_cpp_mixtral_lmsys_1m/run_lmsys_1m_client.py --shard-dir /root/autodl-tmp/datasets/lmsys-chat-1m-prepared/shards --output-dir "$RUN_DIR/outputs"
```

Stage A downloads or reuses the Mixtral Q8_0 GGUF, starts `llama-server`, polls `/metrics` and `/slots`, sends a few smoke requests, and writes a local summary.

The full 1M run should only be started after Stage A and the 100/1k/10k calibration stages are stable.
