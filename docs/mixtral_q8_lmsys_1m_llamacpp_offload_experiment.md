# Mixtral Q8_0 + LMSYS-Chat-1M llama.cpp Expert/KV Offload Experiment Design

## 1. Objective

Design and run a real large-scale MoE inference/offloading experiment on the current single-GPU server, focused on the coupled memory pressure from MoE expert weights and KV Cache under high-throughput serving:

```text
GPU: NVIDIA GeForce RTX 4090D/4090 class, 24 GB VRAM
RAM: about 60 GB usable system memory
Runtime: llama.cpp CUDA build
Model: mixtral-8x7b-instruct-v0.1.Q8_0.gguf
Dataset: LMSYS-Chat-1M, full 1,000,000 conversations
```

The central research question is:

```text
Under large-batch / high-throughput inference, is GPU memory dominated by MoE expert
weights or KV Cache, where is the tipping point, and how do expert/KV offload strategies
affect latency, throughput, GPU memory, CPU memory, and stability?
```

This is not a dense-model benchmark. The target model is large enough that a full-GPU run is expected to fail on 24 GB VRAM, and high-throughput serving can additionally make KV Cache large enough to compete with expert weights for GPU memory.

The target analysis artifact is a set of plots and tables showing:

```text
active_expert_weight_bytes / kv_cache_bytes
gpu_expert_weight_bytes / gpu_kv_cache_bytes
latency and throughput as the above ratio changes
tipping point where KV Cache becomes the dominant incremental GPU-memory consumer
```

## 2. Sources and Known Facts

### 2.1 Model

Target file:

```text
TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF
mixtral-8x7b-instruct-v0.1.Q8_0.gguf
```

Verified file size by Hugging Face file headers:

```text
Q8_0 bytes: 49,624,262,592 bytes
Q8_0 size: 49.62 GB decimal, about 46.22 GiB
```

For comparison:

```text
Q5_K_M bytes: 32,229,279,680 bytes
Q4_K_M bytes: 26,441,533,376 bytes
```

Research implication:

```text
Q8_0 is too large for 24 GB VRAM.
It must rely on CPU RAM plus partial GPU offload.
Under high concurrency, KV Cache also consumes GPU memory proportional to live tokens.
The experiment must measure expert-weight pressure and KV pressure separately.
```

### 2.2 Dataset

Dataset:

```text
lmsys/lmsys-chat-1m
```

Verified Hugging Face dataset metadata:

```text
num_examples: 1,000,000
download_size: about 1.49 GB
dataset_size: about 2.63 GB
format: parquet
task category: conversational
features: conversation_id, model, conversation, turn, language, openai_moderation, redacted
```

The dataset is gated. Before running on the remote server, the Hugging Face account used by the server must accept the LMSYS-Chat-1M Dataset License Agreement.

### 2.3 llama.cpp Offload Controls

The current llama.cpp server documentation exposes the relevant controls:

```text
-ngl, --gpu-layers, --n-gpu-layers N
-cmoe, --cpu-moe
-ncmoe, --n-cpu-moe N
-kvo, --kv-offload
-nkvo, --no-kv-offload
-ctk, --cache-type-k TYPE
-ctv, --cache-type-v TYPE
-np, --parallel N
-b, --batch-size N
-ub, --ubatch-size N
-fa, --flash-attn [on|off|auto]
--metrics
--slots
```

Important semantic boundary:

```text
llama.cpp -cmoe / -ncmoe is static MoE expert weight placement.
It is not a dynamic expert cache or token-level expert prefetch system.

llama.cpp --kv-offload means KQV operations and KV Cache are kept on the GPU.
llama.cpp --no-kv-offload moves KV-related work/cache away from GPU, trading VRAM for latency.

Without llama.cpp routing instrumentation, "active expert weight bytes" is an estimate.
With routing instrumentation, it can be measured from per-layer expert ids.
```

That is still a real MoE/KV offload experiment because the model is genuinely MoE, Q8_0 cannot fit entirely in GPU memory, and high-throughput serving creates a real KV Cache placement problem.

## 3. Experiment Strategy

The full 1M run should not be the first run. The correct execution order is:

```text
Stage A: model download and server smoke test
Stage B: expert/KV offload feasibility sweep on 100 prompts
Stage C: high-throughput calibration on 1,000 and 10,000 prompts
Stage D: full 1,000,000 prompt run using the selected stable configuration
Stage E: expert-weight/KV-cache ratio analysis and report generation
```

Only Stage D runs the full dataset. Running all offload configurations over 1M prompts is technically possible but inefficient on a single 4090D; it should only be done after the sampled sweep identifies configurations worth the cost.

For exact expert activation analysis, run a separate instrumented sample after Stage C:

```text
1,000 to 10,000 prompts
same serving parameters as the selected full-run config
record per-layer unique expert ids and top-k expert counts
```

The full 1M run should prioritize stability and throughput; the instrumented run should prioritize measurement detail.

## 4. Workload Definition

Each LMSYS row is converted into one benchmark request.

Default request construction:

```text
system: You are a helpful assistant.
user: first user message from conversation
```

If the first user message is missing, skip the row and record it in `skipped.jsonl`.

Recommended generation limits for the full 1M run:

```text
max_new_tokens: 64
ctx_size: 2048 for baseline, 4096/8192 for KV-pressure calibration
input truncation: keep the first ctx_size - max_new_tokens - template_margin prompt tokens
temperature: 0.0 for deterministic latency measurement
top_p: 1.0
seed: fixed
```

Rationale:

```text
The objective is offload/runtime analysis, not open-ended answer quality.
Using fixed short generation makes 1M requests feasible and comparable.
High-throughput calibration must vary live tokens through -np and ctx_size.
```

Optional quality run:

```text
Run a separate 1,000 prompt sample with max_new_tokens=256 and temperature=0.7.
Do not mix this with the latency/offload benchmark.
```

## 5. Experimental Variables

### 5.1 MoE Expert Placement

Use this sweep for Stage B/C:

```text
S0_full_gpu_probe:
  args: -ngl 99 -fit off
  expected: fail or OOM
  purpose: prove Q8_0 requires offload on 24 GB VRAM

S1_auto_fit:
  args: -ngl 99 -fit on
  expected: llama.cpp adjusts placement if possible
  purpose: understand llama.cpp automatic fit behavior

S2_all_moe_cpu:
  args: -ngl 99 -cmoe
  expected: stable, lower VRAM, more CPU/RAM pressure
  purpose: conservative offload baseline

S3_first_32_moe_cpu:
  args: -ngl 99 -ncmoe 32
  expected: similar to all MoE CPU for a 32-layer Mixtral
  purpose: explicit layer-count equivalent to all expert CPU placement

S4_first_24_moe_cpu:
  args: -ngl 99 -ncmoe 24
  expected: higher VRAM, faster than S2 if it fits
  purpose: partial MoE placement

S5_first_16_moe_cpu:
  args: -ngl 99 -ncmoe 16
  expected: possible OOM or better throughput if it fits
  purpose: find memory/performance boundary

S6_first_8_moe_cpu:
  args: -ngl 99 -ncmoe 8
  expected: likely high VRAM pressure
  purpose: stress GPU-resident expert strategy

S7_no_moe_cpu:
  args: -ngl 99
  expected: likely OOM for Q8_0
  purpose: negative control
```

If `-ngl 99` behaves differently from `-ngl all` in the current build, keep `-ngl 99` for reproducibility because Mixtral has fewer than 99 layers.

### 5.2 KV Cache Placement and Precision

KV Cache placement is a primary experimental variable, not only a runtime detail.

```text
K0_gpu_kv_f16:
  args: --kv-offload -ctk f16 -ctv f16
  expected: fastest KV path, highest GPU KV memory
  purpose: default high-throughput GPU-KV baseline

K1_cpu_kv_f16:
  args: --no-kv-offload -ctk f16 -ctv f16
  expected: lower GPU memory, higher latency
  purpose: test whether KV must be moved to CPU memory under large batch

K2_gpu_kv_q8:
  args: --kv-offload -ctk q8_0 -ctv q8_0
  expected: lower GPU KV memory than f16, possible quality/perf trade-off
  purpose: KV quantization baseline

K3_gpu_kv_q4:
  args: --kv-offload -ctk q4_0 -ctv q4_0
  expected: lowest GPU KV memory among GPU-KV configs, possible accuracy/perf risk
  purpose: stress memory reduction boundary
```

Use `K1_cpu_kv_f16` only after confirming server stability. CPU KV can become a latency bottleneck.

### 5.3 High-Throughput Pressure Variables

The high-throughput pressure sweep controls how much KV Cache exists at once:

```text
ctx_size: 2048, 4096, 8192
parallel slots: -np 1, -np 2, -np 4, -np 8
batch size: -b 512, -b 1024, -b 2048
ubatch size: -ub 256, -ub 512
flash attention: -fa auto
```

The approximate live-token capacity is:

```text
live_tokens_max ~= ctx_size * parallel_slots
```

The actual live-token count should be sampled from `/slots` and from request-level prompt/completion token counts.

Do not increase `-np` until memory behavior is stable. Each parallel slot increases KV Cache and scheduler pressure.

### 5.4 Derived Memory Ratio Variables

The main plot should use these derived metrics:

```text
kv_cache_bytes_estimated
active_expert_weight_bytes_estimated
active_expert_weight_bytes_measured
gpu_expert_weight_bytes_estimated
gpu_kv_cache_bytes_estimated
active_expert_to_kv_ratio
gpu_expert_to_gpu_kv_ratio
```

Default KV estimate:

```text
kv_cache_bytes_estimated =
  live_tokens * n_layers * n_kv_heads * head_dim * 2(K,V) * bytes_per_kv_element
```

For Mixtral-like 32-layer, hidden-size 4096, 32-head attention with f16 KV, this is approximately:

```text
about 128 KiB per live token
4,096 live tokens  -> about 512 MiB KV
32,768 live tokens -> about 4 GiB KV
131,072 live tokens -> about 16 GiB KV
```

Active expert weight estimate has two modes:

```text
token_path_active:
  top_k experts per layer for a single token path
  useful for theoretical active-parameter comparison

batch_unique_active:
  unique experts touched by all live tokens in a layer/window
  useful for high-throughput memory pressure
```

For high-batch serving, `batch_unique_active` is the relevant quantity. It can approach all experts per layer even though each token only uses top-k experts.

## 6. Recommended Final Full-1M Configuration

Start with the most conservative stable configuration for model feasibility:

```bash
export LD_LIBRARY_PATH=/root/autodl-tmp/llama.cpp/build-cuda-env/bin:/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}

MODEL=/root/autodl-tmp/llama-models/mixtral-8x7b-instruct-v0.1/mixtral-8x7b-instruct-v0.1.Q8_0.gguf
LLAMA_SERVER=/root/autodl-tmp/llama.cpp/build-cuda-env/bin/llama-server

$LLAMA_SERVER \
  -m "$MODEL" \
  -ngl 99 \
  -cmoe \
  --kv-offload \
  -ctk f16 \
  -ctv f16 \
  -fa auto \
  -c 2048 \
  -np 1 \
  -b 512 \
  -ub 256 \
  --metrics \
  --host 127.0.0.1 \
  --port 8080 \
  --log-file /root/autodl-tmp/runs/mixtral_q8_lmsys_1m/server.log
```

Then increase pressure to expose the expert/KV trade-off:

```bash
-ncmoe 24
-np 2
-c 4096
-ctk q8_0
-ctv q8_0
-ub 512
```

The full 1M run should use exactly one selected configuration, stored in `run_config.json`, so the result is reproducible. Stage B/C can test many configurations; Stage D should not.

## 7. Required New Files

Recommended implementation files:

```text
configs/mixtral_q8_lmsys_1m.example.json
experiments/llama_cpp_mixtral_lmsys_1m/prepare_lmsys_chat1m.py
experiments/llama_cpp_mixtral_lmsys_1m/run_llama_server.sh
experiments/llama_cpp_mixtral_lmsys_1m/run_lmsys_1m_client.py
experiments/llama_cpp_mixtral_lmsys_1m/monitor_system.sh
experiments/llama_cpp_mixtral_lmsys_1m/collect_llama_metrics.py
experiments/llama_cpp_mixtral_lmsys_1m/summarize_results.py
experiments/llama_cpp_mixtral_lmsys_1m/plot_expert_kv_ratio.py
experiments/llama_cpp_mixtral_lmsys_1m/llama_cpp_moe_routing_instrumentation.md
experiments/llama_cpp_mixtral_lmsys_1m/README.md
```

### 7.1 `prepare_lmsys_chat1m.py`

Responsibilities:

```text
1. Download/load lmsys/lmsys-chat-1m after HF access is accepted.
2. Extract one prompt per row.
3. Filter or mark redacted rows.
4. Estimate prompt length before llama.cpp tokenization.
5. Write sharded JSONL files.
```

Output:

```text
dataset_manifest.json
shards/shard_000000.jsonl
shards/shard_000001.jsonl
...
skipped.jsonl
```

JSONL schema:

```json
{
  "request_id": "000000123456",
  "conversation_id": "...",
  "language": "English",
  "turn": 1,
  "model_in_lmsys": "gpt-4",
  "prompt_sha256": "...",
  "prompt": "...",
  "redacted": false
}
```

Shard size:

```text
10,000 requests per shard
100 shards for 1,000,000 rows
```

### 7.2 `run_llama_server.sh`

Responsibilities:

```text
1. Activate the llama.cpp conda environment.
2. Export LD_LIBRARY_PATH.
3. Start llama-server with the selected offload configuration.
4. Write server PID.
5. Write server logs.
6. Wait until /health is ready.
```

It must write:

```text
server.pid
server.log
server_start_command.sh
server_health.json
```

### 7.3 `run_lmsys_1m_client.py`

Responsibilities:

```text
1. Read sharded prompts.
2. Send requests to llama-server using OpenAI-compatible /v1/chat/completions.
3. Support resume by skipping existing request_id.
4. Record non-streaming timings from API response when available.
5. Optionally use streaming to measure time-to-first-token.
6. Poll `/slots` around each request batch to capture live-token pressure.
7. Write append-only compressed JSONL output.
```

Recommended client behavior:

```text
concurrency = 1 for Stage B
concurrency = -np for Stage C/D
timeout = 300 seconds per request
max_retries = 3
retry backoff = 2, 8, 30 seconds
```

Output schema:

```json
{
  "request_id": "000000123456",
  "status": "ok",
  "prompt_tokens": 128,
  "completion_tokens": 64,
  "total_tokens": 192,
  "slot_id": 0,
  "live_tokens_before": 1024,
  "live_tokens_after": 1088,
  "latency_ms": 1234.5,
  "ttft_ms": 321.0,
  "tokens_per_second": 51.9,
  "timings": {},
  "output_sha256": "...",
  "output_text": "..."
}
```

For privacy-aware analysis, allow:

```text
--store-output-text false
```

When disabled, store only `output_sha256`, token counts, and timings.

### 7.4 `monitor_system.sh`

Sample every 1 second:

```text
timestamp
gpu_util
gpu_mem_used_mb
gpu_mem_free_mb
gpu_power_w
gpu_pstate
cpu_percent
ram_used_mb
ram_available_mb
swap_used_mb
disk_read_mb_s
disk_write_mb_s
llama_server_rss_mb
llama_server_vms_mb
num_llama_threads
num_open_fds
```

Output:

```text
system_samples.csv
```

### 7.5 `collect_llama_metrics.py`

Responsibilities:

```text
1. Poll `/metrics` when enabled.
2. Poll `/slots` when enabled.
3. Save live slot state, prompt tokens, predicted tokens, and cache usage when available.
4. Normalize metrics into a time-series CSV for ratio analysis.
```

Output:

```text
llama_metrics_samples.jsonl
llama_slots_samples.jsonl
llama_slots_summary.csv
```

Required fields when available:

```text
timestamp
slot_id
slot_state
slot_prompt_tokens
slot_predicted_tokens
slot_n_ctx
slot_n_past
slot_cache_tokens
server_prompt_tokens_total
server_generation_tokens_total
```

### 7.6 `summarize_results.py`

Compute:

```text
num_requests_total
num_requests_ok
num_requests_failed
wall_time_hours
requests_per_second
prompt_tokens_per_second
generation_tokens_per_second
total_tokens_per_second
latency_p50_ms
latency_p95_ms
latency_p99_ms
ttft_p50_ms
ttft_p95_ms
ttft_p99_ms
gpu_mem_peak_mb
ram_peak_mb
kv_cache_bytes_estimated_peak
gpu_kv_cache_bytes_estimated_peak
active_expert_weight_bytes_estimated_peak
active_expert_weight_bytes_measured_peak
active_expert_to_kv_ratio_p50
active_expert_to_kv_ratio_p95
gpu_expert_to_gpu_kv_ratio_p50
gpu_expert_to_gpu_kv_ratio_p95
gpu_util_avg
gpu_util_p95
power_avg_w
estimated_energy_wh
```

Group by:

```text
offload_config
language
prompt_length_bucket
live_token_bucket
ctx_size
parallel_slots
kv_offload
cache_type_k
cache_type_v
moe_cpu_layers
turn
redacted
```

### 7.7 `plot_expert_kv_ratio.py`

Generate:

```text
expert_kv_ratio_vs_live_tokens.png
expert_kv_ratio_vs_batch_size.png
gpu_memory_breakdown.png
latency_vs_expert_kv_ratio.png
throughput_vs_expert_kv_ratio.png
tipping_point_summary.csv
```

The main plot should have:

```text
x-axis: live tokens or effective batch pressure
y-axis: active_expert_weight_bytes / kv_cache_bytes
series: offload_config + kv_config
markers: OOM, swap use, server restart, p95 latency spike
```

### 7.8 `llama_cpp_moe_routing_instrumentation.md`

Document the optional llama.cpp patch needed for measured expert activation:

```text
per request id
decode step
layer id
token count
top-k expert ids
unique expert ids per layer
unique expert count per layer
estimated bytes for unique active experts
```

Without this patch, use estimated active expert bytes and label plots as estimates.

## 8. Result Directory Layout

Use one top-level directory per experiment run:

```text
/root/autodl-tmp/runs/mixtral_q8_lmsys_1m/
  run_config.json
  environment.json
  model/
    model_path.txt
    model_file_sha256.txt
    model_file_size.txt
  dataset/
    dataset_manifest.json
    skipped.jsonl
    shards/
  server/
    server_start_command.sh
    server.log
    server.pid
    server_health.json
  monitor/
    system_samples.csv
    llama_metrics_samples.jsonl
    llama_slots_samples.jsonl
    llama_slots_summary.csv
    moe_routing_samples.jsonl
  outputs/
    shard_000000.results.jsonl.gz
    shard_000001.results.jsonl.gz
    ...
  checkpoints/
    completed_request_ids.txt
    shard_status.json
  summaries/
    summary.json
    summary.csv
    latency_by_bucket.csv
    expert_kv_ratio_by_bucket.csv
    tipping_point_summary.csv
    failure_report.md
  plots/
    expert_kv_ratio_vs_live_tokens.png
    gpu_memory_breakdown.png
    latency_vs_expert_kv_ratio.png
```

Downloaded local mirror:

```text
D:\moe_research\results\mixtral_q8_lmsys_1m\<run_id>\
```

Do not commit model files, dataset files, or raw full outputs.

## 9. Metrics and Analysis

### 9.1 Fit and Stability Metrics

For each Stage B/C configuration:

```text
does_server_start
does_first_request_succeed
exit_code
startup_seconds
load_seconds
peak_vram_mb
peak_ram_mb
swap_used_mb
peak_live_tokens
peak_kv_cache_bytes_estimated
peak_active_expert_weight_bytes_estimated
peak_gpu_expert_weight_bytes_estimated
oom_events
server_restarts
```

### 9.2 Performance Metrics

Primary:

```text
requests_per_second
generation_tokens_per_second
prompt_tokens_per_second
latency_p50/p95/p99
ttft_p50/p95/p99
```

Secondary:

```text
GPU utilization distribution
GPU memory distribution
CPU RAM distribution
KV Cache estimated memory distribution
active expert weight estimated/measured distribution
power and energy
failure/retry rate
```

### 9.3 Expert/KV Memory Metrics

Record and derive:

```text
live_tokens_current
live_tokens_peak
kv_cache_bytes_estimated_current
kv_cache_bytes_estimated_peak
kv_cache_location: gpu or cpu
kv_cache_type_k
kv_cache_type_v
moe_expert_location_strategy
moe_cpu_layers
gpu_expert_weight_bytes_estimated
active_expert_weight_bytes_estimated
active_expert_weight_bytes_measured
active_expert_to_kv_ratio
gpu_expert_to_gpu_kv_ratio
```

Ratio definitions:

```text
active_expert_to_kv_ratio =
  active_expert_weight_bytes / kv_cache_bytes

gpu_expert_to_gpu_kv_ratio =
  gpu_resident_expert_weight_bytes / gpu_resident_kv_cache_bytes
```

The first ratio answers "how much routed expert weight is activated compared with KV generated by the batch".

The second ratio answers "what is actually competing for GPU memory".

### 9.4 Offload Interpretation

Expected trade-off:

```text
More MoE experts on CPU:
  lower VRAM pressure
  higher CPU/RAM bandwidth pressure
  lower risk of OOM
  usually lower generation throughput

More MoE experts on GPU:
  higher VRAM pressure
  potentially higher throughput
  higher risk of OOM or reduced context/parallelism

More KV Cache on GPU:
  faster attention/KQV path
  higher VRAM pressure as live tokens increase
  stronger interaction with -np and ctx_size

More KV Cache on CPU:
  lower GPU memory pressure
  higher CPU/RAM bandwidth pressure
  higher TTFT and decode latency risk
```

Useful plots:

```text
latency_p95 vs peak_vram_mb
generation_tok_s vs peak_ram_mb
gpu_util over time
ram_used over time
kv_cache_bytes vs live_tokens
active_expert_weight_bytes vs live_tokens
active_expert_weight_bytes / kv_cache_bytes vs live_tokens
gpu_expert_weight_bytes / gpu_kv_cache_bytes vs live_tokens
requests_per_second by prompt_length_bucket
failure_rate by prompt_length_bucket
```

## 10. Full 1M Runtime Risk

This experiment is large. On one 4090D with Q8_0 and CPU MoE offload, full 1M generation may take days or longer depending on:

```text
max_new_tokens
average prompt length
selected -cmoe/-ncmoe strategy
selected --kv-offload/--no-kv-offload strategy
KV cache type f16/q8_0/q4_0
parallel slots
CPU memory bandwidth
whether the model page-faults from mmap
```

Therefore the full run must be:

```text
append-only
resumable
sharded
monitored
safe to stop and restart
```

The run is considered valid only if:

```text
1. All 1,000,000 rows are either completed or explicitly recorded as skipped/failed.
2. Failure reasons are categorized.
3. The server command and commit/build info are saved.
4. The exact model file size and hash are saved.
5. System monitoring covers the full wall-clock run.
6. KV Cache estimates and live-token samples cover the full wall-clock run.
7. Expert/KV ratio plots clearly state whether expert activation is measured or estimated.
```

## 11. Stage Plan

### Stage A: Download and Smoke

Tasks:

```text
1. Download Q8_0 GGUF to /root/autodl-tmp/llama-models/mixtral-8x7b-instruct-v0.1/.
2. Compute sha256sum.
3. Start llama-server with -cmoe.
4. Send 3 simple requests.
5. Verify nonzero GPU utilization during generation.
```

Acceptance:

```text
server starts
3/3 requests succeed
GPU memory and RAM are sampled
server log contains CUDA backend information
```

### Stage B: 100-Prompt Expert/KV Feasibility Sweep

Run expert placement and KV placement strategies on the same 100 prompts.

Acceptance:

```text
At least one offload strategy completes 100/100 requests.
S0/S7 failures are saved if they fail.
Peak VRAM/RAM/KV estimate and latency are available for every completed strategy.
At least one --kv-offload and one --no-kv-offload configuration is tested if memory permits.
```

### Stage C: 1k and 10k High-Throughput Calibration

Run only the top 2 stable expert/KV strategies from Stage B, then sweep `-np`, `ctx_size`, `-b`, and `-ub`.

Acceptance:

```text
1k run completes without restart.
10k run completes without restart.
Projected full-1M runtime is estimated.
Final full-run config is selected.
Estimated expert/KV ratio curve is generated.
The live-token point where KV starts to dominate incremental GPU memory is identified or bounded.
```

### Stage D: Full 1M Run

Run selected config over all shards.

Acceptance:

```text
completed + skipped + failed = 1,000,000
summary.json generated
monitoring file covers the complete run
llama metrics and slots samples cover the complete run
no unclassified failures remain
```

### Stage E: Expert/KV Ratio Analysis Report

Produce:

```text
reports/mixtral_q8_lmsys_1m_offload_report.md
results/.../summaries/summary.json
results/.../summaries/summary.csv
results/.../summaries/expert_kv_ratio_by_bucket.csv
results/.../plots/expert_kv_ratio_vs_live_tokens.png
```

Report sections:

```text
environment
model and dataset
offload strategies tested
selected full-run config
latency and throughput
memory behavior
expert weight vs KV Cache ratio
high-throughput tipping point
failure modes
interpretation
next optimization direction
```

## 12. Immediate Next Step

Implement the experiment harness in this order:

```text
1. configs/mixtral_q8_lmsys_1m.example.json
2. prepare_lmsys_chat1m.py
3. run_llama_server.sh
4. run_lmsys_1m_client.py
5. monitor_system.sh
6. collect_llama_metrics.py
7. summarize_results.py
8. plot_expert_kv_ratio.py
9. optional llama.cpp MoE routing instrumentation note
```

Then run Stage A on the remote server before attempting any dataset-scale run.

## 13. References

```text
Mixtral paper: https://arxiv.org/abs/2401.04088
Mixtral HF model: https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1
Mixtral GGUF files: https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF
LMSYS-Chat-1M dataset: https://huggingface.co/datasets/lmsys/lmsys-chat-1m
llama.cpp server docs: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
```
