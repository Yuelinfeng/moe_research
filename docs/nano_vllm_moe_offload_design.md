# nano-vLLM MoE + Expert Offload 修改设计

## 1. 目标

目标是在远程 4090 服务器上的 `nano-vLLM` 中支持运行真实 MoE 模型，第一目标模型为：

```text
Qwen/Qwen1.5-MoE-A2.7B-Chat
```

该模型的 Hugging Face 配置要点：

```text
model_type = qwen2_moe
architectures = Qwen2MoeForCausalLM
num_hidden_layers = 24
hidden_size = 2048
num_experts = 60
num_experts_per_tok = 4
moe_intermediate_size = 1408
shared_expert_intermediate_size = 5632
```

本设计的第一阶段目标不是马上实现最优性能，而是先实现：

1. nano-vLLM 能识别并构造 Qwen2-MoE 模型。
2. 非 expert 权重常驻 GPU。
3. expert 权重主要驻留 CPU，需要时按需搬到 GPU。
4. 推理结果可跑通，并能记录 expert 命中、miss、迁移时间和延迟。

第一版约束：

```text
tensor_parallel_size = 1
enforce_eager = True
CUDA graph disabled
single GPU RTX 4090
model = Qwen/Qwen1.5-MoE-A2.7B-Chat
```

原因：expert offload 会引入动态 CPU/GPU 权重迁移，和当前 nano-vLLM 的 CUDA graph 静态假设冲突。先做 eager 模式可以把“模型正确性”和“offload 策略效果”分开验证。

## 2. 当前 nano-vLLM 代码限制

当前远程代码路径：

```text
/root/autodl-tmp/nano-vllm
```

已验证当前 commit：

```text
bb823b3
```

当前实现的关键限制：

1. `nanovllm/engine/model_runner.py` 直接硬编码实例化 `Qwen3ForCausalLM`。
2. `nanovllm/models/qwen3.py` 只实现 dense Qwen3 结构，没有 MoE router、expert、shared expert。
3. `nanovllm/utils/loader.py` 会遍历 safetensors，把权重直接 copy 到模型参数中，不支持 expert 权重懒加载或 CPU/GPU 分层放置。
4. `nanovllm/layers/linear.py` 只有 dense linear 和 tensor-parallel linear，没有 MoE expert 权重封装。
5. `ModelRunner.capture_cudagraph()` 假设模型权重和执行图稳定，不适合第一版动态 expert offload。
6. `Scheduler` 只调度 token/KV cache，不知道 MoE expert cache，也不会暴露路由信息。

因此，MoE 支持至少需要新增模型结构、MoE 层、expert store/cache、offload 策略和 trace。

## 3. 总体架构

新增架构分为四层：

```text
LLM API
  -> ModelRunner
    -> model registry selects Qwen3 or Qwen2-MoE
      -> Qwen2MoeForCausalLM
        -> Qwen2MoeDecoderLayer
          -> attention
          -> Qwen2MoeSparseMoeBlock
            -> router gate
            -> ExpertCache.ensure_experts(...)
            -> Expert execution
            -> shared expert
```

expert 权重生命周期：

```text
safetensors
  -> ExpertStore loads expert weights to CPU pinned memory
  -> ExpertCache keeps limited experts on GPU
  -> MoE layer requests active experts from ExpertCache
  -> Cache miss triggers CPU -> GPU transfer
  -> Policy evicts unused GPU experts
```

第一版采用同步 on-demand：

```text
router selects top-k experts
  -> check GPU expert cache
  -> missing experts copied synchronously from CPU to GPU
  -> run selected experts
  -> record hit/miss/H2D time
```

后续再加异步 prefetch。

## 4. 新增文件

### 4.1 `nanovllm/models/qwen2_moe.py`

新增 Qwen2-MoE 模型结构。

主要类：

```python
class Qwen2MoeForCausalLM(nn.Module)
class Qwen2MoeModel(nn.Module)
class Qwen2MoeDecoderLayer(nn.Module)
class Qwen2MoeAttention(nn.Module)
```

职责：

1. 对齐 Hugging Face 权重命名：`model.layers.*.mlp.gate.weight`、`experts.*`、`shared_expert.*`。
2. 复用现有 Qwen3 attention、RMSNorm、rotary embedding、embedding/lm_head 逻辑。
3. 把原 dense `Qwen3MLP` 替换为 `Qwen2MoeSparseMoeBlock`。
4. 提供 `packed_modules_mapping`，支持 q/k/v 合并加载。
5. 暴露 `compute_logits()`，保持和现有 `ModelRunner` 兼容。

第一版不做 tensor parallel expert 切分。

### 4.2 `nanovllm/layers/moe.py`

实现 MoE block。

主要类：

```python
class Qwen2MoeSparseMoeBlock(nn.Module)
```

主要逻辑：

1. `router_logits = gate(hidden_states)`。
2. `topk_weight, topk_idx = torch.topk(router_logits, k=num_experts_per_tok)`。
3. 对 top-k 权重做 softmax 或按 Qwen 配置处理 `norm_topk_prob`。
4. 根据 `topk_idx` 聚合每个 expert 对应 token。
5. 调用 `ExpertCache.ensure_experts(layer_id, expert_ids)`。
6. 对每个 active expert 执行 expert MLP。
7. 按 router 权重 scatter-add 回 token 输出。
8. 加上 shared expert 输出：`shared_expert(hidden_states) * sigmoid(shared_expert_gate(hidden_states))`。

第一版允许 Python loop over active experts。性能不是最优，但便于验证 offload 正确性和 trace。

### 4.3 `nanovllm/layers/moe_expert.py`

封装单个 expert 的计算。

主要类：

```python
class MoeExpertWeights
class MoeExpertRuntime
```

职责：

1. 保存一个 expert 的三组权重：`gate_proj`、`up_proj`、`down_proj`。
2. 提供 `to_gpu()`、`to_cpu()`、`forward()`。
3. 支持 CPU pinned tensor 到 GPU tensor 的拷贝。
4. 避免把所有 experts 注册为普通 `nn.Parameter`，否则初始化时会占满 GPU。

expert 计算：

```python
hidden = silu(x @ gate_proj.T) * (x @ up_proj.T)
out = hidden @ down_proj.T
```

### 4.4 `nanovllm/offload/expert_store.py`

CPU expert 权重仓库。

主要类：

```python
@dataclass(frozen=True)
class ExpertKey:
    layer_id: int
    expert_id: int

class ExpertStore:
    def get_cpu_weights(key: ExpertKey) -> MoeExpertWeights
```

职责：

1. 建立 safetensors weight name 到文件路径的索引。
2. 按 `(layer_id, expert_id)` 懒加载 expert 权重。
3. 把 expert 权重保存在 CPU 内存，优先 pinned memory。
4. 提供统计信息：CPU resident experts、load count、load time。

权重名示例：

```text
model.layers.3.mlp.experts.0.gate_proj.weight
model.layers.3.mlp.experts.0.up_proj.weight
model.layers.3.mlp.experts.0.down_proj.weight
```

### 4.5 `nanovllm/offload/expert_cache.py`

GPU expert cache。

主要类：

```python
class ExpertCacheEntry
class ExpertCache:
    def ensure_experts(layer_id: int, expert_ids: torch.Tensor) -> dict[int, MoeExpertRuntime]
```

职责：

1. 维护 GPU 上当前驻留的 experts。
2. cache hit 时直接返回 GPU expert。
3. cache miss 时从 `ExpertStore` 拿 CPU 权重并拷贝到 GPU。
4. 按策略淘汰 GPU experts。
5. 记录 hit/miss、H2D latency、evict count、resident memory。

第一版建议配置：

```text
expert_cache_size = 120
```

含义：最多缓存 120 个 expert 实例。Qwen1.5-MoE 有 24 层 * 60 experts = 1440 experts，不能全部常驻 GPU。

### 4.6 `nanovllm/offload/policy.py`

offload 和淘汰策略。

主要类：

```python
class ExpertPolicy
class OnDemandLRUPolicy(ExpertPolicy)
class StaticHotExpertsPolicy(ExpertPolicy)
```

第一版只实现：

```text
on_demand_lru
```

后续扩展：

1. `static_hot`: 预先固定每层 top-N experts 常驻 GPU。
2. `oracle_trace`: 按离线 trace 预取。
3. `router_predict`: 利用上一 token 或上一层 router 预测。
4. `async_prefetch`: 在 attention 或前一层 MLP 期间异步搬运下一层 experts。

### 4.7 `nanovllm/offload/trace.py`

研究指标记录。

主要类：

```python
class MoeTraceRecorder
```

记录字段：

```text
request_id
step_id
layer_id
token_count
active_experts
topk_histogram
cache_hits
cache_misses
h2d_time_ms
expert_compute_time_ms
evicted_experts
resident_experts
gpu_memory_allocated_mb
```

输出文件：

```text
moe_trace.jsonl
moe_summary.json
```

这部分对研究很关键，因为只看 tokens/s 无法判断 offload 策略是否有效。

### 4.8 `nanovllm/utils/moe_loader.py`

MoE 专用加载器。

主要函数：

```python
def load_qwen2_moe_model(model, model_path, expert_store, config):
    ...
```

职责：

1. 加载非 expert 权重到模型参数。
2. 跳过 `model.layers.*.mlp.experts.*` 的普通参数加载。
3. 把 expert 权重文件位置注册到 `ExpertStore`。
4. 加载 router gate、shared expert、shared expert gate。

原因：现有 `load_model()` 会把所有 expert 权重都 copy 到模型里，不支持 offload。

### 4.9 `nanovllm/models/registry.py`

模型选择注册表。

主要函数：

```python
def get_model_cls(hf_config):
    if hf_config.model_type == "qwen3":
        return Qwen3ForCausalLM
    if hf_config.model_type == "qwen2_moe":
        return Qwen2MoeForCausalLM
```

原因：避免在 `ModelRunner` 里继续硬编码模型类。

### 4.10 `examples/qwen2_moe_offload.py`

最小运行示例。

用途：

1. 快速验证模型能加载。
2. 快速验证一条 prompt 能生成。
3. 输出 MoE trace 摘要。

示例参数：

```python
llm = LLM(
    "/root/autodl-tmp/hf-cache/transformers/models--Qwen--Qwen1.5-MoE-A2.7B-Chat/snapshots/...",
    enforce_eager=True,
    tensor_parallel_size=1,
    enable_moe_offload=True,
    expert_cache_size=120,
    expert_policy="on_demand_lru",
    max_model_len=1024,
    max_num_batched_tokens=1024,
)
```

### 4.11 `bench_moe_offload.py`

MoE offload benchmark。

输出：

```text
load_seconds
prefill_latency_ms
decode_latency_ms
tokens_per_second
cache_hit_rate
timely_hit_rate
h2d_time_ms
expert_compute_time_ms
peak_allocated_mb
resident_experts
```

注意：`cache_hit_rate` 不等于有效预取。后续做 prefetch 时要额外记录 timely utility。

### 4.12 `tests/test_qwen2_moe_shapes.py`

基础 shape 测试。

验证：

1. router top-k shape 正确。
2. expert dispatch/combine shape 正确。
3. shared expert 输出 shape 正确。
4. MoE block 输入输出 shape 保持 `[num_tokens, hidden_size]`。

### 4.13 `tests/test_expert_cache.py`

expert cache 测试。

验证：

1. miss 后加载。
2. hit 不重复加载。
3. 超过 cache size 后触发 eviction。
4. LRU 顺序符合预期。

## 5. 需要修改的现有文件

### 5.1 `nanovllm/config.py`

新增配置字段：

```python
model_arch: str | None = None
enable_moe_offload: bool = False
expert_cache_size: int = 120
expert_cache_memory_gb: float | None = None
expert_policy: str = "on_demand_lru"
expert_offload_device: str = "cpu"
moe_trace_path: str | None = None
disable_cuda_graph_for_moe: bool = True
```

配置约束：

1. `enable_moe_offload=True` 时强制 `tensor_parallel_size=1`。
2. `enable_moe_offload=True` 时强制 `enforce_eager=True` 或跳过 CUDA graph。
3. 如果 `hf_config.model_type == "qwen2_moe"`，但没有启用 MoE path，应给出明确错误。

### 5.2 `nanovllm/engine/model_runner.py`

修改点：

1. 用 `get_model_cls(config.hf_config)` 替代硬编码 `Qwen3ForCausalLM`。
2. 初始化 MoE offload runtime：

```python
expert_store = ExpertStore(config.model)
expert_cache = ExpertCache(expert_store, ...)
trace_recorder = MoeTraceRecorder(...)
```

3. 构造 `Qwen2MoeForCausalLM(hf_config, expert_cache=..., trace_recorder=...)`。
4. 调用模型对应 loader。
5. MoE 模式禁用 `capture_cudagraph()`。
6. 在 `exit()` 时 flush trace。

### 5.3 `nanovllm/utils/loader.py`

修改点：

1. 保留现有 dense `load_model()`。
2. 增加分发函数：

```python
def load_model_auto(model, path, config, expert_store=None):
    if config.hf_config.model_type == "qwen2_moe":
        return load_qwen2_moe_model(...)
    return load_model(model, path)
```

### 5.4 `nanovllm/layers/linear.py`

可选修改。

第一版可以不改，MoE expert 使用独立 `MoeExpertRuntime`。

后续如果要性能优化，可以增加：

```python
class ExpertLinear
class BatchedExpertLinear
```

用于 grouped GEMM 或 Triton kernel。

### 5.5 `pyproject.toml`

可能新增依赖：

```text
psutil
```

用于记录内存和进程信息。第一版也可以只用 torch 统计，避免新增依赖。

## 6. 推理流程

### 6.1 初始化流程

```text
LLM(model_path, enable_moe_offload=True)
  -> Config reads AutoConfig
  -> model_type = qwen2_moe
  -> ModelRunner creates ExpertStore
  -> ModelRunner creates ExpertCache
  -> ModelRunner creates Qwen2MoeForCausalLM
  -> moe_loader loads dense/router/shared weights
  -> ExpertStore indexes expert weights
  -> warmup uses small input
  -> allocate KV cache
```

### 6.2 单步 forward 流程

```text
input_ids, positions
  -> embedding
  -> for each decoder layer:
       attention
       layernorm
       router gate
       top-k expert ids
       ExpertCache.ensure_experts(layer_id, expert_ids)
       execute active routed experts
       execute shared expert
       combine output
  -> final norm
  -> lm_head
  -> sampler
```

### 6.3 expert cache miss 流程

```text
active expert ids = {3, 10, 42, ...}
  -> key = (layer_id, expert_id)
  -> if key in GPU cache:
       hit
  -> else:
       if cache full:
           evict victim by LRU
       cpu_weights = ExpertStore.get_cpu_weights(key)
       gpu_weights = cpu_weights.to(device="cuda", non_blocking=True)
       insert cache
       record h2d_time
```

## 7. 实现阶段

### Phase 0: 工程准备

产出：

1. 创建 nano-vLLM MoE 分支。
2. 固定远程环境和模型路径。
3. 增加最小测试脚本。

验收：

1. 原始 Qwen3-0.6B smoke test 仍能跑。
2. dense path 不被破坏。

### Phase 1: Qwen2-MoE 模型结构

产出：

1. `models/qwen2_moe.py`
2. `layers/moe.py`
3. `layers/moe_expert.py`
4. `models/registry.py`

验收：

1. 能构造模型。
2. 能完成随机输入 shape forward。
3. 不加载完整 expert 权重到 GPU。

### Phase 2: MoE 权重加载

产出：

1. `utils/moe_loader.py`
2. `offload/expert_store.py`

验收：

1. dense/router/shared 权重加载成功。
2. expert 权重能按 `(layer_id, expert_id)` 从 safetensors 找到并加载。
3. GPU 显存不因所有 experts 注册为 parameter 而爆掉。

### Phase 3: 同步 on-demand expert offload

产出：

1. `offload/expert_cache.py`
2. `offload/policy.py`
3. `offload/trace.py`

验收：

1. 单条 prompt 能生成。
2. trace 中能看到 expert hit/miss。
3. cache size 改小会触发 eviction。
4. 峰值显存低于全量 expert 常驻方案。

### Phase 4: Benchmark 和对照

产出：

1. `examples/qwen2_moe_offload.py`
2. `bench_moe_offload.py`
3. 结果目录保存 `metrics.json`、`moe_trace.jsonl`、`moe_summary.json`。

验收：

1. 与 Transformers CPU offload baseline 对比。
2. 输出 latency、tokens/s、H2D 时间、hit rate。
3. 能定位瓶颈是 expert 迁移、expert compute 还是 attention/KV。

### Phase 5: 异步预取

这不是第一版内容。

新增：

1. CUDA stream for H2D。
2. pinned CPU expert weights。
3. prefetch queue。
4. router-history or trace-based prediction。
5. timely utility metrics。

验收重点不是 cache hit rate，而是：

```text
prefetch issued before use
expert available when needed
H2D wait time reduced
redundant prefetch controlled
end-to-end latency improved
```

## 8. 风险和边界

### 8.1 正确性风险

MoE combine 逻辑必须严格对齐 Hugging Face：

1. top-k softmax 维度。
2. `norm_topk_prob` 处理。
3. shared expert gate。
4. dtype：模型 config 是 bfloat16，但当前 baseline 常用 float16。

建议第一版用同一 prompt 对比 Transformers 输出 logits 或生成前几个 token。

### 8.2 性能风险

Python loop over experts 会慢，但第一版可以接受。优化方向：

1. 按 expert 分组 token。
2. 合并小 batch expert GEMM。
3. Triton grouped GEMM。
4. 异步 H2D。

### 8.3 显存风险

4090 24GB 无法舒适常驻所有 expert、KV cache 和中间激活。必须确保：

1. expert 不作为全量 `nn.Parameter` 常驻 GPU。
2. cache size 有上限。
3. KV cache 预留和 expert cache 预留互不挤爆。

### 8.4 CUDA graph 风险

动态 expert offload 会改变权重驻留和执行路径，第一版必须禁用 CUDA graph。后续若要重新启用，只能对固定 resident experts 或固定 batch shape 做局部 graph。

## 9. 最小可运行验收命令

远程环境：

```bash
source /root/miniconda3/bin/activate /root/autodl-tmp/conda-envs/nano-vllm
cd /root/autodl-tmp/nano-vllm
```

预期 smoke test：

```bash
python examples/qwen2_moe_offload.py \
  --model /root/autodl-tmp/hf-cache/transformers/models--Qwen--Qwen1.5-MoE-A2.7B-Chat/snapshots/ec052fda178e241c7c443468d2fa1db6618996be \
  --expert-cache-size 120 \
  --max-new-tokens 16 \
  --output-dir /root/autodl-tmp/runs/nano_vllm_qwen_moe_smoke
```

预期输出：

```text
metrics.json
generations.jsonl
moe_trace.jsonl
moe_summary.json
```

第一版成功标准：

1. `exit_code = 0`
2. 能生成文本
3. `moe_summary.json` 中有非零 expert miss
4. GPU 显存峰值低于全 expert 常驻
5. trace 可以解释每层 active experts 和迁移时间

## 10. 推荐实施顺序

推荐按以下顺序提交：

1. `model registry + config flags`
2. `qwen2_moe model skeleton + shape tests`
3. `moe_loader + expert_store`
4. `expert_cache + on-demand LRU`
5. `trace recorder`
6. `qwen2_moe_offload example`
7. `benchmark script`
8. `Transformers baseline comparison report`

这样每一步都有可验证结果，不会一次性修改过多文件导致难以定位错误。

