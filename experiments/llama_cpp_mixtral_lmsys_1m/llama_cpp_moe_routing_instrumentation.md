# llama.cpp MoE Routing Instrumentation Notes

The default llama.cpp server can expose memory, slot, and timing behavior, but it does not export per-layer MoE routing ids in the HTTP response.

For measured active expert weights, add a small trace hook near the Mixtral MoE routing implementation and write JSONL records:

```json
{
  "timestamp": "2026-05-24T00:00:00+08:00",
  "request_id": "optional",
  "step_id": 123,
  "layer_id": 7,
  "token_count": 16,
  "top_k": 2,
  "unique_expert_ids": [0, 2, 5, 7],
  "unique_expert_count": 4
}
```

Until this patch exists, plots must label active expert bytes as estimates.
