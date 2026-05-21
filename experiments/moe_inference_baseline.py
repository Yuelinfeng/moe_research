#!/usr/bin/env python3
"""Baseline MoE inference benchmark for the remote GPU runner."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_PROMPTS = [
    "Explain why expert routing can become a bottleneck in MoE inference.",
    "Summarize the difference between cache hit rate and timely prefetch utility.",
    "Give one concrete metric for evaluating MoE expert offloading latency.",
]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_command(command: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:  # pragma: no cover - environment diagnostic path.
        return {"error": repr(exc)}


def import_torch_only(output_dir: Path):
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        import torch.nn.functional as F  # type: ignore
    except Exception as exc:
        write_json(output_dir / "error.json", {"stage": "import_torch", "error": repr(exc)})
        raise SystemExit(41) from exc
    return torch, nn, F


class TinyMoELayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, intermediate_size, bias=False),
                    nn.GELU(),
                    nn.Linear(intermediate_size, hidden_size, bias=False),
                )
                for _ in range(num_experts)
            ]
        )
        self.pre_norm = nn.LayerNorm(hidden_size)
        self.post_norm = nn.LayerNorm(hidden_size)
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        normed = self.pre_norm(hidden)
        router_logits = self.router(normed)
        topk_logits, topk_idx = torch.topk(router_logits, self.top_k, dim=-1)
        topk_weights = torch.softmax(topk_logits, dim=-1)

        flat = normed.reshape(-1, normed.shape[-1])
        output = torch.zeros_like(flat)
        route_counts = torch.zeros(self.num_experts, device=hidden.device, dtype=torch.long)

        flat_topk_idx = topk_idx.reshape(-1, self.top_k)
        flat_topk_weights = topk_weights.reshape(-1, self.top_k)
        for slot in range(self.top_k):
            slot_idx = flat_topk_idx[:, slot]
            slot_weight = flat_topk_weights[:, slot].unsqueeze(-1)
            for expert_idx, expert in enumerate(self.experts):
                mask = slot_idx == expert_idx
                if not mask.any():
                    continue
                selected = flat[mask]
                routed = expert(selected) * slot_weight[mask]
                output[mask] += routed
                route_counts[expert_idx] += int(mask.sum().item())

        output = self.post_norm(output + flat)
        return output.reshape_as(hidden), {
            "route_counts": route_counts.detach().cpu().tolist(),
            "mean_router_logit": float(router_logits.mean().item()),
            "std_router_logit": float(router_logits.std(unbiased=False).item()),
        }


class TinyMoEModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        num_experts: int,
        top_k: int,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [TinyMoELayer(hidden_size, intermediate_size, num_experts, top_k) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        hidden = self.embed(input_ids)
        layer_stats: list[dict[str, Any]] = []
        for layer in self.layers:
            hidden, stats = layer(hidden)
            layer_stats.append(stats)
        logits = self.lm_head(self.final_norm(hidden))
        return logits, layer_stats


def run_synthetic_benchmark(args: argparse.Namespace, output_dir: Path, env: dict[str, Any]) -> int:
    device = "cuda" if env["cuda_available"] else "cpu"
    torch_dtype = resolve_dtype(torch, args.dtype, bool(env["cuda_available"]))
    generator = torch.Generator(device="cpu").manual_seed(1234)
    model = TinyMoEModel(
        vocab_size=args.synthetic_vocab_size,
        hidden_size=args.synthetic_hidden_size,
        intermediate_size=args.synthetic_intermediate_size,
        num_layers=args.synthetic_num_layers,
        num_experts=args.synthetic_num_experts,
        top_k=args.synthetic_top_k,
    ).to(device=device, dtype=torch_dtype)
    model.eval()

    load_seconds = 0.0
    rows: list[dict[str, Any]] = []
    generations_path = output_dir / "generations.jsonl"
    route_totals = [0 for _ in range(args.synthetic_num_experts)]

    with generations_path.open("w", encoding="utf-8") as out:
        for idx, prompt in enumerate(load_prompts(args.prompt_file)):
            input_ids = torch.randint(
                low=0,
                high=args.synthetic_vocab_size,
                size=(args.synthetic_batch_size, args.synthetic_seq_len),
                generator=generator,
            ).to(device)
            prompt_tokens = int(input_ids.shape[-1])
            repeats = args.warmup_iters + args.benchmark_iters
            for rep in range(repeats):
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                start = time.perf_counter()
                cur_ids = input_ids
                last_stats: list[dict[str, Any]] = []
                with torch.inference_mode():
                    for _ in range(args.max_new_tokens):
                        logits, layer_stats = model(cur_ids)
                        last_stats = layer_stats
                        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                        cur_ids = torch.cat([cur_ids, next_token], dim=-1)
                if device == "cuda":
                    torch.cuda.synchronize()
                latency = time.perf_counter() - start
                generated_tokens = int(cur_ids.shape[-1] - prompt_tokens)
                for layer_stat in last_stats:
                    for expert_idx, count in enumerate(layer_stat["route_counts"]):
                        route_totals[expert_idx] += int(count)
                row = {
                    "prompt_id": idx,
                    "repeat": rep,
                    "is_warmup": rep < args.warmup_iters,
                    "input_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "latency_s": latency,
                    "tokens_per_s": generated_tokens / latency if latency > 0 else 0.0,
                    "peak_allocated_mb": (
                        torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else 0.0
                    ),
                }
                rows.append(row)
                if rep >= args.warmup_iters:
                    out.write(
                        json.dumps(
                            {
                                "prompt": prompt,
                                "input_ids_shape": list(input_ids.shape),
                                "generated_ids_shape": list(cur_ids.shape),
                                "route_totals": route_totals,
                                **row,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    metric_rows = [row for row in rows if not row["is_warmup"]]
    mean_latency = sum(float(row["latency_s"]) for row in metric_rows) / max(1, len(metric_rows))
    mean_tps = sum(float(row["tokens_per_s"]) for row in metric_rows) / max(1, len(metric_rows))
    max_peak_mb = max((float(row["peak_allocated_mb"]) for row in metric_rows), default=0.0)
    summary = {
        "mode": "synthetic",
        "device": device,
        "dtype": str(torch_dtype),
        "load_seconds": load_seconds,
        "benchmark_rows": len(metric_rows),
        "mean_latency_s": mean_latency,
        "mean_tokens_per_s": mean_tps,
        "max_peak_allocated_mb": max_peak_mb,
        "route_totals": route_totals,
        "synthetic_config": {
            "batch_size": args.synthetic_batch_size,
            "seq_len": args.synthetic_seq_len,
            "vocab_size": args.synthetic_vocab_size,
            "hidden_size": args.synthetic_hidden_size,
            "intermediate_size": args.synthetic_intermediate_size,
            "num_layers": args.synthetic_num_layers,
            "num_experts": args.synthetic_num_experts,
            "top_k": args.synthetic_top_k,
        },
    }
    write_json(output_dir / "metrics.json", summary)

    with (output_dir / "per_prompt_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return 0


def collect_environment(torch: Any) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    payload: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": getattr(torch, "__version__", ""),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()),
        "nvidia_smi": run_command(["nvidia-smi"]),
    }
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        payload.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_total_memory_bytes": int(props.total_memory),
            }
        )
    return payload


def load_prompts(prompt_file: str | None) -> list[str]:
    if not prompt_file:
        return DEFAULT_PROMPTS
    path = Path(prompt_file)
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return prompts or DEFAULT_PROMPTS


def resolve_dtype(torch: Any, dtype: str, cuda_available: bool) -> Any:
    if not cuda_available:
        return torch.float32
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    return torch.float16


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=["synthetic", "hf"], default="synthetic")
    parser.add_argument("--model-id", default="Qwen/Qwen1.5-MoE-A2.7B-Chat")
    parser.add_argument("--prompt-file")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--benchmark-iters", type=int, default=3)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--hf-placement", choices=["auto", "cuda"], default="auto")
    parser.add_argument("--cuda-max-memory", default="12GiB")
    parser.add_argument("--cpu-max-memory", default="38GiB")
    parser.add_argument("--offload-dir", type=Path)
    parser.add_argument("--require-cuda", choices=["0", "1"], default="1")
    parser.add_argument("--synthetic-batch-size", type=int, default=1)
    parser.add_argument("--synthetic-seq-len", type=int, default=64)
    parser.add_argument("--synthetic-vocab-size", type=int, default=2048)
    parser.add_argument("--synthetic-hidden-size", type=int, default=256)
    parser.add_argument("--synthetic-intermediate-size", type=int, default=512)
    parser.add_argument("--synthetic-num-layers", type=int, default=4)
    parser.add_argument("--synthetic-num-experts", type=int, default=8)
    parser.add_argument("--synthetic-top-k", type=int, default=2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch_mod, _, _ = import_torch_only(args.output_dir)
    env = collect_environment(torch_mod)
    write_json(args.output_dir / "environment.json", env)

    if args.require_cuda == "1" and not env["cuda_available"]:
        write_json(
            args.output_dir / "error.json",
            {
                "stage": "cuda_preflight",
                "error": "CUDA is not available to PyTorch; aborting before model load.",
                "nvidia_smi": env.get("nvidia_smi"),
            },
        )
        return 43

    if args.mode == "synthetic":
        return run_synthetic_benchmark(args, args.output_dir, env)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as exc:
        write_json(
            args.output_dir / "error.json",
            {
                "stage": "import_transformers",
                "error": repr(exc),
            },
        )
        return 42

    torch = torch_mod
    model_cls = AutoModelForCausalLM
    tokenizer_cls = AutoTokenizer

    device = "cuda" if env["cuda_available"] else "cpu"
    torch_dtype = resolve_dtype(torch, args.dtype, bool(env["cuda_available"]))
    prompts = load_prompts(args.prompt_file)

    load_start = time.perf_counter()
    try:
        tokenizer = tokenizer_cls.from_pretrained(args.model_id, trust_remote_code=True)
        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if args.hf_placement == "auto" and device == "cuda":
            offload_dir = args.offload_dir or (args.output_dir / "offload")
            offload_dir.mkdir(parents=True, exist_ok=True)
            model_kwargs.update(
                {
                    "device_map": "auto",
                    "max_memory": {0: args.cuda_max_memory, "cpu": args.cpu_max_memory},
                    "offload_folder": str(offload_dir),
                    "offload_state_dict": True,
                }
            )
        model = model_cls.from_pretrained(
            args.model_id,
            **model_kwargs,
        )
    except Exception as exc:
        write_json(
            args.output_dir / "error.json",
            {
                "stage": "load_hf_model",
                "model_id": args.model_id,
                "error": repr(exc),
            },
        )
        return 44
    try:
        if not (args.hf_placement == "auto" and hasattr(model, "hf_device_map")):
            model.to(device)
    except Exception as exc:
        write_json(
            args.output_dir / "error.json",
            {
                "stage": "move_hf_model_to_device",
                "model_id": args.model_id,
                "device": device,
                "hf_placement": args.hf_placement,
                "error": repr(exc),
            },
        )
        return 45
    model.eval()
    load_seconds = time.perf_counter() - load_start
    hf_device_map = getattr(model, "hf_device_map", None)
    input_device = device
    if isinstance(hf_device_map, dict):
        for key in ("model.embed_tokens", "transformer.wte", "embed_tokens"):
            if key in hf_device_map:
                input_device = str(hf_device_map[key])
                break
        else:
            first_device = next(iter(hf_device_map.values()), device)
            input_device = str(first_device)
    if input_device == "disk":
        input_device = device

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    rows: list[dict[str, Any]] = []
    generations_path = args.output_dir / "generations.jsonl"
    with generations_path.open("w", encoding="utf-8") as out:
        for idx, prompt in enumerate(prompts):
            repeats = args.warmup_iters + args.benchmark_iters
            for rep in range(repeats):
                inputs = tokenizer(prompt, return_tensors="pt").to(input_device)
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=pad_token_id,
                    )
                if device == "cuda":
                    torch.cuda.synchronize()
                latency = time.perf_counter() - start
                input_tokens = int(inputs["input_ids"].shape[-1])
                total_tokens = int(output_ids.shape[-1])
                generated_tokens = max(0, total_tokens - input_tokens)
                row = {
                    "prompt_id": idx,
                    "repeat": rep,
                    "is_warmup": rep < args.warmup_iters,
                    "input_tokens": input_tokens,
                    "generated_tokens": generated_tokens,
                    "latency_s": latency,
                    "tokens_per_s": generated_tokens / latency if latency > 0 else 0.0,
                    "peak_allocated_mb": (
                        torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else 0.0
                    ),
                }
                rows.append(row)
                if rep >= args.warmup_iters:
                    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    out.write(json.dumps({"prompt": prompt, "output": text, **row}, ensure_ascii=False) + "\n")

    metric_rows = [row for row in rows if not row["is_warmup"]]
    mean_latency = sum(float(row["latency_s"]) for row in metric_rows) / max(1, len(metric_rows))
    mean_tps = sum(float(row["tokens_per_s"]) for row in metric_rows) / max(1, len(metric_rows))
    max_peak_mb = max((float(row["peak_allocated_mb"]) for row in metric_rows), default=0.0)
    summary = {
        "model_id": args.model_id,
        "device": device,
        "dtype": str(torch_dtype),
        "hf_placement": args.hf_placement,
        "hf_device_map": hf_device_map,
        "input_device": input_device,
        "load_seconds": load_seconds,
        "benchmark_rows": len(metric_rows),
        "mean_latency_s": mean_latency,
        "mean_tokens_per_s": mean_tps,
        "max_peak_allocated_mb": max_peak_mb,
    }
    write_json(args.output_dir / "metrics.json", summary)

    with (args.output_dir / "per_prompt_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
