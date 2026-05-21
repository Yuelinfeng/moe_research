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


def import_or_fail(output_dir: Path):
    missing = []
    try:
        import torch  # type: ignore
    except Exception as exc:
        write_json(output_dir / "error.json", {"stage": "import_torch", "error": repr(exc)})
        raise SystemExit(41) from exc

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception:
        missing.append("transformers")

    if missing:
        write_json(
            output_dir / "error.json",
            {
                "stage": "import_dependencies",
                "missing": missing,
                "hint": "Use the /root/miniconda3/envs/deepseek_moe environment or install transformers.",
            },
        )
        raise SystemExit(42)

    return torch, AutoModelForCausalLM, AutoTokenizer


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
    parser.add_argument("--model-id", default="Qwen/Qwen1.5-MoE-A2.7B-Chat")
    parser.add_argument("--prompt-file")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--benchmark-iters", type=int, default=3)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--require-cuda", choices=["0", "1"], default="1")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch, model_cls, tokenizer_cls = import_or_fail(args.output_dir)
    env = collect_environment(torch)
    write_json(args.output_dir / "environment.json", env)

    if args.require_cuda == "1" and not env["cuda_available"]:
        write_json(
            args.output_dir / "error.json",
            {
                "stage": "cuda_preflight",
                "error": "CUDA is not available to PyTorch; aborting before model download/load.",
                "nvidia_smi": env.get("nvidia_smi"),
            },
        )
        return 43

    device = "cuda" if env["cuda_available"] else "cpu"
    torch_dtype = resolve_dtype(torch, args.dtype, bool(env["cuda_available"]))
    prompts = load_prompts(args.prompt_file)

    load_start = time.perf_counter()
    tokenizer = tokenizer_cls.from_pretrained(args.model_id, trust_remote_code=True)
    model = model_cls.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    load_seconds = time.perf_counter() - load_start

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    rows: list[dict[str, Any]] = []
    generations_path = args.output_dir / "generations.jsonl"
    with generations_path.open("w", encoding="utf-8") as out:
        for idx, prompt in enumerate(prompts):
            repeats = args.warmup_iters + args.benchmark_iters
            for rep in range(repeats):
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
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
