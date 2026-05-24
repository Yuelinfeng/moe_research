#!/usr/bin/env python3
import argparse
import csv
import gzip
import json
import statistics
from pathlib import Path


def open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def iter_results(output_dir: Path):
    for path in sorted(output_dir.glob("*.jsonl")) + sorted(output_dir.glob("*.jsonl.gz")):
        with open_text(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def read_system_peak(path: Path) -> dict:
    if not path.exists():
        return {}
    peaks = {
        "gpu_mem_peak_mb": 0,
        "ram_peak_mb": 0,
        "swap_peak_mb": 0,
        "gpu_util_peak": 0,
    }
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, col in [
                ("gpu_mem_peak_mb", "gpu_mem_used_mb"),
                ("ram_peak_mb", "ram_used_mb"),
                ("swap_peak_mb", "swap_used_mb"),
                ("gpu_util_peak", "gpu_util"),
            ]:
                try:
                    peaks[key] = max(peaks[key], float(row.get(col) or 0))
                except ValueError:
                    pass
    return peaks


def int_or_zero(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def kv_cache_bytes_for_tokens(live_tokens: int, layers: int, kv_heads: int, head_dim: int, kv_bytes: int) -> int:
    return live_tokens * layers * kv_heads * head_dim * 2 * kv_bytes


def read_slot_peaks(path: Path) -> dict:
    peaks = {
        "slot_peak_live_tokens": 0,
        "slot_kv_cache_bytes_estimated_peak": 0,
    }
    if not path.exists():
        return peaks

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            live_tokens = int_or_zero(row.get("slot_live_tokens_estimated"))
            kv_cache_bytes = int_or_zero(row.get("slot_kv_cache_bytes_estimated"))

            # Compatibility with older CSVs generated before slot_live_tokens_estimated existed.
            if not live_tokens:
                live_tokens = max(
                    int_or_zero(row.get("slot_n_past")),
                    int_or_zero(row.get("slot_cache_tokens")),
                    int_or_zero(row.get("slot_prompt_tokens")),
                )

            peaks["slot_peak_live_tokens"] = max(peaks["slot_peak_live_tokens"], live_tokens)
            peaks["slot_kv_cache_bytes_estimated_peak"] = max(
                peaks["slot_kv_cache_bytes_estimated_peak"], kv_cache_bytes
            )
    return peaks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--kv-bytes", type=int, default=2, help="f16=2, q8_0=1, q4_0~1 for coarse estimate")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "outputs"
    summary_dir = run_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows = list(iter_results(output_dir))
    ok = [r for r in rows if r.get("status") == "ok"]
    failed = [r for r in rows if r.get("status") != "ok"]
    latencies = [float(r["latency_ms"]) for r in ok if r.get("latency_ms") is not None]
    tps = [float(r["tokens_per_second"]) for r in ok if r.get("tokens_per_second")]
    request_live_tokens = [
        max(int_or_zero(r.get("live_tokens_before")), int_or_zero(r.get("live_tokens_after"))) for r in ok
    ]
    request_peak_live_tokens = max(request_live_tokens) if request_live_tokens else 0
    slot_peaks = read_slot_peaks(run_dir / "monitor" / "llama_slots_summary.csv")
    peak_live_tokens = max(request_peak_live_tokens, slot_peaks["slot_peak_live_tokens"])
    kv_cache_bytes_peak = max(
        slot_peaks["slot_kv_cache_bytes_estimated_peak"],
        kv_cache_bytes_for_tokens(peak_live_tokens, args.layers, args.kv_heads, args.head_dim, args.kv_bytes),
    )

    summary = {
        "num_requests_total": len(rows),
        "num_requests_ok": len(ok),
        "num_requests_failed": len(failed),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_p99_ms": percentile(latencies, 99),
        "tokens_per_second_avg": statistics.mean(tps) if tps else None,
        "tokens_per_second_p50": percentile(tps, 50),
        "request_peak_live_tokens": request_peak_live_tokens,
        "slot_peak_live_tokens": slot_peaks["slot_peak_live_tokens"],
        "peak_live_tokens": peak_live_tokens,
        "kv_cache_bytes_estimated_peak": kv_cache_bytes_peak,
        "kv_cache_mib_estimated_peak": kv_cache_bytes_peak / (1024 * 1024),
        "kv_cache_estimate_config": {
            "layers": args.layers,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "kv_bytes": args.kv_bytes,
            "bytes_per_token": kv_cache_bytes_for_tokens(1, args.layers, args.kv_heads, args.head_dim, args.kv_bytes),
        },
    }
    summary.update(read_system_peak(run_dir / "monitor" / "system_samples.csv"))

    (summary_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
