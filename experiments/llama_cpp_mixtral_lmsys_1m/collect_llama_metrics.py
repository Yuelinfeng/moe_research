#!/usr/bin/env python3
import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def get_text(url: str, timeout: float = 5.0) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError):
        return None


def get_json(url: str, timeout: float = 5.0):
    text = get_text(url, timeout=timeout)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def normalize_slots(payload):
    if payload is None:
        return []
    if isinstance(payload, dict):
        if isinstance(payload.get("slots"), list):
            return payload["slots"]
        if isinstance(payload.get("data"), list):
            return payload["data"]
    if isinstance(payload, list):
        return payload
    return []


def int_or_zero(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def first_next_token(slot: dict) -> dict:
    next_token = slot.get("next_token")
    if isinstance(next_token, list) and next_token and isinstance(next_token[0], dict):
        return next_token[0]
    if isinstance(next_token, dict):
        return next_token
    return {}


def estimate_slot_live_tokens(slot: dict) -> int:
    next_token = first_next_token(slot)
    n_prompt = int_or_zero(slot.get("n_prompt_tokens", slot.get("prompt_tokens", 0)))
    n_processed = int_or_zero(slot.get("n_prompt_tokens_processed", 0))
    n_cache = int_or_zero(slot.get("n_prompt_tokens_cache", 0))
    n_decoded = int_or_zero(next_token.get("n_decoded", slot.get("predicted_tokens", 0)))
    legacy = int_or_zero(slot.get("n_past", slot.get("cache_tokens", slot.get("n_tokens", 0))))
    return max(n_prompt, n_processed + n_decoded, n_cache + n_processed + n_decoded, legacy)


def estimate_kv_cache_bytes(live_tokens: int, layers: int, kv_heads: int, head_dim: int, kv_bytes: int) -> int:
    return live_tokens * layers * kv_heads * head_dim * 2 * kv_bytes


def slot_row(ts: str, slot: dict, args) -> dict:
    n_ctx = slot.get("n_ctx", slot.get("ctx_size", ""))
    next_token = first_next_token(slot)
    live_tokens = estimate_slot_live_tokens(slot)
    kv_cache_bytes = estimate_kv_cache_bytes(live_tokens, args.layers, args.kv_heads, args.head_dim, args.kv_bytes)
    return {
        "timestamp": ts,
        "slot_id": slot.get("id", slot.get("slot_id", "")),
        "slot_is_processing": slot.get("is_processing", slot.get("state", "")),
        "slot_task_id": slot.get("id_task", slot.get("task_id", "")),
        "slot_n_ctx": n_ctx,
        "slot_prompt_tokens": slot.get("n_prompt_tokens", slot.get("prompt_tokens", "")),
        "slot_prompt_tokens_processed": slot.get("n_prompt_tokens_processed", ""),
        "slot_prompt_tokens_cache": slot.get("n_prompt_tokens_cache", ""),
        "slot_next_n_decoded": next_token.get("n_decoded", slot.get("predicted_tokens", "")),
        "slot_next_n_remain": next_token.get("n_remain", ""),
        "slot_live_tokens_estimated": live_tokens,
        "slot_kv_cache_bytes_estimated": kv_cache_bytes,
        "slot_kv_cache_mib_estimated": kv_cache_bytes / (1024 * 1024),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=0.0, help="0 means until stop file exists")
    parser.add_argument("--stop-file", default="")
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--kv-bytes", type=int, default=2, help="f16=2, q8_0=1, q4_0~1 for coarse estimate")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "llama_metrics_samples.jsonl"
    slots_path = out_dir / "llama_slots_samples.jsonl"
    summary_path = out_dir / "llama_slots_summary.csv"
    stop_file = Path(args.stop_file) if args.stop_file else None

    started = time.time()
    fieldnames = [
        "timestamp",
        "slot_id",
        "slot_is_processing",
        "slot_task_id",
        "slot_n_ctx",
        "slot_prompt_tokens",
        "slot_prompt_tokens_processed",
        "slot_prompt_tokens_cache",
        "slot_next_n_decoded",
        "slot_next_n_remain",
        "slot_live_tokens_estimated",
        "slot_kv_cache_bytes_estimated",
        "slot_kv_cache_mib_estimated",
    ]

    with metrics_path.open("a", encoding="utf-8") as metrics_f, slots_path.open("a", encoding="utf-8") as slots_f, summary_path.open("w", encoding="utf-8", newline="") as summary_f:
        writer = csv.DictWriter(summary_f, fieldnames=fieldnames)
        writer.writeheader()

        while True:
            if stop_file and stop_file.exists():
                break
            if args.duration > 0 and time.time() - started >= args.duration:
                break

            ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            metrics_text = get_text(f"{args.base_url}/metrics")
            if metrics_text is not None:
                metrics_f.write(json.dumps({"timestamp": ts, "text": metrics_text}, ensure_ascii=False) + "\n")
                metrics_f.flush()

            slots_payload = get_json(f"{args.base_url}/slots")
            if slots_payload is not None:
                slots_f.write(json.dumps({"timestamp": ts, "payload": slots_payload}, ensure_ascii=False) + "\n")
                slots_f.flush()
                for slot in normalize_slots(slots_payload):
                    writer.writerow(slot_row(ts, slot, args))
                summary_f.flush()

            time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
