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


def slot_row(ts: str, slot: dict) -> dict:
    n_ctx = slot.get("n_ctx", slot.get("ctx_size", ""))
    n_past = slot.get("n_past", slot.get("cache_tokens", slot.get("n_tokens", "")))
    return {
        "timestamp": ts,
        "slot_id": slot.get("id", slot.get("slot_id", "")),
        "slot_state": slot.get("state", slot.get("is_processing", "")),
        "slot_prompt_tokens": slot.get("prompt_tokens", ""),
        "slot_predicted_tokens": slot.get("predicted_tokens", ""),
        "slot_n_ctx": n_ctx,
        "slot_n_past": n_past,
        "slot_cache_tokens": slot.get("cache_tokens", n_past),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=0.0, help="0 means until stop file exists")
    parser.add_argument("--stop-file", default="")
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
        "slot_state",
        "slot_prompt_tokens",
        "slot_predicted_tokens",
        "slot_n_ctx",
        "slot_n_past",
        "slot_cache_tokens",
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
                    writer.writerow(slot_row(ts, slot))
                summary_f.flush()

            time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
