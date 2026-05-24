#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable


SMOKE_PROMPTS = [
    {
        "request_id": "smoke_000",
        "conversation_id": "smoke_000",
        "language": "English",
        "turn": 1,
        "prompt": "Explain why KV Cache grows with batch size in LLM serving in one sentence.",
        "redacted": False,
    },
    {
        "request_id": "smoke_001",
        "conversation_id": "smoke_001",
        "language": "English",
        "turn": 1,
        "prompt": "In one sentence, explain why MoE expert weights and KV Cache compete for GPU memory.",
        "redacted": False,
    },
    {
        "request_id": "smoke_002",
        "conversation_id": "smoke_002",
        "language": "Chinese",
        "turn": 1,
        "prompt": "用一句话说明大 batch 推理时 KV Cache 为什么会成为显存瓶颈。",
        "redacted": False,
    },
]


def open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict]:
    with open_text(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def discover_inputs(args) -> list[dict]:
    if args.smoke:
        return SMOKE_PROMPTS[: args.limit or None]

    paths: list[Path] = []
    if args.input_jsonl:
        paths.append(Path(args.input_jsonl))
    if args.shard_dir:
        paths.extend(sorted(Path(args.shard_dir).glob("*.jsonl")))
        paths.extend(sorted(Path(args.shard_dir).glob("*.jsonl.gz")))

    rows: list[dict] = []
    for path in paths:
        for row in iter_jsonl(path):
            rows.append(row)
            if args.limit and len(rows) >= args.limit:
                return rows
    return rows


def post_json(url: str, payload: dict, timeout: float) -> tuple[int, dict | str]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, body
    except Exception as exc:
        return 0, repr(exc)


def get_slots(base_url: str) -> dict | list | None:
    try:
        with urllib.request.urlopen(f"{base_url}/slots", timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return None


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


def count_live_tokens(slots_payload) -> int:
    total = 0
    for slot in normalize_slots(slots_payload):
        if isinstance(slot, dict):
            total += estimate_slot_live_tokens(slot)
    return total


def output_text_from_response(payload) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    message = choice.get("message") or {}
    return message.get("content") or choice.get("text") or ""


def run_one(row: dict, args) -> dict:
    prompt = row.get("prompt") or ""
    live_before = count_live_tokens(get_slots(args.base_url))
    payload = {
        "model": args.model_name,
        "messages": [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": False,
    }
    start = time.perf_counter()
    status_code, response = post_json(f"{args.base_url}/v1/chat/completions", payload, args.timeout)
    latency_ms = (time.perf_counter() - start) * 1000.0
    live_after = count_live_tokens(get_slots(args.base_url))

    ok = 200 <= status_code < 300 and isinstance(response, dict)
    text = output_text_from_response(response)
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    timings = response.get("timings", {}) if isinstance(response, dict) else {}
    completion_tokens = int(usage.get("completion_tokens") or usage.get("predicted_n") or 0)
    tokens_per_second = completion_tokens / (latency_ms / 1000.0) if latency_ms > 0 and completion_tokens else 0.0

    result = {
        "request_id": row.get("request_id"),
        "conversation_id": row.get("conversation_id"),
        "language": row.get("language"),
        "turn": row.get("turn"),
        "status": "ok" if ok else "error",
        "http_status": status_code,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "live_tokens_before": live_before,
        "live_tokens_after": live_after,
        "latency_ms": latency_ms,
        "ttft_ms": None,
        "tokens_per_second": tokens_per_second,
        "timings": timings,
        "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest() if text else "",
    }
    if args.store_output_text:
        result["output_text"] = text
    if not ok:
        result["error"] = response
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--model-name", default="mixtral-q8")
    parser.add_argument("--input-jsonl", default="")
    parser.add_argument("--shard-dir", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="results.jsonl.gz")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--store-output-text", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    args = parser.parse_args()

    rows = discover_inputs(args)
    if not rows:
        raise SystemExit("no input rows found")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name

    done_ids = set()
    if out_path.exists():
        for result in iter_jsonl(out_path):
            rid = result.get("request_id")
            if rid:
                done_ids.add(rid)

    rows = [r for r in rows if r.get("request_id") not in done_ids]

    with open_text(out_path, "a") as f:
        if args.concurrency <= 1:
            for row in rows:
                f.write(json.dumps(run_one(row, args), ensure_ascii=False) + "\n")
                f.flush()
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [pool.submit(run_one, row, args) for row in rows]
                for fut in as_completed(futures):
                    f.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
                    f.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
