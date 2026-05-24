#!/usr/bin/env python3
import argparse
import heapq
import hashlib
import json
import os
import sys
from pathlib import Path


def first_user_message(conversation) -> str:
    if not conversation:
        return ""
    for item in conversation:
        if isinstance(item, dict) and item.get("role") == "user":
            return item.get("content") or ""
    return ""


def make_payload(index: int, row: dict, prompt: str) -> dict:
    request_id = f"{index:012d}"
    return {
        "request_id": request_id,
        "conversation_id": row.get("conversation_id"),
        "language": row.get("language"),
        "turn": row.get("turn"),
        "model_in_lmsys": row.get("model"),
        "prompt_chars": len(prompt),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt": prompt,
        "redacted": row.get("redacted"),
    }


def write_payloads(rows: list[dict], shard_dir: Path, shard_size: int) -> int:
    shard_index = 0
    shard_f = None
    try:
        for written, payload in enumerate(rows):
            if shard_f is None or written % shard_size == 0:
                if shard_f is not None:
                    shard_f.close()
                shard_f = (shard_dir / f"shard_{shard_index:06d}.jsonl").open("w", encoding="utf-8")
                shard_index += 1
            shard_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    finally:
        if shard_f is not None:
            shard_f.close()
    return shard_index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="lmsys/lmsys-chat-1m")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--select-mode", choices=["first", "longest"], default="first")
    parser.add_argument("--scan-limit", type=int, default=0, help="rows to scan before selecting longest prompts")
    args = parser.parse_args()

    if args.select_mode == "longest" and args.limit <= 0:
        raise SystemExit("--select-mode longest requires --limit")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("missing dependency: pip install datasets pyarrow") from exc

    out_dir = Path(args.output_dir)
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    skipped_path = out_dir / "skipped.jsonl"
    manifest_path = out_dir / "dataset_manifest.json"

    dataset = load_dataset(args.repo_id, split=args.split, streaming=True)
    selected_rows: list[dict] = []
    rows_scanned = 0
    skipped = 0

    with skipped_path.open("w", encoding="utf-8") as skipped_f:
        if args.select_mode == "longest":
            heap: list[tuple[int, int, dict]] = []
            for i, row in enumerate(dataset):
                if args.scan_limit and rows_scanned >= args.scan_limit:
                    break
                rows_scanned += 1
                prompt = first_user_message(row.get("conversation"))
                request_id = f"{i:012d}"
                if not prompt:
                    skipped_f.write(json.dumps({"request_id": request_id, "reason": "missing_first_user_message"}, ensure_ascii=False) + "\n")
                    skipped += 1
                    continue
                payload = make_payload(i, row, prompt)
                item = (payload["prompt_chars"], i, payload)
                if len(heap) < args.limit:
                    heapq.heappush(heap, item)
                elif item[0] > heap[0][0]:
                    heapq.heapreplace(heap, item)
            selected_rows = [item[2] for item in heap]
            selected_rows.sort(key=lambda item: (-item["prompt_chars"], item["request_id"]))
        else:
            for i, row in enumerate(dataset):
                if args.limit and len(selected_rows) >= args.limit:
                    break
                rows_scanned += 1
                prompt = first_user_message(row.get("conversation"))
                request_id = f"{i:012d}"
                if not prompt:
                    skipped_f.write(json.dumps({"request_id": request_id, "reason": "missing_first_user_message"}, ensure_ascii=False) + "\n")
                    skipped += 1
                    continue
                selected_rows.append(make_payload(i, row, prompt))

    shard_index = write_payloads(selected_rows, shard_dir, args.shard_size)
    prompt_chars = [int(row.get("prompt_chars") or 0) for row in selected_rows]

    manifest = {
        "repo_id": args.repo_id,
        "split": args.split,
        "shard_size": args.shard_size,
        "limit": args.limit,
        "select_mode": args.select_mode,
        "scan_limit": args.scan_limit,
        "rows_scanned": rows_scanned,
        "rows_written": len(selected_rows),
        "rows_skipped": skipped,
        "num_shards": shard_index,
        "prompt_chars_min": min(prompt_chars) if prompt_chars else 0,
        "prompt_chars_max": max(prompt_chars) if prompt_chars else 0,
        "prompt_chars_avg": sum(prompt_chars) / len(prompt_chars) if prompt_chars else 0,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
