#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def first_user_message(conversation) -> str:
    if not conversation:
        return ""
    for item in conversation:
        if isinstance(item, dict) and item.get("role") == "user":
            return item.get("content") or ""
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="lmsys/lmsys-chat-1m")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

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
    shard_index = 0
    written = 0
    skipped = 0
    shard_f = None

    def open_shard(index: int):
        return (shard_dir / f"shard_{index:06d}.jsonl").open("w", encoding="utf-8")

    with skipped_path.open("w", encoding="utf-8") as skipped_f:
        for i, row in enumerate(dataset):
            if args.limit and i >= args.limit:
                break
            prompt = first_user_message(row.get("conversation"))
            request_id = f"{i:012d}"
            if not prompt:
                skipped_f.write(json.dumps({"request_id": request_id, "reason": "missing_first_user_message"}, ensure_ascii=False) + "\n")
                skipped += 1
                continue
            if shard_f is None or written % args.shard_size == 0:
                if shard_f is not None:
                    shard_f.close()
                shard_f = open_shard(shard_index)
                shard_index += 1
            payload = {
                "request_id": request_id,
                "conversation_id": row.get("conversation_id"),
                "language": row.get("language"),
                "turn": row.get("turn"),
                "model_in_lmsys": row.get("model"),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt": prompt,
                "redacted": row.get("redacted"),
            }
            shard_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1

    if shard_f is not None:
        shard_f.close()

    manifest = {
        "repo_id": args.repo_id,
        "split": args.split,
        "shard_size": args.shard_size,
        "rows_written": written,
        "rows_skipped": skipped,
        "num_shards": shard_index,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
