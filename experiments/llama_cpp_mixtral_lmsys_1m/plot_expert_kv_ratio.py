#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("missing dependency: pip install matplotlib") from exc

    rows = []
    with Path(args.summary_csv).open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x = [float(r["live_tokens"]) for r in rows]
    y = [float(r["active_expert_to_kv_ratio"]) for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    plt.xlabel("live tokens")
    plt.ylabel("active expert weight bytes / KV cache bytes")
    plt.title("Expert/KV Ratio vs Live Tokens")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "expert_kv_ratio_vs_live_tokens.png", dpi=160)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
