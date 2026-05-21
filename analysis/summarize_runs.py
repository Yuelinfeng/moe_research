#!/usr/bin/env python3
"""Summarize downloaded AIRS MoE run directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def read_manifest(run_dir: Path) -> dict[str, object]:
    manifest_json = run_dir / "manifest.json"
    if manifest_json.exists():
        return json.loads(manifest_json.read_text(encoding="utf-8"))

    manifest_env = run_dir / "manifest.env"
    data: dict[str, object] = {}
    if manifest_env.exists():
        for line in manifest_env.read_text(encoding="utf-8").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key] = value
    return data


def summarize_run(run_dir: Path) -> dict[str, object]:
    manifest = read_manifest(run_dir)
    exit_code = manifest.get("exit_code", read_text(run_dir / "exit_code.txt"))

    return {
        "run_id": manifest.get("run_id", run_dir.name),
        "run_dir": str(run_dir),
        "run_label": manifest.get("run_label", ""),
        "branch": manifest.get("branch", ""),
        "commit": manifest.get("commit", ""),
        "short_commit": manifest.get("short_commit", ""),
        "started_at": manifest.get("started_at", ""),
        "finished_at": manifest.get("finished_at", ""),
        "exit_code": exit_code,
        "stdout_bytes": (run_dir / "stdout.log").stat().st_size
        if (run_dir / "stdout.log").exists()
        else 0,
        "stderr_bytes": (run_dir / "stderr.log").stat().st_size
        if (run_dir / "stderr.log").exists()
        else 0,
        "run_command": manifest.get("run_command", read_text(run_dir / "run_command.sh")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    run_dirs = sorted(
        [path for path in args.results_root.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
    )

    rows = [summarize_run(path) for path in run_dirs]
    fieldnames = [
        "run_id",
        "run_dir",
        "run_label",
        "branch",
        "short_commit",
        "commit",
        "started_at",
        "finished_at",
        "exit_code",
        "stdout_bytes",
        "stderr_bytes",
        "run_command",
    ]

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
