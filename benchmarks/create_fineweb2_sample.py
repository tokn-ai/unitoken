from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


DEFAULT_OUTPUT = Path("out") / "fineweb2_1GiB.txt"
GIB = 1024 ** 3


def write_prefix(out, text: str, byte_budget: int) -> int:
  encoded = text.encode("utf-8")
  if len(encoded) <= byte_budget:
    out.write(encoded)
    return len(encoded)

  prefix = encoded[:byte_budget]
  while prefix:
    try:
      prefix.decode("utf-8")
      break
    except UnicodeDecodeError:
      prefix = prefix[:-1]

  out.write(prefix)
  return len(prefix)


def parquet_files(input_dir: Path, limit: int | None) -> list[Path]:
  files = sorted(input_dir.glob("*.parquet"))
  if limit is not None:
    files = files[:limit]
  if not files:
    raise FileNotFoundError(f"no parquet files found in {input_dir}")
  return files


def create_sample(args: argparse.Namespace) -> dict[str, Any]:
  files = parquet_files(args.input_dir, args.max_files)
  args.output.parent.mkdir(parents=True, exist_ok=True)

  started = time.perf_counter()
  bytes_written = 0
  docs_written = 0
  files_read = 0

  with args.output.open("wb") as out:
    for path in files:
      if bytes_written >= args.size_bytes:
        break

      files_read += 1
      parquet = pq.ParquetFile(path)
      for batch in parquet.iter_batches(batch_size=args.batch_size, columns=[args.column]):
        if bytes_written >= args.size_bytes:
          break
        for scalar in batch.column(0):
          if bytes_written >= args.size_bytes:
            break
          text = scalar.as_py()
          if text is None:
            continue

          if docs_written > 0:
            separator_budget = args.size_bytes - bytes_written
            if separator_budget > 0:
              separator = b"\n\n"[:separator_budget]
              out.write(separator)
              bytes_written += len(separator)
          if bytes_written >= args.size_bytes:
            break

          bytes_written += write_prefix(out, text, args.size_bytes - bytes_written)
          docs_written += 1
        if bytes_written >= args.size_bytes:
          break

  if bytes_written < args.size_bytes and args.pad:
    with args.output.open("ab") as out:
      padding = args.size_bytes - bytes_written
      out.write(b" " * padding)
      bytes_written += padding

  return {
    "input_dir": str(args.input_dir),
    "output": str(args.output),
    "size_bytes": args.size_bytes,
    "bytes_written": bytes_written,
    "docs_written": docs_written,
    "files_read": files_read,
    "elapsed_s": time.perf_counter() - started,
  }


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Create a raw UTF-8 FineWeb2 text sample from local Parquet shards.")
  parser.add_argument("--input-dir", type=Path, required=True)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--size-bytes", type=int, default=GIB)
  parser.add_argument("--column", default="text")
  parser.add_argument("--batch-size", type=int, default=1024)
  parser.add_argument("--max-files", type=int)
  parser.add_argument("--pad", action="store_true", help="Pad with spaces if the source data is smaller than requested.")
  parser.add_argument("--json", type=Path)
  args = parser.parse_args(argv)

  if args.size_bytes < 1:
    parser.error("--size-bytes must be at least 1")
  if args.batch_size < 1:
    parser.error("--batch-size must be at least 1")

  result = create_sample(args)
  rendered = json.dumps(result, indent=2)
  print(rendered)
  if args.json:
    args.json.write_text(rendered + "\n", encoding="utf-8")
  return 0 if result["bytes_written"] == args.size_bytes else 1


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
