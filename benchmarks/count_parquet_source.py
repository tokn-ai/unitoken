from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from uni_tokenizer import PreTokenizer


GIB = 1024 ** 3
DEFAULT_WORDS_OUTPUT = (
  Path("out")
  / "data"
  / "fineweb2"
  / "cmn_Hani"
  / "fineweb2_cmn_Hani_5GiB.unicode_bigram_top100k_min16"
  / "_words.json"
)


class ParquetSource:
  def __init__(
    self,
    input_dir: Path,
    *,
    size_bytes: int,
    column: str,
    parquet_batch_size: int,
  ) -> None:
    self.paths = sorted(input_dir.glob("*.parquet"))
    if not self.paths:
      raise FileNotFoundError(f"no Parquet files found in {input_dir}")
    self.size_bytes = size_bytes
    self.column = column
    self.parquet_batch_size = parquet_batch_size
    self.scan_number = 0
    self.last_scan: dict[str, int | float] = {}

  def scan(self) -> Iterator[str]:
    self.scan_number += 1
    started = time.perf_counter()
    bytes_read = 0
    records_read = 0
    files_read = 0
    next_report = GIB

    for path in self.paths:
      files_read += 1
      parquet = pq.ParquetFile(path)
      batches = parquet.iter_batches(
        batch_size=self.parquet_batch_size,
        columns=[self.column],
        use_threads=True,
      )
      for batch in batches:
        for text in batch.column(0).to_pylist():
          if text is None:
            continue
          text_bytes = len(text.encode("utf-8"))
          if records_read > 0 and bytes_read + text_bytes > self.size_bytes:
            self.last_scan = {
              "bytes": bytes_read,
              "records": records_read,
              "files": files_read,
              "elapsed_s": time.perf_counter() - started,
            }
            return
          bytes_read += text_bytes
          records_read += 1
          yield text
          if bytes_read >= next_report:
            elapsed = time.perf_counter() - started
            print(
              f"scan={self.scan_number} read={bytes_read / GIB:.2f} GiB "
              f"records={records_read:,} rate={bytes_read / GIB / elapsed:.3f} GiB/s",
              file=sys.stderr,
              flush=True,
            )
            next_report += GIB

    self.last_scan = {
      "bytes": bytes_read,
      "records": records_read,
      "files": files_read,
      "elapsed_s": time.perf_counter() - started,
    }


def max_rss_bytes() -> int:
  rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  return rss if sys.platform == "darwin" else rss * 1024


def scan_result(source: ParquetSource) -> dict[str, int | float]:
  result = dict(source.last_scan)
  elapsed = float(result["elapsed_s"])
  result["throughput_mib_s"] = int(result["bytes"]) / 1024 ** 2 / elapsed
  result["max_rss_bytes"] = max_rss_bytes()
  return result


def run(args: argparse.Namespace) -> dict[str, Any]:
  source = ParquetSource(
    args.input_dir,
    size_bytes=args.size_bytes,
    column=args.column,
    parquet_batch_size=args.parquet_batch_size,
  )
  pretokenizer = PreTokenizer([])

  bigram_counter = pretokenizer.bigram_counter()
  bigram_counter.add_source(
    source.scan(),
    max_records=args.max_records,
    max_bytes=args.max_bytes,
  )
  bigram_pass = scan_result(source)

  started = time.perf_counter()
  bigrams = bigram_counter.selected(top_k=args.top_k, min_freq=args.min_freq)
  selection_s = time.perf_counter() - started
  del bigram_counter

  word_counter = pretokenizer.with_unicode_bigrams(bigrams).word_counter()
  word_counter.add_source(
    source.scan(),
    max_records=args.max_records,
    max_bytes=args.max_bytes,
  )
  word_pass = scan_result(source)

  args.words_output.parent.mkdir(parents=True, exist_ok=True)
  started = time.perf_counter()
  word_counter.save(args.words_output)
  save_s = time.perf_counter() - started

  return {
    "contract": "parquet_source_two_pass_exact_words",
    "input_dir": str(args.input_dir),
    "size_bytes": args.size_bytes,
    "column": args.column,
    "parquet_batch_size": args.parquet_batch_size,
    "source_batch": {
      "max_records": args.max_records,
      "max_bytes": args.max_bytes,
    },
    "unicode_bigrams": {
      "top_k": args.top_k,
      "min_freq": args.min_freq,
      "selected": len(bigrams),
      "selection_s": selection_s,
    },
    "bigram_pass": bigram_pass,
    "word_pass": word_pass,
    "unique_words": word_counter.len,
    "words_output": str(args.words_output),
    "words_output_bytes": args.words_output.stat().st_size,
    "save_words_s": save_s,
    "max_rss_bytes": max_rss_bytes(),
  }


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
    description="Count exact words from a replayable Parquet source using the two-pass native API."
  )
  parser.add_argument("--input-dir", type=Path, required=True)
  parser.add_argument("--size-bytes", type=int, default=5 * GIB)
  parser.add_argument("--column", default="text")
  parser.add_argument("--parquet-batch-size", type=int, default=2048)
  parser.add_argument("--max-records", type=int, default=4096)
  parser.add_argument("--max-bytes", type=int, default=64 * 1024 * 1024)
  parser.add_argument("--top-k", type=int, default=100_000)
  parser.add_argument("--min-freq", type=int, default=16)
  parser.add_argument("--words-output", type=Path, default=DEFAULT_WORDS_OUTPUT)
  parser.add_argument("--json", type=Path)
  args = parser.parse_args(argv)

  if args.size_bytes < 1:
    parser.error("--size-bytes must be at least 1")
  if args.parquet_batch_size < 1:
    parser.error("--parquet-batch-size must be at least 1")

  result = run(args)
  rendered = json.dumps(result, indent=2)
  print(rendered)
  if args.json:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(rendered + "\n", encoding="utf-8")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
