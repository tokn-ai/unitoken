from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from uni_tokenizer import PreTokenizer


SPECIAL_TOKENS = ["<|endoftext|>"]


def time_call(fn: Callable[[], Any], repeats: int) -> tuple[Any, dict[str, float]]:
  samples = []
  result = None
  for _ in range(repeats):
    started = time.perf_counter()
    result = fn()
    samples.append(time.perf_counter() - started)
  return result, {
    "min_s": min(samples),
    "median_s": statistics.median(samples),
    "mean_s": statistics.mean(samples),
  }


def summarize_words(words: dict[str, int]) -> dict[str, int]:
  return {
    "unique_words": len(words),
    "occurrences": sum(words.values()),
  }


def profile(args: argparse.Namespace) -> dict[str, Any]:
  pretokenizer = PreTokenizer(SPECIAL_TOKENS, SPECIAL_TOKENS[0])
  file_size = args.input.stat().st_size

  boundaries, boundary_timing = time_call(
    lambda: pretokenizer.find_chunk_boundaries(
      args.input,
      args.chunks,
      chunk_size=args.chunk_size,
      boundary=args.boundary,
    ),
    args.repeats,
  )

  first_segments = boundaries[:args.segments]
  segment_results = []
  for offset, length in first_segments:
    words, timing = time_call(
      lambda offset=offset, length=length: pretokenizer.get_words_from_segment(args.input, offset, length),
      args.repeats,
    )
    segment_results.append({
      "offset": offset,
      "length": length,
      **timing,
      **summarize_words(words),
    })

  full_words, full_timing = time_call(
    lambda: pretokenizer.get_words_from_file(
      args.input,
      args.chunks,
      chunk_size=args.chunk_size,
      boundary=args.boundary,
    ),
    args.repeats,
  )

  return {
    "input": str(args.input),
    "bytes": file_size,
    "chunks": args.chunks,
    "chunk_size": args.chunk_size,
    "boundary": args.boundary,
    "repeats": args.repeats,
    "boundary_count": len(boundaries),
    "find_chunk_boundaries": boundary_timing,
    "segments": segment_results,
    "get_words_from_file": {
      **full_timing,
      **summarize_words(full_words),
    },
  }


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Profile unitoken pretokenizer phases on a raw text file.")
  parser.add_argument("input", type=Path)
  parser.add_argument("--chunks", type=int, default=64)
  parser.add_argument("--chunk-size", type=int)
  parser.add_argument("--boundary", choices=["auto", "eot", "line", "utf8"], default="auto")
  parser.add_argument("--segments", type=int, default=4)
  parser.add_argument("--repeats", type=int, default=1)
  parser.add_argument("--json", type=Path)
  args = parser.parse_args(argv)

  if args.chunks < 1:
    parser.error("--chunks must be at least 1")
  if args.chunk_size is not None and args.chunk_size < 1:
    parser.error("--chunk-size must be at least 1")
  if args.segments < 0:
    parser.error("--segments must be non-negative")
  if args.repeats < 1:
    parser.error("--repeats must be at least 1")

  result = profile(args)
  rendered = json.dumps(result, indent=2)
  print(rendered)
  if args.json:
    args.json.write_text(rendered + "\n", encoding="utf-8")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
