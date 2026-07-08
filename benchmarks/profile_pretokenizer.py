from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

from uni_tokenizer import BoundaryMode
from uni_tokenizer import PreTokenizer

from common import DEFAULT_CHUNK_SIZE
from common import SPECIAL_TOKENS
from common import add_report_args
from common import benchmark_metadata
from common import resolve_report_path
from common import write_report


SCRIPT_NAME = "profile_pretokenizer"


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
  boundary = cast(BoundaryMode, args.boundary)

  boundaries, boundary_timing = time_call(
    lambda: pretokenizer.find_chunk_boundaries(
      args.input,
      chunk_size=args.chunk_size,
      boundary=boundary,
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
      chunk_size=args.chunk_size,
      boundary=boundary,
    ),
    args.repeats,
  )

  return {
    "metadata": benchmark_metadata(
      contract="raw_text_unitoken_pretokenizer_profile",
      script_name=SCRIPT_NAME,
      dataset_name=args.dataset_name,
      config_name=args.config_name,
      experiment_name=args.experiment_name,
      notes=[
        "Profiles unitoken pretokenizer chunk boundary and word inventory phases.",
      ],
    ),
    "source": {
      "input_kind": "raw_text",
      "input": str(args.input),
      "bytes": file_size,
      "chunk_size": args.chunk_size,
      "boundary": boundary,
    },
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
  parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
  parser.add_argument("--boundary", choices=["auto", "eot", "line", "utf8"], default="auto")
  parser.add_argument("--segments", type=int, default=4)
  parser.add_argument("--repeats", type=int, default=1)
  add_report_args(parser)
  args = parser.parse_args(argv)

  if args.chunk_size < 1:
    parser.error("--chunk-size must be at least 1")
  if args.segments < 0:
    parser.error("--segments must be non-negative")
  if args.repeats < 1:
    parser.error("--repeats must be at least 1")

  result = profile(args)
  rendered = json.dumps(result, indent=2)
  if not args.quiet:
    print(rendered)
  write_report(resolve_report_path(args, script_name=SCRIPT_NAME), rendered)
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
