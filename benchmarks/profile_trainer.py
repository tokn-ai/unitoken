from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from uni_tokenizer import BpeTrainer
from uni_tokenizer import BoundaryMode
from uni_tokenizer import PreTokenizer

from compare_hf_training import SPECIAL_TOKENS
from compare_hf_training import load_words
from compare_hf_training import train_hugging_face


DEFAULT_CHUNK_SIZE = 1024 * 1024


def percentile(values: Sequence[float], q: float) -> float | None:
  if not values:
    return None
  if len(values) == 1:
    return values[0]
  index = round((len(values) - 1) * q)
  return sorted(values)[index]


def duration_summary(values: Sequence[float]) -> dict[str, Any]:
  if not values:
    return {
      "count": 0,
      "total_s": 0.0,
      "min_s": None,
      "median_s": None,
      "p90_s": None,
      "p99_s": None,
      "max_s": None,
    }
  return {
    "count": len(values),
    "total_s": sum(values),
    "min_s": min(values),
    "median_s": statistics.median(values),
    "p90_s": percentile(values, 0.90),
    "p99_s": percentile(values, 0.99),
    "max_s": max(values),
  }


def bucket_steps(step_times: Sequence[float], bucket_size: int) -> list[dict[str, Any]]:
  buckets = []
  for start in range(0, len(step_times), bucket_size):
    chunk = step_times[start:start + bucket_size]
    buckets.append({
      "first_step": start + 1,
      "last_step": start + len(chunk),
      **duration_summary(chunk),
    })
  return buckets


def read_words_from_text(path: Path, chunk_size: int, boundary: BoundaryMode) -> tuple[list[tuple[str, int]], dict[str, Any]]:
  started = time.perf_counter()
  pretokenizer = PreTokenizer(SPECIAL_TOKENS, SPECIAL_TOKENS[0])
  words = pretokenizer.get_words_from_file(path, chunk_size=chunk_size, boundary=boundary)
  elapsed_s = time.perf_counter() - started
  return list(words.items()), {
    "pretokenize_s": elapsed_s,
    "unique_words": len(words),
    "occurrences": sum(words.values()),
  }


def profile_unitoken(words: Sequence[tuple[str, int]], vocab_size: int, bucket_size: int) -> dict[str, Any]:
  gc.collect()
  trainer = BpeTrainer(SPECIAL_TOKENS, ch="u8", initial_alphabet="byte_level")

  started = time.perf_counter()
  trainer.add_words(words)
  add_words_s = time.perf_counter() - started
  initial_vocab_size = trainer.vocab_size

  started = time.perf_counter()
  trainer.init_training()
  init_training_s = time.perf_counter() - started

  step_times = []
  while trainer.vocab_size < vocab_size:
    started = time.perf_counter()
    next_vocab_size = trainer.step()
    step_times.append(time.perf_counter() - started)
    if next_vocab_size is None:
      break

  return {
    "vocab_size": trainer.vocab_size,
    "initial_vocab_size": initial_vocab_size,
    "add_words_s": add_words_s,
    "init_training_s": init_training_s,
    "step_summary": duration_summary(step_times),
    "step_buckets": bucket_steps(step_times, bucket_size),
    "total_train_s": add_words_s + init_training_s + sum(step_times),
  }


def profile_hugging_face(words: Sequence[tuple[str, int]], vocab_size: int) -> dict[str, Any]:
  gc.collect()
  started = time.perf_counter()
  result = train_hugging_face(words, vocab_size)
  return {
    "vocab_size": result["vocab_size"],
    "total_train_s": time.perf_counter() - started,
  }


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Profile unitoken BPE trainer phases against Hugging Face.")
  input_group = parser.add_mutually_exclusive_group(required=True)
  input_group.add_argument("--words", type=Path, help="JSON word-frequency fixture.")
  input_group.add_argument("--text", type=Path, help="Raw UTF-8 text file to pretokenize before profiling.")
  parser.add_argument("--vocab-size", type=int, default=10000)
  parser.add_argument("--max-occurrences", type=int, help="Truncate the weighted corpus for a faster smoke profile.")
  parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
  parser.add_argument("--boundary", choices=["auto", "eot", "line", "utf8"], default="auto")
  parser.add_argument("--bucket-size", type=int, default=500)
  parser.add_argument("--skip-hf", action="store_true", help="Only profile unitoken.")
  parser.add_argument("--json", type=Path)
  args = parser.parse_args(argv)

  if args.vocab_size < 1:
    parser.error("--vocab-size must be at least 1")
  if args.chunk_size < 1:
    parser.error("--chunk-size must be at least 1")
  if args.bucket_size < 1:
    parser.error("--bucket-size must be at least 1")

  source: dict[str, Any]
  if args.text:
    boundary = cast(BoundaryMode, args.boundary)
    words, source = read_words_from_text(args.text, args.chunk_size, boundary)
    source.update({
      "text": str(args.text),
      "text_bytes": args.text.stat().st_size,
      "boundary": boundary,
      "chunk_size": args.chunk_size,
    })
  else:
    words = load_words(args.words, args.max_occurrences)
    source = {
      "words": str(args.words),
      "unique_words": len(words),
      "occurrences": sum(freq for _, freq in words),
    }

  result = {
    "source": source,
    "target_vocab_size": args.vocab_size,
    "unitoken": profile_unitoken(words, args.vocab_size, args.bucket_size),
  }
  if not args.skip_hf:
    result["huggingface"] = profile_hugging_face(words, args.vocab_size)

  rendered = json.dumps(result, indent=2)
  print(rendered)
  if args.json:
    args.json.write_text(rendered + "\n", encoding="utf-8")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
