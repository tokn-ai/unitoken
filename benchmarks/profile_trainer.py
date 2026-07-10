from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from uni_tokenizer import BpeTrainer
from uni_tokenizer import BoundaryMode
from uni_tokenizer import PreTokenizer

from common import DEFAULT_CHUNK_SIZE
from common import SPECIAL_TOKENS
from common import add_report_args
from common import benchmark_metadata
from common import bucket_steps
from common import duration_summary
from common import resolve_report_path
from common import write_report


SCRIPT_NAME = "profile_trainer"


def pretokenize(path: Path, chunk_size: int, boundary: BoundaryMode) -> tuple[list[tuple[str, int]], dict[str, Any]]:
  gc.collect()
  started = time.perf_counter()
  pretokenizer = PreTokenizer(SPECIAL_TOKENS, SPECIAL_TOKENS[0])
  words = pretokenizer.get_words_from_file(path, chunk_size=chunk_size, boundary=boundary)
  pretokenize_s = time.perf_counter() - started
  return list(words.items()), {
    "pretokenize_s": pretokenize_s,
    "unique_words": len(words),
    "occurrences": sum(words.values()),
  }


def train(words: Sequence[tuple[str, int]], vocab_size: int, bucket_size: int) -> dict[str, Any]:
  gc.collect()
  trainer = BpeTrainer(SPECIAL_TOKENS, unit="byte", initial_alphabet="byte_level")

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

  train_s = add_words_s + init_training_s + sum(step_times)
  return {
    "vocab_size": trainer.vocab_size,
    "initial_vocab_size": initial_vocab_size,
    "add_words_s": add_words_s,
    "init_training_s": init_training_s,
    "step_summary": duration_summary(step_times),
    "step_buckets": bucket_steps(step_times, bucket_size),
    "train_s": train_s,
  }


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Profile end-to-end unitoken BPE training from raw UTF-8 text.")
  parser.add_argument("--text", type=Path, required=True, help="Raw UTF-8 text file.")
  parser.add_argument("--vocab-size", type=int, default=10000)
  parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
  parser.add_argument("--boundary", choices=["auto", "eot", "line", "utf8"], default="auto")
  parser.add_argument("--bucket-size", type=int, default=500)
  add_report_args(parser)
  args = parser.parse_args(argv)

  if args.vocab_size < 1:
    parser.error("--vocab-size must be at least 1")
  if args.chunk_size < 1:
    parser.error("--chunk-size must be at least 1")
  if args.bucket_size < 1:
    parser.error("--bucket-size must be at least 1")

  boundary = cast(BoundaryMode, args.boundary)
  words, pretokenizer_result = pretokenize(args.text, args.chunk_size, boundary)
  trainer_result = train(words, args.vocab_size, args.bucket_size)

  result = {
    "metadata": benchmark_metadata(
      contract="raw_text_unitoken_trainer_profile",
      script_name=SCRIPT_NAME,
      dataset_name=args.dataset_name,
      config_name=args.config_name,
      experiment_name=args.experiment_name,
      notes=[
        "This is an end-to-end unitoken training profile from raw text.",
        "Timing includes pretokenization, add_words, init_training, and BPE steps.",
      ],
    ),
    "source": {
      "input_kind": "raw_text",
      "text": str(args.text),
      "text_bytes": args.text.stat().st_size,
      "boundary": boundary,
      "chunk_size": args.chunk_size,
    },
    "target_vocab_size": args.vocab_size,
    "pretokenizer": pretokenizer_result,
    "unitoken": {
      **trainer_result,
      "total_s": pretokenizer_result["pretokenize_s"] + trainer_result["train_s"],
    },
  }

  rendered = json.dumps(result, indent=2)
  if not args.quiet:
    print(rendered)
  write_report(resolve_report_path(args, script_name=SCRIPT_NAME, vocab_size=args.vocab_size), rendered)
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
