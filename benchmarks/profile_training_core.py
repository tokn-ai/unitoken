from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from uni_tokenizer import BpeTrainer

from common import SPECIAL_TOKENS
from common import add_report_args
from common import benchmark_metadata
from common import bucket_steps
from common import duration_summary
from common import load_words
from common import resolve_report_path
from common import write_report


SCRIPT_NAME = "profile_training_core"


def profile_training_core(words: Sequence[tuple[str, int]], vocab_size: int, bucket_size: int) -> dict[str, Any]:
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


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Profile unitoken BPE training core from a compressed word-frequency inventory.")
  parser.add_argument("--words", type=Path, required=True, help="JSON word-frequency inventory.")
  parser.add_argument("--vocab-size", type=int, default=10000)
  parser.add_argument("--max-occurrences", type=int, help="Truncate the weighted corpus for a faster smoke profile.")
  parser.add_argument("--bucket-size", type=int, default=500)
  add_report_args(parser)
  args = parser.parse_args(argv)

  if args.vocab_size < 1:
    parser.error("--vocab-size must be at least 1")
  if args.bucket_size < 1:
    parser.error("--bucket-size must be at least 1")

  words = load_words(args.words, args.max_occurrences)
  result = {
    "metadata": benchmark_metadata(
      contract="fixed_words_unitoken_training_core_profile",
      script_name=SCRIPT_NAME,
      dataset_name=args.dataset_name,
      config_name=args.config_name,
      experiment_name=args.experiment_name,
      notes=[
        "Unitoken receives compressed (word, frequency) pairs.",
        "This isolates training core phases and excludes pretokenization and external library comparisons.",
      ],
    ),
    "source": {
      "input_kind": "words_json",
      "words": str(args.words),
      "unique_words": len(words),
      "occurrences": sum(freq for _, freq in words),
      "unitoken_input_kind": "compressed_word_counts",
    },
    "target_vocab_size": args.vocab_size,
    "unitoken": profile_training_core(words, args.vocab_size, args.bucket_size),
  }

  rendered = json.dumps(result, indent=2)
  if not args.quiet:
    print(rendered)
  write_report(resolve_report_path(args, script_name=SCRIPT_NAME, vocab_size=args.vocab_size), rendered)
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
