from __future__ import annotations

import argparse
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
from uni_tokenizer import UnicodeBigramMixedBoundary

from common import DEFAULT_CHUNK_SIZE
from common import SPECIAL_TOKENS
from common import add_report_args
from common import benchmark_metadata
from common import resolve_report_path
from common import write_report


SCRIPT_NAME = "unicode_bigram_split"


def duration_summary(values: Sequence[float]) -> dict[str, Any]:
  if not values:
    return {
      "count": 0,
      "total_s": 0.0,
      "min_s": None,
      "median_s": None,
      "max_s": None,
    }
  return {
    "count": len(values),
    "total_s": sum(values),
    "min_s": min(values),
    "median_s": statistics.median(values),
    "max_s": max(values),
  }


def word_length_summary(words: dict[str, int]) -> dict[str, Any]:
  lengths = [len(word) for word in words]
  if not lengths:
    return {
      "min": None,
      "median": None,
      "p90": None,
      "max": None,
    }
  lengths.sort()
  return {
    "min": lengths[0],
    "median": statistics.median(lengths),
    "p90": lengths[round((len(lengths) - 1) * 0.90)],
    "max": lengths[-1],
  }


def inventory_summary(words: dict[str, int]) -> dict[str, Any]:
  frequencies = list(words.values())
  singleton_count = sum(1 for freq in frequencies if freq == 1)
  return {
    "unique_words": len(words),
    "occurrences": sum(frequencies),
    "singleton_count": singleton_count,
    "singleton_ratio": singleton_count / len(words) if words else None,
    "word_length": word_length_summary(words),
  }


def save_words(path: Path, words: dict[str, int]) -> None:
  sorted_words = dict(sorted(words.items(), key=lambda item: (item[1], item[0]), reverse=True))
  path.write_text(json.dumps(sorted_words, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def train_unitoken(words: dict[str, int], vocab_size: int) -> dict[str, Any]:
  trainer = BpeTrainer(SPECIAL_TOKENS, ch="u8", initial_alphabet="byte_level")
  started = time.perf_counter()
  trainer.add_words(list(words.items()))
  add_words_s = time.perf_counter() - started

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
    "add_words_s": add_words_s,
    "init_training_s": init_training_s,
    "step_summary": duration_summary(step_times),
    "total_train_s": add_words_s + init_training_s + sum(step_times),
  }


def collect_words(pretokenizer: PreTokenizer, path: Path, chunk_size: int, boundary: BoundaryMode) -> tuple[dict[str, int], float]:
  started = time.perf_counter()
  words = pretokenizer.get_words_from_file(path, chunk_size=chunk_size, boundary=boundary)
  return words, time.perf_counter() - started


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Experiment with Unicode bigram pre-splitting.")
  parser.add_argument("--text", type=Path, required=True)
  parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
  parser.add_argument("--boundary", choices=["auto", "eot", "line", "utf8"], default="auto")
  parser.add_argument("--top-k", type=int, default=100_000)
  parser.add_argument("--min-freq", type=int, default=16)
  parser.add_argument("--unicode-bigram-mixed-boundary", choices=["keep", "split"], default="keep")
  parser.add_argument("--vocab-size", type=int, help="Optionally train unitoken on both inventories.")
  parser.add_argument("--save-words", type=Path, help="Save the Unicode bigram split word-frequency inventory as JSON.")
  add_report_args(parser)
  args = parser.parse_args(argv)

  if args.chunk_size < 1:
    parser.error("--chunk-size must be at least 1")
  if args.top_k < 1:
    parser.error("--top-k must be at least 1")
  if args.min_freq < 1:
    parser.error("--min-freq must be at least 1")

  boundary = cast(BoundaryMode, args.boundary)
  unicode_bigram_mixed_boundary = cast(UnicodeBigramMixedBoundary, args.unicode_bigram_mixed_boundary)
  base = PreTokenizer(SPECIAL_TOKENS, SPECIAL_TOKENS[0])

  started = time.perf_counter()
  unicode_bigrams = base.build_unicode_bigrams_from_file(
    args.text,
    chunk_size=args.chunk_size,
    boundary=boundary,
    top_k=args.top_k,
    min_freq=args.min_freq,
  )
  build_bigrams_s = time.perf_counter() - started

  baseline_words, baseline_pretokenize_s = collect_words(base, args.text, args.chunk_size, boundary)
  split = PreTokenizer(
    SPECIAL_TOKENS,
    SPECIAL_TOKENS[0],
    unicode_bigrams=unicode_bigrams,
    unicode_bigram_mixed_boundary=unicode_bigram_mixed_boundary,
  )
  split_words, split_pretokenize_s = collect_words(split, args.text, args.chunk_size, boundary)
  if args.save_words:
    args.save_words.parent.mkdir(parents=True, exist_ok=True)
    save_words(args.save_words, split_words)

  result: dict[str, Any] = {
    "metadata": benchmark_metadata(
      contract="unicode_bigram_split_experiment",
      script_name=SCRIPT_NAME,
      dataset_name=args.dataset_name,
      config_name=args.config_name,
      experiment_name=args.experiment_name,
      notes=[
        "Compares baseline pretokenization with unicode-bigram-guided splitting.",
      ],
    ),
    "source": {
      "input_kind": "raw_text",
      "text": str(args.text),
      "text_bytes": args.text.stat().st_size,
      "chunk_size": args.chunk_size,
      "boundary": boundary,
    },
    "unicode_bigram": {
      "top_k": args.top_k,
      "min_freq": args.min_freq,
      "mixed_boundary": unicode_bigram_mixed_boundary,
      "retained": len(unicode_bigrams),
      "build_s": build_bigrams_s,
    },
    "baseline": {
      "pretokenize_s": baseline_pretokenize_s,
      **inventory_summary(baseline_words),
    },
    "unicode_bigram_split": {
      "pretokenize_s": split_pretokenize_s,
      **inventory_summary(split_words),
    },
  }

  if args.vocab_size:
    result["target_vocab_size"] = args.vocab_size
    result["baseline"]["training"] = train_unitoken(baseline_words, args.vocab_size)
    result["unicode_bigram_split"]["training"] = train_unitoken(split_words, args.vocab_size)

  rendered = json.dumps(result, indent=2)
  if not args.quiet:
    print(rendered)
  write_report(resolve_report_path(args, script_name=SCRIPT_NAME, vocab_size=args.vocab_size), rendered)
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
