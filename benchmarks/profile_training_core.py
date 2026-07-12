from __future__ import annotations

import argparse
import ctypes
import gc
import json
import resource
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from uni_tokenizer import BpeTrainer

from common import SPECIAL_TOKENS
from common import add_report_args
from common import benchmark_metadata
from common import bigram_frequency_guard
from common import bucket_steps
from common import duration_summary
from common import load_words
from common import load_word_inventory_manifest
from common import resolve_report_path
from common import word_inventory_manifest_path
from common import write_report


SCRIPT_NAME = "profile_training_core"


def peak_rss_bytes() -> int:
  peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  return peak if sys.platform == "darwin" else peak * 1024


def current_rss_bytes() -> int:
  if sys.platform.startswith("linux"):
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return resident_pages * resource.getpagesize()
  if sys.platform == "darwin":
    class MachTaskBasicInfo(ctypes.Structure):
      _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("resident_size_max", ctypes.c_uint64),
        ("user_time_seconds", ctypes.c_int32),
        ("user_time_microseconds", ctypes.c_int32),
        ("system_time_seconds", ctypes.c_int32),
        ("system_time_microseconds", ctypes.c_int32),
        ("policy", ctypes.c_int32),
        ("suspend_count", ctypes.c_int32),
      ]

    lib_system = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    info = MachTaskBasicInfo()
    count = ctypes.c_uint32(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint32))
    result = lib_system.task_info(
      lib_system.mach_task_self(),
      20,
      ctypes.byref(info),
      ctypes.byref(count),
    )
    if result != 0:
      raise OSError(f"task_info failed with Mach error {result}")
    return info.resident_size
  raise RuntimeError(f"current RSS is unsupported on {sys.platform}")


def profile_training_core(
  words: list[tuple[str, int]],
  vocab_size: int,
  bucket_size: int,
  unit: str,
  hot_pair_window_size: int | None,
  rss_after_load_bytes: int,
) -> dict[str, Any]:
  gc.collect()
  trainer = BpeTrainer(
    SPECIAL_TOKENS,
    unit=unit,
    initial_alphabet="byte_level" if unit == "byte" else None,
    hot_pair_window_size=hot_pair_window_size,
  )

  started = time.perf_counter()
  trainer.add_words(words)
  add_words_s = time.perf_counter() - started
  initial_vocab_size = trainer.vocab_size
  rss_after_transfer_bytes = current_rss_bytes()
  words.clear()
  gc.collect()
  rss_after_source_release_bytes = current_rss_bytes()

  started = time.perf_counter()
  trainer.init_training()
  init_training_s = time.perf_counter() - started
  rss_after_init_training_bytes = current_rss_bytes()

  step_times = []
  training_rss_samples = [rss_after_init_training_bytes]
  while trainer.vocab_size < vocab_size:
    started = time.perf_counter()
    next_vocab_size = trainer.step()
    step_times.append(time.perf_counter() - started)
    if next_vocab_size is None:
      break
    if len(step_times) % bucket_size == 0:
      training_rss_samples.append(current_rss_bytes())
  training_rss_samples.append(current_rss_bytes())

  return {
    "vocab_size": trainer.vocab_size,
    "final_merge_freq": trainer.last_merge_freq,
    "initial_vocab_size": initial_vocab_size,
    "add_words_s": add_words_s,
    "init_training_s": init_training_s,
    "step_summary": duration_summary(step_times),
    "step_buckets": bucket_steps(step_times, bucket_size),
    "total_train_s": add_words_s + init_training_s + sum(step_times),
    "process_peak_rss_bytes": peak_rss_bytes(),
    "rss_phases_bytes": {
      "after_load": rss_after_load_bytes,
      "after_transfer": rss_after_transfer_bytes,
      "after_source_release": rss_after_source_release_bytes,
      "after_init_training": rss_after_init_training_bytes,
      "observed_training_peak": max(training_rss_samples),
      "after_training": training_rss_samples[-1],
    },
    "hot_pair_window_stats": trainer.hot_pair_window_stats,
  }


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Profile unitoken BPE training core from a compressed word-frequency inventory.")
  parser.add_argument("--words", type=Path, required=True, help="JSON word-frequency inventory.")
  parser.add_argument("--vocab-size", type=int, default=10000)
  parser.add_argument("--unit", choices=["byte", "unicode"], default="byte", help="BPE unit used for training.")
  parser.add_argument("--max-occurrences", type=int, help="Truncate the weighted corpus for a faster smoke profile.")
  parser.add_argument("--bucket-size", type=int, default=500)
  parser.add_argument(
    "--hot-pair-window-size",
    type=int,
    help="Retain occurrence postings for only the exact top-K pair window.",
  )
  add_report_args(parser)
  args = parser.parse_args(argv)

  if args.vocab_size < 1:
    parser.error("--vocab-size must be at least 1")
  if args.bucket_size < 1:
    parser.error("--bucket-size must be at least 1")
  if args.hot_pair_window_size is not None and args.hot_pair_window_size < 1:
    parser.error("--hot-pair-window-size must be at least 1")

  words = load_words(args.words, args.max_occurrences)
  unique_words = len(words)
  occurrences = sum(freq for _, freq in words)
  rss_after_load_bytes = current_rss_bytes()
  manifest = load_word_inventory_manifest(args.words)
  unitoken_result = profile_training_core(
    words,
    args.vocab_size,
    args.bucket_size,
    args.unit,
    args.hot_pair_window_size,
    rss_after_load_bytes,
  )
  guard = (
    bigram_frequency_guard(unitoken_result["final_merge_freq"], manifest)
    if args.max_occurrences is None
    else None
  )
  if guard is not None:
    unitoken_result["bigram_frequency_guard"] = guard
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
      "unique_words": unique_words,
      "occurrences": occurrences,
      "unitoken_input_kind": "compressed_word_counts",
      "unit": args.unit,
      "hot_pair_window_size": args.hot_pair_window_size,
      "word_inventory_manifest_path": (
        str(word_inventory_manifest_path(args.words))
        if manifest is not None
        else None
      ),
      "word_inventory_manifest": manifest,
    },
    "target_vocab_size": args.vocab_size,
    "unitoken": unitoken_result,
  }

  rendered = json.dumps(result, indent=2)
  if not args.quiet:
    print(rendered)
  write_report(resolve_report_path(args, script_name=SCRIPT_NAME, vocab_size=args.vocab_size), rendered)
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
