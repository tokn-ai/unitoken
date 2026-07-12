from __future__ import annotations

import importlib
import json
import platform
import statistics
import subprocess
import sys
from argparse import ArgumentParser
from argparse import Namespace
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SPECIAL_TOKENS = ["<|endoftext|>"]
DEFAULT_CHUNK_SIZE = 1024 * 1024
DEFAULT_BENCHMARK_DIR = Path("out") / "benchmarks"
WORD_INVENTORY_MANIFEST_CONTRACT = "unitoken_word_inventory_manifest_v1"


def word_inventory_manifest_path(words_path: Path) -> Path:
  return words_path.with_name(f"{words_path.stem}.manifest.json")


def write_word_inventory_manifest(
  words_path: Path,
  *,
  source: dict[str, Any],
  pretokenizer: dict[str, Any],
  unicode_bigrams: dict[str, Any] | None,
  unique_words: int,
  weighted_occurrences: int | None,
) -> Path:
  manifest_path = word_inventory_manifest_path(words_path)
  manifest = {
    "contract": WORD_INVENTORY_MANIFEST_CONTRACT,
    "words": {
      "file_name": words_path.name,
      "bytes": words_path.stat().st_size,
      "unique_words": unique_words,
      "weighted_occurrences": weighted_occurrences,
    },
    "source": source,
    "pretokenizer": pretokenizer,
    "unicode_bigrams": unicode_bigrams,
  }
  manifest_path.write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
  )
  return manifest_path


def load_word_inventory_manifest(words_path: Path) -> dict[str, Any] | None:
  manifest_path = word_inventory_manifest_path(words_path)
  if not manifest_path.exists():
    return None
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  if manifest.get("contract") != WORD_INVENTORY_MANIFEST_CONTRACT:
    raise ValueError(f"unsupported word inventory manifest: {manifest_path}")
  if manifest.get("words", {}).get("file_name") != words_path.name:
    raise ValueError(f"word inventory manifest does not describe {words_path}")
  if manifest.get("words", {}).get("bytes") != words_path.stat().st_size:
    raise ValueError(f"word inventory manifest byte size does not match {words_path}")
  return manifest


def bigram_frequency_guard(
  final_merge_freq: int | None,
  manifest: dict[str, Any] | None,
) -> dict[str, int | bool | None] | None:
  if manifest is None or manifest.get("unicode_bigrams") is None:
    return None
  bigrams = manifest["unicode_bigrams"]
  cutoff_freq = bigrams.get("cutoff_freq")
  max_excluded_freq = bigrams.get("max_excluded_freq")
  return {
    "cutoff_freq": cutoff_freq,
    "max_excluded_freq": max_excluded_freq,
    "final_merge_freq": final_merge_freq,
    "final_merge_minus_bigram_cutoff": (
      final_merge_freq - cutoff_freq
      if final_merge_freq is not None and cutoff_freq is not None
      else None
    ),
    "final_merge_above_bigram_cutoff": (
      final_merge_freq > cutoff_freq
      if final_merge_freq is not None and cutoff_freq is not None
      else None
    ),
    "final_merge_above_max_excluded_freq": (
      final_merge_freq > max_excluded_freq
      if final_merge_freq is not None and max_excluded_freq is not None
      else None
    ),
  }


def load_words(path: Path, max_occurrences: int | None) -> list[tuple[str, int]]:
  raw_words = json.loads(path.read_text(encoding="utf-8"))
  words = list(raw_words.items())

  if max_occurrences is None:
    return words

  if max_occurrences <= 0:
    return []

  total = sum(freq for _, freq in words)
  if total <= max_occurrences:
    return words

  if max_occurrences < len(words):
    return [(word, 1) for word, _ in words[:max_occurrences]]

  base_total = len(words)
  remaining = max_occurrences - base_total
  scaled = []
  fractions = []
  assigned = 0
  for index, (word, freq) in enumerate(words):
    exact_extra = remaining * freq / total
    extra = int(exact_extra)
    assigned += extra
    scaled.append((word, 1 + extra))
    fractions.append((exact_extra - extra, index))

  leftover = remaining - assigned
  extra_indexes = {
    index
    for _, index in sorted(fractions, reverse=True)[:leftover]
  }
  return [
    (word, freq + (1 if index in extra_indexes else 0))
    for index, (word, freq) in enumerate(scaled)
  ]


def percentile(values: list[float] | tuple[float, ...], q: float) -> float | None:
  if not values:
    return None
  if len(values) == 1:
    return values[0]
  index = round((len(values) - 1) * q)
  return sorted(values)[index]


def duration_summary(values: list[float] | tuple[float, ...]) -> dict[str, Any]:
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


def bucket_steps(step_times: list[float], bucket_size: int) -> list[dict[str, Any]]:
  buckets = []
  for start in range(0, len(step_times), bucket_size):
    chunk = step_times[start:start + bucket_size]
    buckets.append({
      "first_step": start + 1,
      "last_step": start + len(chunk),
      **duration_summary(chunk),
    })
  return buckets


def git_sha() -> str | None:
  try:
    return subprocess.check_output(
      ["git", "rev-parse", "HEAD"],
      cwd=REPO_ROOT,
      text=True,
      stderr=subprocess.DEVNULL,
    ).strip()
  except (OSError, subprocess.CalledProcessError):
    return None


def module_version(name: str) -> str | None:
  try:
    module = importlib.import_module(name)
  except ImportError:
    return None
  return getattr(module, "__version__", None)


def unitoken_extension_path() -> str | None:
  try:
    module = importlib.import_module("uni_tokenizer._lib")
  except ImportError:
    return None
  module_file = getattr(module, "__file__", None)
  if module_file is None:
    return None
  return str(Path(module_file).resolve())


def add_report_args(parser: ArgumentParser) -> None:
  parser.add_argument("--dataset-name", required=True, help="Short dataset name used in benchmark report filenames.")
  parser.add_argument("--config-name", default="default", help="Short config key used in benchmark report filenames.")
  parser.add_argument("--experiment-name", default="baseline", help="Short experiment key used in benchmark report filenames.")
  parser.add_argument("--json-dir", type=Path, default=DEFAULT_BENCHMARK_DIR, help="Root directory for generated benchmark JSON reports.")
  parser.add_argument("--json", type=Path, help="Explicit JSON report path. Overrides --json-dir and naming fields.")
  parser.add_argument("--quiet", action="store_true", help="Write the JSON report without printing it to stdout.")


def report_path(
  *,
  script_name: str,
  dataset_name: str,
  config_name: str,
  experiment_name: str,
  vocab_size: int | None = None,
  json_dir: Path = DEFAULT_BENCHMARK_DIR,
) -> Path:
  filename = f"{dataset_name}.{config_name}.{experiment_name}"
  if vocab_size is not None:
    filename += f".vocab{vocab_size}"
  filename += ".json"
  return json_dir / script_name / filename


def resolve_report_path(args: Namespace, *, script_name: str, vocab_size: int | None = None) -> Path:
  if args.json:
    return args.json
  return report_path(
    script_name=script_name,
    dataset_name=args.dataset_name,
    config_name=args.config_name,
    experiment_name=args.experiment_name,
    vocab_size=vocab_size,
    json_dir=args.json_dir,
  )


def write_report(path: Path, rendered: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(rendered + "\n", encoding="utf-8")


def benchmark_metadata(
  *,
  contract: str,
  script_name: str,
  dataset_name: str,
  config_name: str,
  experiment_name: str,
  notes: list[str] | None = None,
) -> dict[str, Any]:
  return {
    "benchmark_contract": contract,
    "script_name": script_name,
    "dataset_name": dataset_name,
    "config_name": config_name,
    "experiment_name": experiment_name,
    "notes": notes or [],
    "generated_at": datetime.now(UTC).isoformat(),
    "git_sha": git_sha(),
    "python": {
      "executable": sys.executable,
      "version": platform.python_version(),
    },
    "packages": {
      "uni_tokenizer": {
        "version": module_version("uni_tokenizer"),
        "extension_path": unitoken_extension_path(),
      },
      "tokenizers": {
        "version": module_version("tokenizers"),
      },
    },
  }
