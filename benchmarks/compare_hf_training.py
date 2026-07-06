from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer
from tokenizers import models
from tokenizers import pre_tokenizers
from tokenizers import trainers

from uni_tokenizer import BpeTrainer
from uni_tokenizer import PreTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORDS = REPO_ROOT / "fixtures" / "_words.tinystories_sample_5M.json"
SPECIAL_TOKENS = ["<|endoftext|>"]


def bytes_to_unicode() -> dict[int, str]:
  bs = (
    list(range(ord("!"), ord("~") + 1))
    + list(range(ord("¡"), ord("¬") + 1))
    + list(range(ord("®"), ord("ÿ") + 1))
  )
  cs = bs[:]
  n = 0
  for byte in range(256):
    if byte not in bs:
      bs.append(byte)
      cs.append(256 + n)
      n += 1
  return dict(zip(bs, map(chr, cs)))


BYTE_ENCODER = bytes_to_unicode()


def to_byte_level_token(token: bytes) -> str:
  return "".join(BYTE_ENCODER[byte] for byte in token)


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
  scaled = [
    (word, freq + (1 if index in extra_indexes else 0))
    for index, (word, freq) in enumerate(scaled)
  ]
  return scaled


def expanded_words(words: Sequence[tuple[str, int]]) -> Iterable[str]:
  for word, freq in words:
    for _ in range(freq):
      yield word


def train_unitoken(words: Sequence[tuple[str, int]], vocab_size: int) -> dict[str, Any]:
  trainer = BpeTrainer(SPECIAL_TOKENS, ch="u8", initial_alphabet="byte_level")
  trainer.add_words(words)
  trainer.train(vocab_size)
  vocab = {
    to_byte_level_token(token): rank
    for token, rank in dict(trainer.vocabs.items()).items()
  }
  return {
    "vocab": vocab,
    "vocab_size": trainer.vocab_size,
  }


def train_hugging_face(words: Sequence[tuple[str, int]], vocab_size: int) -> dict[str, Any]:
  tokenizer = Tokenizer(models.BPE())
  tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
  trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    special_tokens=SPECIAL_TOKENS,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
  )
  tokenizer.train_from_iterator(
    expanded_words(words),
    trainer=trainer,
    length=sum(freq for _, freq in words),
  )
  return {
    "vocab": tokenizer.get_vocab(),
    "vocab_size": tokenizer.get_vocab_size(),
  }


def iter_text_chunks(path: Path, chunk_bytes: int) -> Iterable[str]:
  with path.open("r", encoding="utf-8") as file:
    while True:
      chunk = file.read(chunk_bytes)
      if not chunk:
        break
      yield chunk


def train_unitoken_from_text(path: Path, vocab_size: int, num_chunks: int) -> dict[str, Any]:
  started = time.perf_counter()
  pretokenizer = PreTokenizer(SPECIAL_TOKENS, SPECIAL_TOKENS[0])
  words = pretokenizer.get_words_from_file(path, num_chunks)
  pretokenize_s = time.perf_counter() - started

  started = time.perf_counter()
  train_result = train_unitoken(list(words.items()), vocab_size)
  train_s = time.perf_counter() - started
  train_result.update({
    "pretokenize_s": pretokenize_s,
    "train_s": train_s,
    "unique_words": len(words),
    "occurrences": sum(words.values()),
  })
  return train_result


def train_hugging_face_from_text(path: Path, vocab_size: int, chunk_bytes: int) -> dict[str, Any]:
  tokenizer = Tokenizer(models.BPE())
  tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
  trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    special_tokens=SPECIAL_TOKENS,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
  )
  tokenizer.train_from_iterator(
    iter_text_chunks(path, chunk_bytes),
    trainer=trainer,
  )
  return {
    "vocab": tokenizer.get_vocab(),
    "vocab_size": tokenizer.get_vocab_size(),
  }


def time_call(label: str, fn: Callable[[], dict[str, Any]], repeats: int) -> dict[str, Any]:
  samples = []
  result = None
  for _ in range(repeats):
    gc.collect()
    started = time.perf_counter()
    result = fn()
    samples.append(time.perf_counter() - started)

  assert result is not None
  timed = {
    "label": label,
    "repeats": repeats,
    "min_s": min(samples),
    "median_s": statistics.median(samples),
    "mean_s": statistics.mean(samples),
    "vocab_size": result["vocab_size"],
    "vocab": result["vocab"],
  }
  for key, value in result.items():
    if key in timed or key == "vocab":
      continue
    if isinstance(value, (int, float, str, bool, type(None))):
      timed[key] = value
  return timed


def without_vocab(result: dict[str, Any]) -> dict[str, Any]:
  return {key: value for key, value in result.items() if key != "vocab"}


def run_words(args: argparse.Namespace) -> dict[str, Any]:
  words = load_words(args.words, args.max_occurrences)
  occurrence_count = sum(freq for _, freq in words)

  unitoken = time_call(
    "unitoken.train",
    lambda: train_unitoken(words, args.vocab_size),
    args.repeats,
  )
  hf = time_call(
    "huggingface.train",
    lambda: train_hugging_face(words, args.vocab_size),
    args.repeats,
  )

  same_vocab = unitoken["vocab"] == hf["vocab"]
  unitoken_median = unitoken["median_s"]
  hf_median = hf["median_s"]
  speedup = hf_median / unitoken_median if unitoken_median else None

  results = {
    "words": str(args.words),
    "unique_words": len(words),
    "occurrences": occurrence_count,
    "target_vocab_size": args.vocab_size,
    "same_vocab": same_vocab,
    "speedup_hf_over_unitoken_median": speedup,
    "benchmarks": [
      without_vocab(unitoken),
      without_vocab(hf),
    ],
  }

  if not same_vocab:
    unitoken_vocab = unitoken["vocab"]
    hf_vocab = hf["vocab"]
    results["vocab_diff"] = {
      "unitoken_only": sorted(set(unitoken_vocab) - set(hf_vocab))[:args.diff_limit],
      "huggingface_only": sorted(set(hf_vocab) - set(unitoken_vocab))[:args.diff_limit],
      "rank_mismatches": [
        {
          "token": token,
          "unitoken_rank": unitoken_vocab[token],
          "huggingface_rank": hf_vocab[token],
        }
        for token in sorted(set(unitoken_vocab) & set(hf_vocab))
        if unitoken_vocab[token] != hf_vocab[token]
      ][:args.diff_limit],
    }

  return results


def run_text(args: argparse.Namespace) -> dict[str, Any]:
  unitoken = time_call(
    "unitoken.raw_train",
    lambda: train_unitoken_from_text(args.text, args.vocab_size, args.chunks),
    args.repeats,
  )
  hf = time_call(
    "huggingface.raw_train",
    lambda: train_hugging_face_from_text(args.text, args.vocab_size, args.hf_chunk_bytes),
    args.repeats,
  )

  same_vocab = unitoken["vocab"] == hf["vocab"]
  unitoken_median = unitoken["median_s"]
  hf_median = hf["median_s"]
  speedup = hf_median / unitoken_median if unitoken_median else None
  unitoken_train_s = unitoken.get("train_s")
  train_phase_speedup = hf_median / unitoken_train_s if unitoken_train_s else None

  return {
    "text": str(args.text),
    "text_bytes": args.text.stat().st_size,
    "target_vocab_size": args.vocab_size,
    "same_vocab": same_vocab,
    "speedup_hf_over_unitoken_median": speedup,
    "speedup_hf_over_unitoken_train_phase": train_phase_speedup,
    "benchmarks": [
      without_vocab(unitoken),
      without_vocab(hf),
    ],
    "note": "Raw-text mode includes unitoken pretokenization plus training. Hugging Face receives fixed-size text chunks, so vocab parity can differ at chunk boundaries.",
  }


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Benchmark unitoken BPE training against Hugging Face tokenizers.")
  input_group = parser.add_mutually_exclusive_group()
  input_group.add_argument("--words", type=Path, help="JSON word-frequency fixture.")
  input_group.add_argument("--text", type=Path, help="Raw UTF-8 text file for end-to-end training.")
  parser.add_argument("--vocab-size", type=int, default=2000)
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--max-occurrences", type=int, help="Truncate the weighted corpus for a faster smoke benchmark.")
  parser.add_argument("--chunks", type=int, default=1024, help="Desired unitoken pretokenizer chunks in --text mode.")
  parser.add_argument("--hf-chunk-bytes", type=int, default=8 * 1024 * 1024, help="Text chunk size yielded to Hugging Face in --text mode.")
  parser.add_argument("--diff-limit", type=int, default=10)
  parser.add_argument("--json", type=Path)
  args = parser.parse_args(argv)
  if args.words is None and args.text is None:
    args.words = DEFAULT_WORDS
  if args.repeats < 1:
    parser.error("--repeats must be at least 1")
  if args.chunks < 1:
    parser.error("--chunks must be at least 1")
  if args.hf_chunk_bytes < 1:
    parser.error("--hf-chunk-bytes must be at least 1")

  results = run_text(args) if args.text else run_words(args)
  rendered = json.dumps(results, indent=2)
  print(rendered)
  if args.json:
    args.json.write_text(rendered + "\n", encoding="utf-8")
  return 0 if args.text or results["same_vocab"] else 1


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
