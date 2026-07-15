from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from uni_tokenizer import BpeTrainer

from common import SPECIAL_TOKENS
from common import load_words


def file_sha256_prefix(path: Path, length: int = 12) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()[:length]


def train(
  words: Sequence[tuple[str, int]],
  vocab_size: int,
  unit: str,
  *,
  bbpe_fallback: bool = False,
  primary_vocab_ratio: float = 0.9,
) -> BpeTrainer:
  trainer = BpeTrainer(
    SPECIAL_TOKENS,
    unit=unit,
    initial_alphabet="byte_level" if unit == "byte" else None,
  )
  trainer.add_words(words)
  if bbpe_fallback:
    trainer.train_with_bbpe_fallback(
      vocab_size,
      primary_vocab_ratio=primary_vocab_ratio,
    )
  else:
    trainer.train(vocab_size)
  return trainer


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Train and save a golden tokenizer model from a word-frequency inventory.")
  parser.add_argument("--words", type=Path, required=True, help="JSON word-frequency inventory.")
  parser.add_argument("--dataset-name", required=True, help="Dataset directory name under out/models/golden.")
  parser.add_argument("--vocab-size", type=int, required=True)
  parser.add_argument("--unit", choices=["byte", "unicode"], default="unicode", help="BPE unit used for training.")
  parser.add_argument("--format", choices=["gpt2", "unitoken"], default="unitoken")
  parser.add_argument(
    "--bbpe-fallback",
    action="store_true",
    help="Train a Unicode model with a terminal byte-BPE fallback phase.",
  )
  parser.add_argument(
    "--primary-vocab-ratio",
    type=float,
    help="Fraction of learned slots reserved for primary Unicode training (default: 0.9).",
  )
  parser.add_argument("--out-dir", type=Path, default=Path("out") / "models" / "golden")
  parser.add_argument("--max-occurrences", type=int, help="Truncate the weighted corpus for a smaller smoke model.")
  args = parser.parse_args(argv)

  if args.vocab_size < 1:
    parser.error("--vocab-size must be at least 1")
  if args.unit == "unicode" and args.format != "unitoken":
    parser.error("--format must be unitoken when --unit=unicode")
  if args.bbpe_fallback and args.unit != "unicode":
    parser.error("--bbpe-fallback requires --unit=unicode")
  if args.primary_vocab_ratio is not None and not args.bbpe_fallback:
    parser.error("--primary-vocab-ratio requires --bbpe-fallback")

  primary_vocab_ratio = (
    args.primary_vocab_ratio
    if args.primary_vocab_ratio is not None
    else 0.9
  )
  if not math.isfinite(primary_vocab_ratio) or not 0.0 <= primary_vocab_ratio <= 1.0:
    parser.error("--primary-vocab-ratio must be finite and between 0 and 1")

  words = load_words(args.words, args.max_occurrences)
  started = time.perf_counter()
  trainer = train(
    words,
    args.vocab_size,
    args.unit,
    bbpe_fallback=args.bbpe_fallback,
    primary_vocab_ratio=primary_vocab_ratio,
  )
  train_s = time.perf_counter() - started

  input_hash = file_sha256_prefix(args.words)
  model_dir = args.out_dir / f"{args.dataset_name}.{input_hash}"
  model_dir.mkdir(parents=True, exist_ok=True)
  suffix_parts = [f"vocab{args.vocab_size}", args.unit]
  if args.max_occurrences is not None:
    suffix_parts.append(f"occ{args.max_occurrences}")
  if args.bbpe_fallback:
    ratio_label = format(primary_vocab_ratio * 100.0, ".12g").replace(".", "p")
    suffix_parts.append(f"bbpe-r{ratio_label}")
  suffix_parts.append(args.format)
  suffix = ".".join(suffix_parts)
  vocab_path = model_dir / f"vocab.{suffix}.json"
  merges_path = model_dir / f"merges.{suffix}.txt"
  model = trainer.validate_model()
  model.save_files(vocab_path, merges_path, format=args.format)

  print(f"saved {vocab_path}")
  print(f"saved {merges_path}")
  print(f"train_s={train_s:.3f}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
