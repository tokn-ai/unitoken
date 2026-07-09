from __future__ import annotations

import argparse
import hashlib
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


def train(words: Sequence[tuple[str, int]], vocab_size: int, ch: str) -> BpeTrainer:
  trainer = BpeTrainer(
    SPECIAL_TOKENS,
    ch=ch,
    initial_alphabet="byte_level" if ch == "u8" else None,
  )
  trainer.add_words(words)
  trainer.train(vocab_size)
  return trainer


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Train and save a golden tokenizer model from a word-frequency inventory.")
  parser.add_argument("--words", type=Path, required=True, help="JSON word-frequency inventory.")
  parser.add_argument("--dataset-name", required=True, help="Dataset directory name under out/models/golden.")
  parser.add_argument("--vocab-size", type=int, required=True)
  parser.add_argument("--ch", choices=["u8", "char"], default="char", help="Trainer character level. Use char for Uni models.")
  parser.add_argument("--output-format", choices=["gpt2", "uni"], default="uni")
  parser.add_argument("--out-dir", type=Path, default=Path("out") / "models" / "golden")
  parser.add_argument("--max-occurrences", type=int, help="Truncate the weighted corpus for a smaller smoke model.")
  args = parser.parse_args(argv)

  if args.vocab_size < 1:
    parser.error("--vocab-size must be at least 1")
  if args.ch == "char" and args.output_format != "uni":
    parser.error("--output-format must be uni when --ch=char")

  words = load_words(args.words, args.max_occurrences)
  started = time.perf_counter()
  trainer = train(words, args.vocab_size, args.ch)
  train_s = time.perf_counter() - started

  input_hash = file_sha256_prefix(args.words)
  model_dir = args.out_dir / f"{args.dataset_name}.{input_hash}"
  model_dir.mkdir(parents=True, exist_ok=True)
  suffix = f"vocab{args.vocab_size}.{args.output_format}"
  vocab_path = model_dir / f"vocab.{suffix}.json"
  merges_path = model_dir / f"merges.{suffix}.txt"
  trainer.save_files(vocab_path, merges_path, output_format=args.output_format)

  print(f"saved {vocab_path}")
  print(f"saved {merges_path}")
  print(f"train_s={train_s:.3f}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
