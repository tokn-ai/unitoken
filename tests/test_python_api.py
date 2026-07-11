from pathlib import Path

import numpy as np
import pytest

from uni_tokenizer import BpeEncoder, BpeTrainer, PreTokenizer


def test_pretokenizer_uses_pat_str_and_returns_words() -> None:
  pretokenizer = PreTokenizer([], pat_str=r"[^\s]")

  assert pretokenizer.get_words("ab a") == {"a": 2, "b": 1}


def test_encoder_uses_unit_and_singular_vocab() -> None:
  encoder = BpeEncoder(
    unit="byte",
    vocab={b"a": 0, b"b": 1, b"ab": 2},
    merges=[(b"a", b"b")],
    pat_str=r"\S+",
  )

  assert encoder.unit == "byte"
  assert encoder.encode("ab") == [2]
  encoded = encoder.encode_to_numpy("ab")
  assert isinstance(encoded, np.ndarray)
  assert encoded.tolist() == [2]


def test_trainer_exposes_unit_and_singular_vocab() -> None:
  trainer = BpeTrainer(["<|endoftext|>"], unit="byte")

  assert trainer.unit == "byte"
  assert isinstance(trainer.vocab, dict)


def test_unknown_unit_is_rejected() -> None:
  with pytest.raises(ValueError, match="Unknown unit"):
    BpeTrainer([], unit="characters")  # type: ignore[arg-type]


def test_gpt2_format_rejects_unicode_unit_without_creating_files(tmp_path: Path) -> None:
  trainer = BpeTrainer([], unit="unicode")
  vocab_file = tmp_path / "vocab.json"
  merges_file = tmp_path / "merges.txt"

  with pytest.raises(ValueError, match="not compatible"):
    trainer.save_files(vocab_file, merges_file, format="gpt2")

  assert not vocab_file.exists()
  assert not merges_file.exists()
