import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers import models
from tokenizers import pre_tokenizers
from tokenizers import trainers

from uni_tokenizer import BpeTrainer


def _bytes_to_unicode() -> dict[int, str]:
  bs = (
    list(range(ord("!"), ord("~") + 1))
    + list(range(ord("¡"), ord("¬") + 1))
    + list(range(ord("®"), ord("ÿ") + 1))
  )
  cs = bs[:]
  n = 0
  for b in range(256):
    if b not in bs:
      bs.append(b)
      cs.append(256 + n)
      n += 1
  return dict(zip(bs, map(chr, cs)))


def _to_byte_level_token(token: bytes) -> str:
  byte_encoder = _bytes_to_unicode()
  return "".join(byte_encoder[b] for b in token)


def _train_hf_from_words(words: list[tuple[str, int]], vocab_size: int) -> dict[str, int]:
  tokenizer = Tokenizer(models.BPE())
  tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
  trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    special_tokens=["<|endoftext|>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
  )
  tokenizer.train_from_iterator(
    (word for word, freq in words for _ in range(freq)),
    trainer=trainer,
  )
  return tokenizer.get_vocab()


def _train_ours_from_words(
  words: list[tuple[str, int]],
  vocab_size: int,
  parallel_merge_min_occurs_in: int | None = None,
) -> dict[str, int]:
  trainer = BpeTrainer(
    ["<|endoftext|>"],
    unit="byte",
    initial_alphabet="byte_level",
    parallel_merge_min_occurs_in=parallel_merge_min_occurs_in,
  )
  trainer.add_words(words)
  trainer.train(vocab_size)
  return {
    _to_byte_level_token(token): rank
    for token, rank in trainer.vocab.items()
  }


def test_bpe_training_tie_break_matches_hugging_face_byte_level() -> None:
  """Equal-frequency pairs should resolve to the same first merge as HF BPE."""
  words = [("ab", 1), ("cd", 1)]

  ours_vocab = _train_ours_from_words(words, 258)
  hf_vocab = _train_hf_from_words(words, 258)

  assert ours_vocab["ab"] == hf_vocab["ab"] == 257
  assert "cd" not in ours_vocab
  assert "cd" not in hf_vocab


def test_bpe_training_forced_parallel_merge_matches_default() -> None:
  words = [("ababc", 5), ("ababcbabc", 30), ("abcbabcab", 200)]

  default_vocab = _train_ours_from_words(words, 259)
  forced_parallel_vocab = _train_ours_from_words(words, 259, parallel_merge_min_occurs_in=1)

  assert forced_parallel_vocab == default_vocab
  assert forced_parallel_vocab["ab"] == 257
  assert forced_parallel_vocab["abc"] == 258


def test_bpe_training_learned_tokens_match_hugging_face_on_5m_fixture() -> None:
  root = Path(__file__).resolve().parents[1]
  words = json.loads((root / "fixtures" / "_words.tinystories_sample_5M.json").read_text())
  words = list(words.items())

  ours_vocab = _train_ours_from_words(words, 2000)
  hf_vocab = _train_hf_from_words(words, 2000)

  assert ours_vocab == hf_vocab
  assert ours_vocab["he"] == hf_vocab["he"] == 257
  assert ours_vocab["Ġthe"] == hf_vocab["Ġthe"] == 263
