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


def _train_ours_from_words(words: list[tuple[str, int]], vocab_size: int) -> dict[str, int]:
  trainer = BpeTrainer(["<|endoftext|>"], ch="u8")
  trainer.add_words(words)
  trainer.train(vocab_size)
  return {
    _to_byte_level_token(token): rank
    for token, rank in dict(trainer.vocabs.items()).items()
  }


def test_bpe_training_tie_break_matches_hugging_face_byte_level() -> None:
  """Equal-frequency pairs should resolve to the same first merge as HF BPE."""
  words = [("ab", 1), ("cd", 1)]

  ours_vocab = _train_ours_from_words(words, 258)
  hf_vocab = _train_hf_from_words(words, 258)

  assert ours_vocab["ab"] == hf_vocab["ab"] == 257
  assert "cd" not in ours_vocab
  assert "cd" not in hf_vocab


def test_bpe_training_learned_tokens_match_hugging_face_on_5m_fixture() -> None:
  root = Path(__file__).resolve().parents[1]
  words = json.loads((root / "fixtures" / "_words.tinystories_sample_5M.json").read_text())
  words = list(words.items())

  ours_vocab = _train_ours_from_words(words, 2000)
  hf_vocab = _train_hf_from_words(words, 2000)

  ours_learned = {token: rank for token, rank in ours_vocab.items() if rank > 256}
  hf_learned = {token: rank for token, rank in hf_vocab.items() if rank > 256}

  assert set(ours_learned) == set(hf_learned)
  assert ours_learned["he"] == hf_learned["he"] == 257
  assert ours_learned["Ġthe"] == hf_learned["Ġthe"] == 263

  rank_diffs = {
    token: (ours_learned[token], hf_learned[token])
    for token in ours_learned
    if ours_learned[token] != hf_learned[token]
  }
  assert rank_diffs == {
    "ĠOne": (527, 528),
    "round": (528, 527),
    "Ġz": (1117, 1118),
    "ble": (1118, 1117),
    "ĠCan": (1163, 1164),
    "Hell": (1164, 1163),
    "Ġour": (1180, 1181),
    "eet": (1181, 1180),
    "Ġice": (1218, 1219),
    "que": (1219, 1218),
    "Ġowl": (1398, 1399),
    "red": (1399, 1398),
    "Ġve": (1649, 1650),
    "sc": (1650, 1649),
  }
