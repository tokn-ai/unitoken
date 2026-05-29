import unittest
from pathlib import Path

import tiktoken
from uni_tokenizer import Encoding, list_encoding_names
from uni_tokenizer.tiktoken_compat import _load_gpt2_vocab


R50K_PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"


class TiktokenCompatTests(unittest.TestCase):
  def make_encoding(self) -> Encoding:
    ranks = {
      b"a": 0,
      b"b": 1,
      b"c": 2,
      b"ab": 3,
      b"abc": 4,
      b" ": 5,
      b"x": 6,
      b"<": 8,
      b"|": 9,
      b">": 10,
      b"e": 11,
      b"n": 12,
      b"d": 13,
      b"o": 14,
      b"f": 15,
      b"t": 16,
      b"h": 17,
      b"i": 18,
    }
    return Encoding(
      "toy",
      mergeable_ranks=ranks,
      special_tokens={"<|endoftext|>": 7},
    )

  def test_tiktoken_shim_exports_encoding(self) -> None:
    self.assertIn("tinystories_sample_5M", list_encoding_names())
    if getattr(tiktoken.Encoding, "__module__", "").startswith("uni_tokenizer"):
      self.assertIs(tiktoken.Encoding, Encoding)

  def test_encode_decode_round_trip(self) -> None:
    enc = self.make_encoding()

    tokens = enc.encode("abc ab")

    self.assertEqual(tokens, [4, 5, 3])
    self.assertEqual(enc.decode(tokens), "abc ab")
    self.assertEqual(enc.decode_bytes(tokens), b"abc ab")

  def test_single_token_helpers(self) -> None:
    enc = self.make_encoding()

    self.assertEqual(enc.encode_single_token(b"abc"), 4)
    self.assertEqual(enc.decode_single_token_bytes(4), b"abc")
    self.assertEqual(enc.decode_tokens_bytes([4, 5, 3]), [b"abc", b" ", b"ab"])

  def test_special_token_policy_matches_tiktoken_shape(self) -> None:
    enc = self.make_encoding()

    with self.assertRaises(ValueError):
      enc.encode("<|endoftext|>")

    self.assertEqual(enc.encode("<|endoftext|>", allowed_special="all"), [7])
    self.assertNotEqual(enc.encode("<|endoftext|>", disallowed_special=()), [7])
    self.assertEqual(enc.encode_ordinary("<|endoftext|>"), enc.encode("<|endoftext|>", disallowed_special=()))

  def test_batch_helpers(self) -> None:
    enc = self.make_encoding()

    batch = enc.encode_batch(["ab", "abc"], num_threads=1)

    self.assertEqual(batch, [[3], [4]])
    self.assertEqual(enc.decode_batch(batch, num_threads=1), ["ab", "abc"])
    self.assertEqual(enc.encode_ordinary_batch(["ab", "abc"], num_threads=1), [[3], [4]])
    self.assertEqual(enc.decode_bytes_batch(batch, num_threads=1), [b"ab", b"abc"])

  def test_extra_encoding_helpers(self) -> None:
    enc = self.make_encoding()

    self.assertEqual(enc.eot_token, 7)
    self.assertTrue(enc.is_special_token(7))
    self.assertFalse(enc.is_special_token(4))
    self.assertEqual(enc.encode_to_numpy("ab").tolist(), [3])
    self.assertEqual(enc.encode_with_unstable("ab"), ([3], []))
    self.assertEqual(enc.decode_with_offsets([4, 5, 3]), ("abc ab", [0, 3, 4]))
    self.assertIn(b"abc", enc.token_byte_values())

  def test_matches_upstream_tiktoken_on_toy_encoding(self) -> None:
    ranks = {
      b"a": 0,
      b"b": 1,
      b"c": 2,
      b"ab": 3,
      b"abc": 4,
      b" ": 5,
      b"x": 6,
    }
    ours = Encoding("toy", pat_str=R50K_PAT, mergeable_ranks=ranks, special_tokens={"<|endoftext|>": 7})
    theirs = tiktoken.Encoding("toy", pat_str=R50K_PAT, mergeable_ranks=ranks, special_tokens={"<|endoftext|>": 7})

    for text in ["ab", "abc ab", "x ab"]:
      self.assertEqual(ours.encode(text), theirs.encode(text))
      self.assertEqual(ours.encode_ordinary(text), theirs.encode_ordinary(text))

    self.assertEqual(ours.decode([4, 5, 3]), theirs.decode([4, 5, 3]))
    self.assertEqual(ours.decode_single_token_bytes(4), theirs.decode_single_token_bytes(4))

  def test_matches_upstream_tiktoken_on_fixture_prefix(self) -> None:
    root = Path(__file__).resolve().parents[1]
    name = "tinystories_sample_5M"
    text = (root / "fixtures" / f"{name}.txt").read_text(encoding="utf-8")[:50_000]
    ours = Encoding.from_files(
      name,
      vocab_file=root / "fixtures" / f"vocab.{name}.json",
      merges_file=root / "fixtures" / f"merges.{name}.txt",
      special_tokens={"<|endoftext|>": 0},
      pat_str=R50K_PAT,
    )
    theirs = tiktoken.Encoding(
      name,
      pat_str=R50K_PAT,
      mergeable_ranks=_load_gpt2_vocab(root / "fixtures" / f"vocab.{name}.json"),
      special_tokens={"<|endoftext|>": 0},
    )

    self.assertEqual(ours.encode(text, allowed_special="all"), theirs.encode(text, allowed_special="all"))


if __name__ == "__main__":
  unittest.main()
