import unittest

import tiktoken
from uni_tokenizer import Encoding


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
    }
    return Encoding(
      "toy",
      mergeable_ranks=ranks,
      special_tokens={"<|endoftext|>": 7},
    )

  def test_tiktoken_shim_exports_encoding(self) -> None:
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


if __name__ == "__main__":
  unittest.main()
