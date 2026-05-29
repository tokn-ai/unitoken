from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import inspect
from os import PathLike
from pathlib import Path
from typing import Literal, NoReturn

from .encoder import BpeEncoder, IdxArray

AllowedSpecial = Literal["all"] | set[str]
DisallowedSpecial = Literal["all"] | Collection[str]


def raise_disallowed_special_token(token: str) -> NoReturn:
  raise ValueError(
    f"Encountered text corresponding to disallowed special token {token!r}. "
    "Pass allowed_special to allow it, or disallowed_special=() to disable this check."
  )


def _byte_encoder() -> dict[int, str]:
  chars = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
  result = chars[:]
  n = 0
  for b in range(2**8):
    if b not in chars:
      chars.append(b)
      result.append(2**8 + n)
      n += 1
  return dict(zip(chars, (chr(n) for n in result), strict=True))


_BYTE_DECODER = {v: k for k, v in _byte_encoder().items()}


def _decode_gpt2_token(token: str) -> bytes:
  return bytes(_BYTE_DECODER[ch] for ch in token)


def _load_gpt2_vocab(path: str | PathLike) -> dict[bytes, int]:
  import json

  with open(path, "r", encoding="utf-8") as f:
    raw = json.load(f)
  return {_decode_gpt2_token(token): int(idx) for token, idx in raw.items()}


def _load_merges(path: str | PathLike, ranks: Mapping[bytes, int]) -> list[tuple[bytes, bytes]]:
  merges: list[tuple[bytes, bytes]] = []
  with open(path, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      if " => " in line:
        line = line.rsplit(" => ", 1)[0].strip()
      parts = line.split()
      if len(parts) != 2:
        continue
      left = _decode_gpt2_token(parts[0])
      right = _decode_gpt2_token(parts[1])
      if left in ranks and right in ranks:
        merges.append((left, right))
  return merges


def _infer_merges_from_ranks(ranks: Mapping[bytes, int]) -> list[tuple[bytes, bytes]]:
  known = set(ranks)
  merges: list[tuple[bytes, bytes]] = []
  for token, rank in sorted(ranks.items(), key=lambda item: item[1]):
    if len(token) <= 1:
      continue
    candidates: list[tuple[int, int, bytes, bytes]] = []
    for i in range(1, len(token)):
      left = token[:i]
      right = token[i:]
      left_rank = ranks.get(left)
      right_rank = ranks.get(right)
      if left_rank is None or right_rank is None:
        continue
      if left_rank < rank and right_rank < rank:
        candidates.append((max(left_rank, right_rank), min(left_rank, right_rank), left, right))
    if candidates:
      _, _, left, right = min(candidates)
      merges.append((left, right))
      known.add(token)
  return merges


class Encoding:
  """tiktoken-shaped encoding backed by unitoken's Rust BPE encoder."""

  def __init__(
      self,
      name: str,
      *,
      pat_str: str | None = None,
      mergeable_ranks: Mapping[bytes, int] | None = None,
      special_tokens: Mapping[str, int] | None = None,
      explicit_n_vocab: int | None = None,
      ch: Literal["u8", "char"] = "u8",
      output_format: Literal["gpt2", "uni"] | None = None,
      _encoder: BpeEncoder | None = None,
      _ordinary_encoder: BpeEncoder | None = None,
      _token_bytes: Mapping[int, bytes] | None = None,
      _merges: Sequence[tuple[bytes, bytes]] | None = None,
  ) -> None:
    self.name = name
    self._pat_str = pat_str
    self._special_tokens = dict(special_tokens or {})
    self._special_tokens_set = set(self._special_tokens)
    self._mergeable_ranks = dict(mergeable_ranks or {})
    self._explicit_n_vocab = explicit_n_vocab
    self._ch = ch
    self._output_format = output_format or ("uni" if ch == "char" else "gpt2")

    token_bytes = dict(_token_bytes or {})
    for token, idx in self._mergeable_ranks.items():
      token_bytes[idx] = token
    for token, idx in self._special_tokens.items():
      token_bytes[idx] = token.encode("utf-8")
    self._token_bytes = token_bytes

    if _encoder is not None:
      self._encoder = _encoder
    else:
      vocabs = {token: idx for idx, token in self._token_bytes.items()}
      merges = list(_merges) if _merges is not None else _infer_merges_from_ranks(self._mergeable_ranks)
      self._encoder = BpeEncoder(
        ch=ch,
        special_tokens=list(self._special_tokens),
        merges=merges,
        vocabs=vocabs,
        pat_str=pat_str,
      )
    if _ordinary_encoder is not None:
      self._ordinary_encoder = _ordinary_encoder
    else:
      vocabs = {token: idx for idx, token in self._token_bytes.items()}
      merges = list(_merges) if _merges is not None else _infer_merges_from_ranks(self._mergeable_ranks)
      self._ordinary_encoder = BpeEncoder(
        ch=ch,
        special_tokens=[],
        merges=merges,
        vocabs=vocabs,
        pat_str=pat_str,
      )

    if explicit_n_vocab is not None and self.n_vocab != explicit_n_vocab:
      raise ValueError(f"explicit_n_vocab={explicit_n_vocab} does not match n_vocab={self.n_vocab}")

  @classmethod
  def from_files(
      cls,
      name: str,
      *,
      vocab_file: str | PathLike,
      merges_file: str | PathLike,
      special_tokens: Mapping[str, int] | None = None,
      ch: Literal["u8", "char"] = "u8",
      output_format: Literal["gpt2", "uni"] | None = None,
      pat_str: str | None = None,
  ) -> "Encoding":
    spec = output_format or ("uni" if ch == "char" else "gpt2")
    if spec != "gpt2":
      encoder = BpeEncoder.load(
        ch=ch,
        output_format=spec,
        special_tokens=list(special_tokens or {}),
        merges_file=merges_file,
        vocabs_file=vocab_file,
        pat_str=pat_str,
      )
      ordinary_encoder = BpeEncoder.load(
        ch=ch,
        output_format=spec,
        special_tokens=[],
        merges_file=merges_file,
        vocabs_file=vocab_file,
        pat_str=pat_str,
      )
      return cls(name, special_tokens=special_tokens, ch=ch, output_format=spec, _encoder=encoder, _ordinary_encoder=ordinary_encoder)

    ranks = _load_gpt2_vocab(vocab_file)
    token_bytes = {idx: token for token, idx in ranks.items()}
    merges = _load_merges(merges_file, ranks)
    if special_tokens:
      ranks = {token: idx for token, idx in ranks.items() if token.decode("utf-8", "ignore") not in special_tokens}
    encoder = BpeEncoder.load(
      ch=ch,
      output_format=spec,
      special_tokens=list(special_tokens or {}),
      merges_file=merges_file,
      vocabs_file=vocab_file,
      pat_str=pat_str,
    )
    ordinary_encoder = BpeEncoder.load(
      ch=ch,
      output_format=spec,
      special_tokens=[],
      merges_file=merges_file,
      vocabs_file=vocab_file,
      pat_str=pat_str,
    )
    return cls(
      name,
      mergeable_ranks=ranks,
      special_tokens=special_tokens,
      ch=ch,
      output_format=spec,
      _encoder=encoder,
      _ordinary_encoder=ordinary_encoder,
      _token_bytes=token_bytes,
      _merges=merges,
    )

  @property
  def special_tokens_set(self) -> set[str]:
    return set(self._special_tokens_set)

  @property
  def eot_token(self) -> int:
    return self._special_tokens["<|endoftext|>"]

  @property
  def n_vocab(self) -> int:
    if self._token_bytes:
      return max(self._token_bytes) + 1
    return 0

  @property
  def max_token_value(self) -> int:
    if not self._token_bytes:
      return -1
    return max(self._token_bytes)

  def _collect_disallowed(self, allowed_special: AllowedSpecial, disallowed_special: DisallowedSpecial) -> set[str]:
    allowed = self._special_tokens_set if allowed_special == "all" else set(allowed_special)
    if disallowed_special == "all":
      return self._special_tokens_set - allowed
    return set(disallowed_special) - allowed

  def _raise_if_disallowed(self, text: str, allowed_special: AllowedSpecial, disallowed_special: DisallowedSpecial) -> None:
    for token in self._collect_disallowed(allowed_special, disallowed_special):
      if token in text:
        raise_disallowed_special_token(token)

  def _encode_impl(self, text: str, allowed_special: AllowedSpecial) -> list[int]:
    if allowed_special == "all":
      return self._encoder.encode_string(text).tolist()
    if not set(allowed_special):
      return self._ordinary_encoder.encode_string(text).tolist()
    return self._encoder.encode_string(text).tolist()

  def encode(
      self,
      text: str,
      *,
      allowed_special: AllowedSpecial = set(),
      disallowed_special: DisallowedSpecial = "all",
  ) -> list[int]:
    self._raise_if_disallowed(text, allowed_special, disallowed_special)
    return self._encode_impl(text, allowed_special)

  def encode_ordinary(self, text: str) -> list[int]:
    return self._ordinary_encoder.encode_string(text).tolist()

  def encode_to_numpy(
      self,
      text: str,
      *,
      allowed_special: AllowedSpecial = set(),
      disallowed_special: DisallowedSpecial = "all",
  ) -> IdxArray:
    self._raise_if_disallowed(text, allowed_special, disallowed_special)
    if allowed_special == "all" or set(allowed_special):
      return self._encoder.encode_string(text)
    return self._ordinary_encoder.encode_string(text)

  def encode_single_token(self, text_or_bytes: str | bytes) -> int:
    token = text_or_bytes.encode("utf-8") if isinstance(text_or_bytes, str) else text_or_bytes
    for idx, candidate in self._token_bytes.items():
      if candidate == token:
        return idx
    raise KeyError(text_or_bytes)

  def encode_batch(
      self,
      text: Sequence[str],
      *,
      num_threads: int = 8,
      allowed_special: AllowedSpecial = set(),
      disallowed_special: DisallowedSpecial = "all",
  ) -> list[list[int]]:
    if num_threads <= 1:
      return [self.encode(s, allowed_special=allowed_special, disallowed_special=disallowed_special) for s in text]
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
      return list(executor.map(lambda s: self.encode(s, allowed_special=allowed_special, disallowed_special=disallowed_special), text))

  def encode_ordinary_batch(self, text: Sequence[str], *, num_threads: int = 8) -> list[list[int]]:
    if num_threads <= 1:
      return [self.encode_ordinary(s) for s in text]
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
      return list(executor.map(self.encode_ordinary, text))

  def encode_with_unstable(
      self,
      text: str,
      *,
      allowed_special: AllowedSpecial = set(),
      disallowed_special: DisallowedSpecial = "all",
  ) -> tuple[list[int], list[list[int]]]:
    return self.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special), []

  def decode(self, tokens: Sequence[int], errors: str = "replace") -> str:
    if errors == "replace":
      return self._encoder.decode(tokens)
    return self.decode_bytes(tokens).decode("utf-8", errors)

  def decode_bytes(self, tokens: Sequence[int]) -> bytes:
    try:
      return b"".join(self.decode_single_token_bytes(token) for token in tokens)
    except KeyError:
      return self._encoder.decode(tokens).encode("utf-8")

  def decode_single_token_bytes(self, token: int) -> bytes:
    return self._token_bytes[token]

  def decode_tokens_bytes(self, tokens: Sequence[int]) -> list[bytes]:
    return [self.decode_single_token_bytes(token) for token in tokens]

  def decode_batch(self, batch: Sequence[Sequence[int]], *, errors: str = "replace", num_threads: int = 8) -> list[str]:
    if num_threads <= 1:
      return [self.decode(tokens, errors=errors) for tokens in batch]
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
      return list(executor.map(lambda tokens: self.decode(tokens, errors=errors), batch))

  def decode_bytes_batch(self, batch: Sequence[Sequence[int]], *, num_threads: int = 8) -> list[bytes]:
    if num_threads <= 1:
      return [self.decode_bytes(tokens) for tokens in batch]
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
      return list(executor.map(self.decode_bytes, batch))

  def decode_with_offsets(self, tokens: Sequence[int]) -> tuple[str, list[int]]:
    offsets = []
    pieces = []
    char_count = 0
    for token in tokens:
      offsets.append(char_count)
      piece = self.decode_single_token_bytes(token).decode("utf-8", "replace")
      pieces.append(piece)
      char_count += len(piece)
    return "".join(pieces), offsets

  def token_byte_values(self) -> list[bytes]:
    return [self._token_bytes[idx] for idx in sorted(self._token_bytes)]

  def is_special_token(self, token: int) -> bool:
    return token in set(self._special_tokens.values())


Encoding.__signature__ = inspect.Signature([
  inspect.Parameter("name", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation="str"),
  inspect.Parameter("pat_str", inspect.Parameter.KEYWORD_ONLY, annotation="str"),
  inspect.Parameter("mergeable_ranks", inspect.Parameter.KEYWORD_ONLY, annotation="dict[bytes, int]"),
  inspect.Parameter("special_tokens", inspect.Parameter.KEYWORD_ONLY, annotation="dict[str, int]"),
  inspect.Parameter("explicit_n_vocab", inspect.Parameter.KEYWORD_ONLY, default=None, annotation="int | None"),
])


def _fixture_encoding(name: str) -> Encoding:
  root = Path.cwd()
  if not (root / "fixtures").exists():
    root = Path(__file__).resolve().parents[2]
  vocab_file = root / f"fixtures/vocab.{name}.json"
  merges_file = root / f"fixtures/merges.{name}.txt"
  if not vocab_file.exists() or not merges_file.exists():
    raise ValueError(f"Unknown encoding {name!r}. Use Encoding.from_files for local unitoken models.")
  return Encoding.from_files(
    name,
    vocab_file=vocab_file,
    merges_file=merges_file,
    special_tokens={"<|endoftext|>": 0},
  )


def get_encoding(encoding_name: str) -> Encoding:
  return _fixture_encoding(encoding_name)


def encoding_for_model(model_name: str) -> Encoding:
  return get_encoding(model_name)


def encoding_name_for_model(model_name: str) -> str:
  if model_name not in list_encoding_names():
    raise KeyError(
      f"Could not automatically map {model_name} to a tokeniser. "
      "Please use `tiktoken.get_encoding` to explicitly get the tokeniser you expect."
    )
  return model_name


def list_encoding_names() -> list[str]:
  root = Path.cwd()
  if not (root / "fixtures").exists():
    root = Path(__file__).resolve().parents[2]
  root = root / "fixtures"
  names = []
  for vocab_file in root.glob("vocab.*.json"):
    name = vocab_file.name.removeprefix("vocab.").removesuffix(".json")
    if (root / f"merges.{name}.txt").exists():
      names.append(name)
  return sorted(names)
