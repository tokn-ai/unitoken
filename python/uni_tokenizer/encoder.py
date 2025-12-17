from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import cast, TYPE_CHECKING
from ._lib import BpeEncoderBase
import numpy as np

IdxArray = np.ndarray[tuple[int], np.dtype[np.uint32]]

if TYPE_CHECKING:
  from .trainer import CharLevel, OutputFormat

class BpeEncoder:
  """BPE encoder.

  This is a thin Python wrapper around the Rust implementation.

  Parameters
  ----------
  ch:
    Character level: `"u8"` (byte-level, GPT-2-style) or `"char"` (Unicode-aware, Uni spec).
  special_tokens:
    Optional list of special tokens. When provided, they are treated as indivisible tokens.
  merges / vocabs:
    In-memory merge rules and vocabulary. If omitted, use :meth:`load` to load from files.
  """
  def __init__(
      self,
      ch: "CharLevel" = "u8",
      *,
      special_tokens: Sequence[str] | None = None,
      merges: list[tuple[bytes, bytes]] | None = None,
      vocabs: dict[bytes, int] | None = None,
      _encoder: BpeEncoderBase | None = None,
  ) -> None:
    self.char_level = ch
    if _encoder is not None:
      self._encoder = _encoder
    else:
      spec = "uni" if ch == "char" else "gpt2"
      self._encoder = BpeEncoderBase(
        spec=spec,
        char_level=ch,
        merges_filename=None,
        vocab_filename=None,
        merges=merges,
        vocabs=cast(dict[Sequence[int], int], vocabs),
        special_tokens=special_tokens,
      )

  @classmethod
  def load(
    cls,
    name: str | None = None,
    *,
    ch: "CharLevel" = "u8",
    output_format: "OutputFormat | None" = None,
    special_tokens: Sequence[str] | None = None,
    input_dir: str | PathLike | None = None,
    merges_file: str | PathLike | None = None,
    vocabs_file: str | PathLike | None = None,
  ) -> "BpeEncoder":
    """Load an encoder from vocab/merge files.

    Parameters
    ----------
    name:
      Optional model name used to derive default filenames:
      `merges.{name}[{ch}].txt` and `vocab.{name}[{ch}].json`.
    ch:
      Character level (`"u8"` or `"char"`).
    output_format:
      Override the spec used to decode the files (`"gpt2"` or `"uni"`).
      If omitted, defaults to `"gpt2"` for `ch="u8"` and `"uni"` for `ch="char"`.
    special_tokens:
      Optional list of special tokens to configure the encoder.
    input_dir:
      Optional directory to resolve `merges_file`/`vocabs_file` relative to.
    merges_file / vocabs_file:
      Explicit filenames/paths for merges and vocab.
    """
    spec = output_format
    if spec is None:
      spec = "uni" if ch == "char" else "gpt2"
    if name is not None:
      if merges_file is None:
        merges_file = f"merges.{name}[{ch}].txt"
      if vocabs_file is None:
        vocabs_file = f"vocab.{name}[{ch}].json"
    if input_dir is not None:
      if merges_file is not None:
        merges_file = Path(input_dir) / merges_file
      if vocabs_file is not None:
        vocabs_file = Path(input_dir) / vocabs_file
    return cls(
      ch=ch,
      _encoder=BpeEncoderBase(
        spec=spec,
        char_level=ch,
        merges_filename=merges_file,
        vocab_filename=vocabs_file,
        merges=None,
        vocabs=None,
        special_tokens=special_tokens,
      ),
    )

  def encode_word(self, /, word: str) -> list[int]:
    """Encode a single word into token ids."""
    return self._encoder.encode_word(word)

  def encode_words(self, /, words: Sequence[str]) -> list[list[int]]:
    """Encode multiple words into token ids."""
    return self._encoder.encode_words(words)

  def encode_string(self, /, s: str) -> IdxArray:
    """Encode an arbitrary string into a NumPy array of token ids."""
    return self._encoder.encode_string(s)

  def encode_file(self, /, path: str | PathLike, num_chunks: int = 1024) -> IdxArray:
    """Encode a text file into a NumPy array of token ids.

    Parameters
    ----------
    path:
      Path to a UTF-8 text file.
    num_chunks:
      Number of chunks to split the file into (chunk boundaries are aligned on the end-of-text token).
    """
    return self._encoder.encode_file(path, num_chunks)

  def decode(self, /, idxs: Sequence[int] | IdxArray) -> str:
    """Decode token ids back into a UTF-8 string."""
    return self._encoder.decode(idxs)
