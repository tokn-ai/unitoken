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
  def __init__(
      self,
      ch: "CharLevel" = "u8",
      *,
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
      )

  @classmethod
  def load(
    cls,
    name: str | None = None,
    *,
    ch: "CharLevel" = "u8",
    output_format: "OutputFormat | None" = None,
    input_dir: str | PathLike | None = None,
    merges_file: str | PathLike | None = None,
    vocabs_file: str | PathLike | None = None,
  ) -> "BpeEncoder":
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
      ),
    )

  def encode_word(self, /, word: str) -> list[int]:
    return self._encoder.encode_word(word)

  def encode_words(self, /, words: Sequence[str]) -> list[list[int]]:
    return self._encoder.encode_words(words)

  def encode_string(self, /, s: str) -> IdxArray:
    return self._encoder.encode_string(s)

  def encode_file(self, /, path: str | PathLike, num_chunks: int = 1024) -> IdxArray:
    return self._encoder.encode_file(path, num_chunks)
