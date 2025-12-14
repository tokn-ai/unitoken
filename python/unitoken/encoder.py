from ast import TypeVar
from collections.abc import Sequence
from os import PathLike
from typing import cast, TYPE_CHECKING
from ._lib import BpeEncoderBase
import numpy as np

IdxArray = np.ndarray[tuple[int], np.dtype[np.uint32]]

if TYPE_CHECKING:
  from .trainer import CharLevel, OutputFormat

class BpeEncoder:
  def __init__(
      self,
      char_level: "CharLevel" = "u8",
      output_format: "OutputFormat | None" = None,
      *,
      name: str | None = None,
      merges_file: str | PathLike | None = None,
      vocabs_file: str | PathLike | None = None,
      merges: list[tuple[bytes, bytes]] | None = None,
      vocabs: dict[bytes, int] | None = None,
  ) -> None:
    self.char_level = char_level
    spec = output_format
    if spec is None:
      spec = "uni" if char_level == "char" else "gpt2"
    self.output_format = spec
    if name is not None:
      if merges is None:
        merges_file = f"merges.{name}[{char_level}].txt"
      if vocabs is None:
        vocabs_file = f"vocab.{name}[{char_level}].json"
    self._encoder = BpeEncoderBase(
      spec=spec,
      char_level=char_level,
      merges_filename=merges_file,
      vocab_filename=vocabs_file,
      merges=merges,
      vocabs=cast(dict[Sequence[int], int], vocabs),
    )

  def encode_word(self, /, word: str) -> list[int]:
    return self._encoder.encode_word(word)

  def encode_words(self, /, words: Sequence[str]) -> list[list[int]]:
    return self._encoder.encode_words(words)

  def encode_string(self, /, s: str) -> IdxArray:
    return self._encoder.encode_string(s)

  def encode_file(self, /, path: str | PathLike, num_chunks: int) -> IdxArray:
    return self._encoder.encode_file(path, num_chunks)
