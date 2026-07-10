from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import cast
from ._lib import BpeEncoderBase
from .trainer import FileFormat, Unit, _resolve_format, _validate_unit
import numpy as np

IdxArray = np.ndarray[tuple[int], np.dtype[np.uint32]]

class BpeEncoder:
  """BPE encoder.

  This is a thin Python wrapper around the Rust implementation.

  Parameters
  ----------
  unit:
    Atomic BPE unit: `"byte"` or `"unicode"`.
  special_tokens:
    Optional list of special tokens. When provided, they are treated as indivisible tokens.
  merges / vocab:
    In-memory merge rules and vocabulary. If omitted, use :meth:`load` to load from files.
  """
  def __init__(
      self,
      unit: "Unit" = "byte",
      *,
      special_tokens: Sequence[str] | None = None,
      merges: list[tuple[bytes, bytes]] | None = None,
      vocab: dict[bytes, int] | None = None,
      pat_str: str | None = None,
  ) -> None:
    _validate_unit(unit)
    self.unit = unit
    file_format = _resolve_format(unit, None)
    self._encoder = BpeEncoderBase(
      format=file_format,
      unit=unit,
      merges_file=None,
      vocab_file=None,
      merges=merges,
      vocab=cast(dict[Sequence[int], int], vocab),
      special_tokens=special_tokens,
      pat_str=pat_str,
    )

  @classmethod
  def _from_encoder(cls, unit: "Unit", encoder: BpeEncoderBase) -> "BpeEncoder":
    instance = cls.__new__(cls)
    instance.unit = unit
    instance._encoder = encoder
    return instance

  @classmethod
  def load(
    cls,
    name: str | None = None,
    *,
    unit: "Unit" = "byte",
    format: "FileFormat | None" = None,
    special_tokens: Sequence[str] | None = None,
    input_dir: str | PathLike | None = None,
    merges_file: str | PathLike | None = None,
    vocab_file: str | PathLike | None = None,
    pat_str: str | None = None,
  ) -> "BpeEncoder":
    """Load an encoder from vocab/merge files.

    Parameters
    ----------
    name:
      Optional model name used to derive default filenames:
      `merges.{name}[{unit}].txt` and `vocab.{name}[{unit}].json`.
    unit:
      Atomic BPE unit (`"byte"` or `"unicode"`).
    format:
      Override the format used to decode the files (`"gpt2"` or `"unitoken"`).
      If omitted, defaults to `"gpt2"` for byte units and `"unitoken"` for Unicode units.
    special_tokens:
      Optional list of special tokens to configure the encoder.
    input_dir:
      Optional directory to resolve `merges_file`/`vocab_file` relative to.
    merges_file / vocab_file:
      Explicit filenames/paths for merges and vocab.
    """
    resolved_format = _resolve_format(unit, format)
    if name is not None:
      if merges_file is None:
        merges_file = f"merges.{name}[{unit}].txt"
      if vocab_file is None:
        vocab_file = f"vocab.{name}[{unit}].json"
    if input_dir is not None:
      if merges_file is not None:
        merges_file = Path(input_dir) / merges_file
      if vocab_file is not None:
        vocab_file = Path(input_dir) / vocab_file
    return cls._from_encoder(
      unit,
      BpeEncoderBase(
        format=resolved_format,
        unit=unit,
        merges_file=merges_file,
        vocab_file=vocab_file,
        merges=None,
        vocab=None,
        special_tokens=special_tokens,
        pat_str=pat_str,
      ),
    )

  def encode_word(self, /, word: str) -> list[int]:
    """Encode a single word into token ids."""
    return self._encoder.encode_word(word)

  def encode_words(self, /, words: Sequence[str]) -> list[list[int]]:
    """Encode multiple words into token ids."""
    return self._encoder.encode_words(words)

  def encode(self, /, text: str) -> list[int]:
    """Encode text into a Python list of token ids."""
    return self._encoder.encode(text)

  def encode_to_numpy(self, /, text: str) -> IdxArray:
    """Encode text into a NumPy array of token ids."""
    return self._encoder.encode_to_numpy(text)

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
