from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from typing import Literal

from ._lib import PreTokenizer as _PreTokenizer


BoundaryMode = Literal["auto", "eot", "line", "utf8"]


class PreTokenizer:
  def __init__(
    self,
    special_tokens: Sequence[str],
    eot_token: str | None = None,
    pat: str | None = None,
  ) -> None:
    self._inner = _PreTokenizer(special_tokens, eot_token, pat)

  def find_chunk_boundaries(
    self,
    path: str | PathLike,
    *,
    chunk_size: int = 1024 * 1024,
    boundary: BoundaryMode = "auto",
  ) -> list[tuple[int, int]]:
    return self._inner.find_chunk_boundaries(path, chunk_size=chunk_size, boundary=boundary)

  def get_words_from_file(
    self,
    path: str | PathLike,
    *,
    chunk_size: int = 1024 * 1024,
    boundary: BoundaryMode = "auto",
  ) -> dict[str, int]:
    return self._inner.get_words_from_file(path, chunk_size=chunk_size, boundary=boundary)

  def get_words_from_segment(
    self,
    path: str | PathLike,
    offset: int,
    length: int,
  ) -> dict[str, int]:
    return self._inner.get_words_from_segment(path, offset, length)
