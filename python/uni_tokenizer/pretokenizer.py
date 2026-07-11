from __future__ import annotations

from collections.abc import Iterator, Sequence
from os import PathLike
from typing import Literal, Protocol

from ._lib import BigramCounter, PreTokenizer as _PreTokenizer, WordCounter


BoundaryMode = Literal["auto", "eot", "line", "utf8"]
UnicodeBigramMixedBoundary = Literal["keep", "split"]

class Source(Protocol):
  def scan(self) -> Iterator[str]:
    """Return a new complete scan of independent text records."""
    ...


class PreTokenizer:
  def __init__(
    self,
    special_tokens: Sequence[str],
    eot_token: str | None = None,
    pat_str: str | None = None,
    unicode_bigrams: Sequence[str] | None = None,
    unicode_bigram_mixed_boundary: UnicodeBigramMixedBoundary = "keep",
  ) -> None:
    self._special_tokens = list(special_tokens)
    self._eot_token = eot_token
    self._pat_str = pat_str
    self._unicode_bigrams = list(unicode_bigrams) if unicode_bigrams is not None else None
    self._unicode_bigram_mixed_boundary = unicode_bigram_mixed_boundary
    self._inner = _PreTokenizer(special_tokens, eot_token, pat_str, unicode_bigrams, unicode_bigram_mixed_boundary)

  def with_unicode_bigrams(self, bigrams: Sequence[str]) -> "PreTokenizer":
    """Return a pretokenizer using a frozen Unicode bigram set."""
    return PreTokenizer(
      self._special_tokens,
      self._eot_token,
      self._pat_str,
      bigrams,
      self._unicode_bigram_mixed_boundary,
    )

  def bigram_counter(self) -> BigramCounter:
    """Create an empty mergeable Unicode bigram counter."""
    return self._inner.bigram_counter()

  def word_counter(self) -> WordCounter:
    """Create an empty mergeable word counter."""
    return self._inner.word_counter()

  def get_words(self, text: str) -> dict[str, int]:
    """Pretokenize text and return word frequencies."""
    return self._inner.get_words(text)

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

  def build_unicode_bigrams_from_file(
    self,
    path: str | PathLike,
    *,
    chunk_size: int = 1024 * 1024,
    boundary: BoundaryMode = "auto",
    top_k: int = 100_000,
    min_freq: int = 16,
  ) -> list[str]:
    return self._inner.build_unicode_bigrams_from_file(
      path,
      chunk_size=chunk_size,
      boundary=boundary,
      top_k=top_k,
      min_freq=min_freq,
    )
