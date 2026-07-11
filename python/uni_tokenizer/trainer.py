from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Literal
from ._lib import BpeTrainer_Character_CharIdx, BpeTrainer_u8_Idx, WordCounter

Unit = Literal["byte", "unicode"]
FileFormat = Literal["unitoken", "gpt2"]
InitialAlphabet = Literal["raw", "byte_level"]
TieBreak = Literal["smallest_pair_id", "largest_content"]

def _validate_unit(unit: str) -> None:
  if unit not in ("byte", "unicode"):
    raise ValueError(f"Unknown unit: {unit}")

def _resolve_format(unit: Unit, format: FileFormat | None) -> FileFormat:
  _validate_unit(unit)
  resolved_format = format or ("unitoken" if unit == "unicode" else "gpt2")
  if resolved_format not in ("gpt2", "unitoken"):
    raise ValueError(f"Unknown format: {resolved_format}")
  if unit == "unicode" and resolved_format == "gpt2":
    raise ValueError('format="gpt2" is not compatible with unit="unicode"')
  return resolved_format

class BpeTrainer:
  """Train a BPE model from a word-frequency inventory.

  This wraps the Rust trainer classes exposed via the extension module.

  Parameters
  ----------
  special_tokens:
    Sequence of tokens reserved in the vocabulary.
  unit:
    Atomic BPE unit: `"byte"` or `"unicode"`.
  """
  def __init__(
    self,
    special_tokens: Sequence[str],
    *,
    unit: Unit = "byte",
    initial_alphabet: InitialAlphabet | None = None,
    tie_break: TieBreak | None = None,
    parallel_merge_min_occurs_in: int | None = None,
  ) -> None:
    _validate_unit(unit)
    self._unit = unit
    if unit == "unicode":
      self._trainer = BpeTrainer_Character_CharIdx(
        special_tokens=special_tokens,
        initial_alphabet=initial_alphabet,
        tie_break=tie_break,
        parallel_merge_min_occurs_in=parallel_merge_min_occurs_in,
      )
    elif unit == "byte":
      self._trainer = BpeTrainer_u8_Idx(
        special_tokens=special_tokens,
        initial_alphabet=initial_alphabet,
        tie_break=tie_break,
        parallel_merge_min_occurs_in=parallel_merge_min_occurs_in,
      )

  @property
  def vocab_size(self) -> int:
    """Current vocabulary size."""
    return self._trainer.vocab_size()

  @property
  def unit(self) -> Unit:
    """Atomic BPE unit used by this trainer."""
    return self._unit

  @property
  def vocab(self) -> dict[bytes, int]:
    """Return a snapshot of the current token-to-id vocabulary."""
    return dict(self._trainer.get_vocab().items())

  def add_words(self, words: Mapping[str, int] | Sequence[tuple[str, int]]) -> None:
    """Add training data.

    Accepts either a mapping `{word: freq}` or an explicit sequence of `(word, freq)` pairs.
    """
    if isinstance(words, Mapping):
      words = list(words.items())
    self._trainer.add_words(words)

  def add_word_counter(self, counter: WordCounter) -> None:
    """Replace the training inventory by consuming an exact native word counter.

    The counter is empty and reusable after this call. Unlike `words()`, this
    transfer does not construct a Python dictionary.
    """
    self._trainer.add_word_counter(counter)

  def init_training(self) -> None:
    """Initialize internal training state."""
    self._trainer.init_training()

  def train(self, vocab_size: int) -> None:
    """Train until the vocab reaches `vocab_size` entries.

    Training runs inside Rust until the target is reached.
    """
    self._trainer.train_until(vocab_size)

  def step(self) -> int:
    """Perform one training step.

    Returns the updated vocabulary size.
    """
    return self._trainer.step()

  def validate_model(self) -> None:
    """Validate vocabulary uniqueness and merge dependency order."""
    self._trainer.validate_model()

  def save(self, name: str, *, outdir: str | PathLike = ".", format: FileFormat | None = None) -> None:
    """Save `vocab.{name}[{unit}].json` and `merges.{name}[{unit}].txt` into `outdir`."""
    vocab_path = Path(outdir) / f"vocab.{name}[{self.unit}].json"
    merges_path = Path(outdir) / f"merges.{name}[{self.unit}].txt"
    self.save_files(vocab_path, merges_path, format=format)

  def save_files(
    self,
    vocab_path: str | PathLike,
    merges_path: str | PathLike,
    *,
    format: FileFormat | None = None,
  ) -> None:
    """Save vocab and merges to explicit paths."""
    resolved_format = _resolve_format(self.unit, format)
    self._trainer.save_vocab(vocab_path, resolved_format)
    self._trainer.save_merges_txt(merges_path, resolved_format)
