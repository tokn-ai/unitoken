from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Literal
from ._lib import BpeTrainer_Character_CharIdx, BpeTrainer_u8_Idx

CharLevel = Literal["char", "u8"]
OutputFormat = Literal["uni", "gpt2"]

class BpeTrainer:
  """Train a BPE model from a word-frequency inventory.

  This wraps the Rust trainer classes exposed via the extension module.

  Parameters
  ----------
  special_tokens:
    Sequence of special tokens. The first token is treated as the end-of-text marker.
  ch:
    Character level: `"u8"` (byte-level, GPT-2-style) or `"char"` (Unicode-aware, Uni spec).
  output_format:
    Output spec for serialization (`"gpt2"` or `"uni"`). For `ch="char"`, only `"uni"` is supported.
  """
  def __init__(self, special_tokens: Sequence[str], *, ch: CharLevel = "u8", output_format: OutputFormat | None = None) -> None:
    # super().__init__()
    self._ch: CharLevel = ch
    self.output_format: OutputFormat = "uni"
    if ch == "char":
      self._trainer = BpeTrainer_Character_CharIdx(special_tokens=special_tokens)
    else:
      self.output_format = output_format or "gpt2"
      self._trainer = BpeTrainer_u8_Idx(special_tokens=special_tokens)

  @property
  def vocab_size(self) -> int:
    """Current vocabulary size."""
    return self._trainer.vocab_size()

  @property
  def char_level(self) -> CharLevel:
    """Character level used by this trainer."""
    return self._ch

  @property
  def vocabs(self):
    """Vocabulary view.

    Returns a `Vocabs` object (from the extension module) supporting `len`, `get`, and `items`.
    """
    return self._trainer.get_vocabs()

  def add_words(self, words: Mapping[str, int] | Sequence[tuple[str, int]]) -> None:
    """Add training data.

    Accepts either a mapping `{word: freq}` or an explicit sequence of `(word, freq)` pairs.
    """
    if isinstance(words, Mapping):
      words = list(words.items())
    self._trainer.add_words(words)

  def init_training(self) -> None:
    """Initialize internal training state."""
    self._trainer.init_training()

  def train(self, vocab_size: int) -> None:
    """Train until the vocab reaches `vocab_size` entries.

    This is a convenience wrapper over repeated :meth:`step` calls.
    """
    self.init_training()
    for _ in range(self.vocab_size, vocab_size):
      if self.step() is None:
        return

  def step(self) -> int | None:
    """Perform one training step.

    Returns the updated vocabulary size, or `None` if no further merges can be made.
    """
    try:
      return self._trainer.step() or self._trainer.vocab_size()
    except:
      return None

  def save(self, name: str, *, outdir: str | PathLike = ".", output_format: OutputFormat | None = None) -> None:
    """Save `vocab.{name}[{ch}].json` and `merges.{name}[{ch}].txt` into `outdir`."""
    vocab_path = Path(outdir) / f"vocab.{name}[{self.char_level}].json"
    merges_path = Path(outdir) / f"merges.{name}[{self.char_level}].txt"
    spec = output_format or self.output_format
    self._trainer.save_vocab(vocab_path, spec)
    self._trainer.save_merges_txt(merges_path, spec)
