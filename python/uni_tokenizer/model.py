from os import PathLike
from pathlib import Path
from typing import cast

from ._lib import BpeModelBase
from .trainer import FileFormat, Unit, _resolve_format


class BpeModel:
  """An immutable BPE model produced by :meth:`BpeTrainer.validate_model`."""

  def __init__(self, model: BpeModelBase) -> None:
    self._model = model

  @property
  def unit(self) -> Unit:
    """Atomic BPE unit used by this model."""
    return cast(Unit, self._model.unit)

  @property
  def vocab(self) -> dict[bytes, int]:
    """Return a snapshot of the validated token-to-id vocabulary."""
    return dict(self._model.get_vocab().items())

  @property
  def last_merge_freq(self) -> int | None:
    """Frequency of the final pair merge, if the model contains one."""
    return self._model.last_merge_freq

  def save_vocab_json(
    self,
    path: str | PathLike,
    *,
    format: FileFormat | None = None,
  ) -> None:
    """Save the validated vocabulary to a JSON file."""
    self._model.save_vocab(path, _resolve_format(self.unit, format))

  def save_merges_txt(
    self,
    path: str | PathLike,
    *,
    format: FileFormat | None = None,
  ) -> None:
    """Save the validated merge list to a text file."""
    self._model.save_merges_txt(path, _resolve_format(self.unit, format))

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
    """Save the validated vocabulary and merge list to explicit paths."""
    resolved_format = _resolve_format(self.unit, format)
    self._model.save_vocab(vocab_path, resolved_format)
    self._model.save_merges_txt(merges_path, resolved_format)
