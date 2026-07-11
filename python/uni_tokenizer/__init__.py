from .trainer import BpeTrainer, FileFormat, Unit
from .encoder import BpeEncoder
from .pretokenizer import BigramCounter, BoundaryMode, PreTokenizer, Source, UnicodeBigramMixedBoundary, WordCounter
from .tiktoken_compat import (
  Encoding,
  encoding_for_model,
  encoding_name_for_model,
  get_encoding,
  list_encoding_names,
)

try:
  from importlib.metadata import version as _pkg_version
  __version__ = _pkg_version("uni-tokenizer")
except Exception:  # pragma: no cover
  __version__ = "0.0.0"

__all__ = [
  "BpeTrainer",
  "BigramCounter",
  "BpeEncoder",
  "BoundaryMode",
  "Encoding",
  "FileFormat",
  "PreTokenizer",
  "Source",
  "UnicodeBigramMixedBoundary",
  "Unit",
  "WordCounter",
  "encoding_for_model",
  "encoding_name_for_model",
  "get_encoding",
  "list_encoding_names",
  "__version__",
]
