from .trainer import BpeTrainer
from .encoder import BpeEncoder
from ._lib import Vocabs
from .pretokenizer import BoundaryMode, PreTokenizer, UnicodeBigramBoundary
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
  "BpeEncoder",
  "BoundaryMode",
  "Encoding",
  "PreTokenizer",
  "UnicodeBigramBoundary",
  "Vocabs",
  "encoding_for_model",
  "encoding_name_for_model",
  "get_encoding",
  "list_encoding_names",
  "__version__",
]
