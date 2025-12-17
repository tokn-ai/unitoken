from .trainer import BpeTrainer
from .encoder import BpeEncoder
from ._lib import PreTokenizer, Vocabs

try:
  from importlib.metadata import version as _pkg_version
  __version__ = _pkg_version("unitoken")
except Exception:  # pragma: no cover
  __version__ = "0.0.0"

__all__ = [
  "BpeTrainer",
  "BpeEncoder",
  "PreTokenizer",
  "Vocabs",
  "__version__",
]
