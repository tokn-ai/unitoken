from collections.abc import Callable
from typing import Any

from uni_tokenizer.tiktoken_compat import Encoding, get_encoding as _get_encoding, list_encoding_names

ENCODINGS: dict[str, Encoding] = {}
ENCODING_CONSTRUCTORS: dict[str, Callable[[], dict[str, Any]]] | None = None


def get_encoding(encoding_name: str) -> Encoding:
  if encoding_name not in ENCODINGS:
    ENCODINGS[encoding_name] = _get_encoding(encoding_name)
  return ENCODINGS[encoding_name]

__all__ = [
  "ENCODINGS",
  "ENCODING_CONSTRUCTORS",
  "Encoding",
  "get_encoding",
  "list_encoding_names",
]
