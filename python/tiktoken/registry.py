from collections.abc import Callable
from typing import Any

from uni_tokenizer.tiktoken_compat import Encoding, get_encoding, list_encoding_names

ENCODINGS: dict[str, Encoding] = {}
ENCODING_CONSTRUCTORS: dict[str, Callable[[], dict[str, Any]]] | None = None

__all__ = [
  "ENCODINGS",
  "ENCODING_CONSTRUCTORS",
  "Encoding",
  "get_encoding",
  "list_encoding_names",
]
