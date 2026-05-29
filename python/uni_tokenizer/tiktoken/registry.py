from __future__ import annotations

from collections.abc import Callable
import functools
import importlib
import pkgutil
from collections.abc import Sequence
import threading
from typing import Any

import uni_tokenizer.tiktoken as tiktoken
import tiktoken_ext

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
