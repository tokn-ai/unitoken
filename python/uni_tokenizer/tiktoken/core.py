from __future__ import annotations

import functools
from collections.abc import Collection, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import AbstractSet, Literal, NoReturn, TYPE_CHECKING

from uni_tokenizer.tiktoken_compat import Encoding, raise_disallowed_special_token

__all__ = [
  "Encoding",
  "raise_disallowed_special_token",
]
