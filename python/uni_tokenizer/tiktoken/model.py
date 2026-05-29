from __future__ import annotations

from uni_tokenizer.tiktoken_compat import Encoding, list_encoding_names as _list_encoding_names

from .registry import get_encoding

MODEL_PREFIX_TO_ENCODING: dict[str, str] = {}
MODEL_TO_ENCODING: dict[str, str] = {}


def encoding_name_for_model(model_name: str) -> str:
  if model_name in MODEL_TO_ENCODING:
    return MODEL_TO_ENCODING[model_name]
  for prefix, encoding_name in MODEL_PREFIX_TO_ENCODING.items():
    if model_name.startswith(prefix):
      return encoding_name
  if model_name in _list_encoding_names():
    return model_name
  raise KeyError(
    f"Could not automatically map {model_name} to a tokeniser. "
    "Please use `tiktoken.get_encoding` to explicitly get the tokeniser you expect."
  )


def encoding_for_model(model_name: str) -> Encoding:
  return get_encoding(encoding_name_for_model(model_name))

__all__ = [
  "Encoding",
  "MODEL_PREFIX_TO_ENCODING",
  "MODEL_TO_ENCODING",
  "encoding_for_model",
  "encoding_name_for_model",
  "get_encoding",
]
