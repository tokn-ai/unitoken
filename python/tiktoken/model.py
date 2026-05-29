from uni_tokenizer.tiktoken_compat import (
  Encoding,
  encoding_for_model,
  encoding_name_for_model,
  get_encoding,
)

MODEL_PREFIX_TO_ENCODING: dict[str, str] = {}
MODEL_TO_ENCODING: dict[str, str] = {}

__all__ = [
  "Encoding",
  "MODEL_PREFIX_TO_ENCODING",
  "MODEL_TO_ENCODING",
  "encoding_for_model",
  "encoding_name_for_model",
  "get_encoding",
]
