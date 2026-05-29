from __future__ import annotations

import base64
import hashlib
import json
import urllib.request
from pathlib import Path


def check_hash(data: bytes, expected_hash: str) -> bool:
  return hashlib.sha256(data).hexdigest() == expected_hash


def read_file(blobpath: str) -> bytes:
  if blobpath.startswith(("http://", "https://")):
    with urllib.request.urlopen(blobpath) as response:
      return response.read()
  return Path(blobpath).read_bytes()


def read_file_cached(blobpath: str, expected_hash: str | None = None) -> bytes:
  data = read_file(blobpath)
  if expected_hash is not None and not check_hash(data, expected_hash):
    raise ValueError(f"Hash mismatch for data downloaded from {blobpath}")
  return data


def load_tiktoken_bpe(tiktoken_bpe_file: str, expected_hash: str | None = None) -> dict[bytes, int]:
  contents = read_file_cached(tiktoken_bpe_file, expected_hash)
  ranks: dict[bytes, int] = {}
  for line in contents.splitlines():
    if not line:
      continue
    token, rank = line.split()
    ranks[base64.b64decode(token)] = int(rank)
  return ranks


def dump_tiktoken_bpe(bpe_ranks: dict[bytes, int], tiktoken_bpe_file: str) -> None:
  lines = []
  for token, rank in sorted(bpe_ranks.items(), key=lambda item: item[1]):
    encoded = base64.b64encode(token).decode("ascii")
    lines.append(f"{encoded} {rank}")
  Path(tiktoken_bpe_file).write_text("\n".join(lines) + "\n", encoding="utf-8")


def data_gym_to_mergeable_bpe_ranks(
    vocab_bpe_file: str,
    encoder_json_file: str,
    vocab_bpe_hash: str | None = None,
    encoder_json_hash: str | None = None,
    clobber_one_byte_tokens: bool = False,
) -> dict[bytes, int]:
  del vocab_bpe_file, vocab_bpe_hash, clobber_one_byte_tokens
  encoder = json.loads(read_file_cached(encoder_json_file, encoder_json_hash))
  return {token.encode("utf-8"): int(rank) for token, rank in encoder.items()}
