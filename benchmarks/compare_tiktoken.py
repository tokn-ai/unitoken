from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from uni_tokenizer import Encoding
from uni_tokenizer.tiktoken_compat import _load_gpt2_vocab


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = REPO_ROOT / "fixtures" / "tinystories_sample_5M.txt"
DEFAULT_PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"


def bench(label: str, fn: Callable[[], Any], repeats: int) -> dict[str, Any]:
  samples = []
  result = None
  for _ in range(repeats):
    started = time.perf_counter()
    result = fn()
    samples.append(time.perf_counter() - started)
  return {
    "label": label,
    "repeats": repeats,
    "min_s": min(samples),
    "median_s": statistics.median(samples),
    "mean_s": statistics.mean(samples),
    "tokens": len(result) if hasattr(result, "__len__") else None,
  }


def load_unitoken_encoding(name: str) -> Encoding:
  return Encoding.from_files(
    name,
    vocab_file=REPO_ROOT / "fixtures" / f"vocab.{name}.json",
    merges_file=REPO_ROOT / "fixtures" / f"merges.{name}.txt",
    special_tokens={"<|endoftext|>": 0},
    pat_str=DEFAULT_PAT,
  )


def load_upstream_tiktoken(name: str, *, fixture_encoding: str, use_registry: bool):
  try:
    module = importlib.import_module("tiktoken")
  except ImportError:
    return None, "upstream tiktoken is not installed"

  module_path = Path(getattr(module, "__file__", "") or "")
  if module_path.is_relative_to(REPO_ROOT / "python"):
    return None, f"imported unitoken's tiktoken shim at {module_path}, not upstream tiktoken"

  if use_registry:
    try:
      return module.get_encoding(name), None
    except Exception as exc:
      return None, f"upstream tiktoken could not load {name!r}: {exc}"

  ranks = _load_gpt2_vocab(REPO_ROOT / "fixtures" / f"vocab.{fixture_encoding}.json")
  try:
    return module.Encoding(
      f"unitoken-{fixture_encoding}",
      pat_str=DEFAULT_PAT,
      mergeable_ranks=ranks,
      special_tokens={"<|endoftext|>": 0},
    ), None
  except Exception as exc:
    return None, f"upstream tiktoken could not construct local fixture encoding {fixture_encoding!r}: {exc}"


def run(args: argparse.Namespace) -> dict[str, Any]:
  text = Path(args.input).read_text(encoding="utf-8")[:args.bytes]
  unitoken = load_unitoken_encoding(args.unitoken_encoding)
  upstream, upstream_error = load_upstream_tiktoken(
    args.tiktoken_encoding,
    fixture_encoding=args.unitoken_encoding,
    use_registry=args.use_upstream_registry,
  )

  results = {
    "input": str(args.input),
    "bytes": len(text.encode("utf-8")),
    "repeats": args.repeats,
    "unitoken_encoding": args.unitoken_encoding,
    "tiktoken_encoding": args.tiktoken_encoding,
    "benchmarks": [],
    "upstream_error": upstream_error,
  }

  unitoken_ids = unitoken.encode(text, allowed_special="all")
  results["benchmarks"].append(bench("unitoken.encode", lambda: unitoken.encode(text, allowed_special="all"), args.repeats))
  results["benchmarks"].append(bench("unitoken.decode", lambda: unitoken.decode(unitoken_ids), args.repeats))

  if upstream is not None:
    upstream_ids = upstream.encode(text, allowed_special="all")
    results["same_tokens"] = unitoken_ids == upstream_ids
    results["unitoken_tokens"] = len(unitoken_ids)
    results["tiktoken_tokens"] = len(upstream_ids)
    results["benchmarks"].append(bench("tiktoken.encode", lambda: upstream.encode(text, allowed_special="all"), args.repeats))
    results["benchmarks"].append(bench("tiktoken.decode", lambda: upstream.decode(upstream_ids), args.repeats))

  return results


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Compare unitoken's tiktoken-compatible API with upstream tiktoken.")
  parser.add_argument("--input", type=Path, default=DEFAULT_FIXTURE)
  parser.add_argument("--bytes", type=int, default=200_000)
  parser.add_argument("--repeats", type=int, default=5)
  parser.add_argument("--unitoken-encoding", default="tinystories_sample_5M")
  parser.add_argument("--tiktoken-encoding", default="gpt2")
  parser.add_argument("--use-upstream-registry", action="store_true", help="Use tiktoken.get_encoding instead of constructing upstream Encoding from local fixture ranks.")
  parser.add_argument("--json", type=Path)
  args = parser.parse_args(argv)

  results = run(args)
  rendered = json.dumps(results, indent=2)
  print(rendered)
  if args.json:
    args.json.write_text(rendered + "\n", encoding="utf-8")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
