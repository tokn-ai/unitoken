import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from common import (  # noqa: E402
  bigram_frequency_guard,
  load_word_inventory_manifest,
  word_inventory_manifest_path,
  write_word_inventory_manifest,
)


def test_word_inventory_manifest_round_trip_and_frequency_guard(tmp_path: Path) -> None:
  words_path = tmp_path / "_words.json"
  words_path.write_text(json.dumps({"你好": 7}), encoding="utf-8")

  manifest_path = write_word_inventory_manifest(
    words_path,
    source={"input_kind": "memory", "name": "fixture"},
    pretokenizer={"pat_str": "default"},
    unicode_bigrams={
      "top_k": 2,
      "min_freq": 1,
      "selected": 3,
      "cutoff_freq": 5,
      "max_excluded_freq": 4,
    },
    unique_words=1,
    weighted_occurrences=7,
  )

  assert manifest_path == word_inventory_manifest_path(words_path)
  manifest = load_word_inventory_manifest(words_path)
  assert manifest is not None
  assert manifest["words"]["file_name"] == "_words.json"
  assert bigram_frequency_guard(6, manifest) == {
    "cutoff_freq": 5,
    "max_excluded_freq": 4,
    "final_merge_freq": 6,
    "final_merge_minus_bigram_cutoff": 1,
    "final_merge_above_bigram_cutoff": True,
    "final_merge_above_max_excluded_freq": True,
  }
  assert bigram_frequency_guard(5, manifest)["final_merge_above_bigram_cutoff"] is False
  assert bigram_frequency_guard(None, manifest)["final_merge_above_bigram_cutoff"] is None


def test_word_inventory_manifest_rejects_mismatched_words_file(tmp_path: Path) -> None:
  words_path = tmp_path / "_words.json"
  words_path.write_text("{}", encoding="utf-8")
  manifest_path = word_inventory_manifest_path(words_path)
  manifest_path.write_text(
    json.dumps({
      "contract": "unitoken_word_inventory_manifest_v1",
      "words": {"file_name": "other.json"},
    }),
    encoding="utf-8",
  )

  with pytest.raises(ValueError, match="does not describe"):
    load_word_inventory_manifest(words_path)
