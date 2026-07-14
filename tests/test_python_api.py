from pathlib import Path
import threading

import numpy as np
import pytest

from uni_tokenizer import BpeEncoder, BpeModel, BpeTrainer, PreTokenizer


def test_pretokenizer_uses_pat_str_and_returns_words() -> None:
  pretokenizer = PreTokenizer([], pat_str=r"[^\s]")

  assert pretokenizer.get_words("ab a") == {"a": 2, "b": 1}


def test_encoder_uses_unit_and_singular_vocab() -> None:
  encoder = BpeEncoder(
    unit="byte",
    vocab={b"a": 0, b"b": 1, b"ab": 2},
    merges=[(b"a", b"b")],
    pat_str=r"\S+",
  )

  assert encoder.unit == "byte"
  assert encoder.encode("ab") == [2]
  encoded = encoder.encode_to_numpy("ab")
  assert isinstance(encoded, np.ndarray)
  assert encoded.tolist() == [2]


def test_encoder_can_disable_vocab_bigram_splitting() -> None:
  enabled = BpeEncoder(
    unit="byte",
    vocab={b"a": 0, b"b": 1, b"ab": 2, b"x": 3},
    merges=[(b"a", b"b")],
    pat_str=r"\S+",
  )
  disabled = BpeEncoder(
    unit="byte",
    vocab={b"a": 0, b"b": 1, b"ab": 2, b"x": 3},
    merges=[(b"a", b"b")],
    pat_str=r"\S+",
    split_on_vocab_bigrams=False,
  )
  text = "ab" + "x" * 32

  assert enabled._encoder.pre_tokenizer().get_words(text) == {"ab": 1, "x": 32}
  assert disabled._encoder.pre_tokenizer().get_words(text) == {text: 1}
  assert enabled.encode(text) == disabled.encode(text) == disabled.encode_word(text)


def test_encode_word_is_atomic_bpe_not_exact_vocab_lookup() -> None:
  encoder = BpeEncoder(
    unit="byte",
    vocab={b"a": 0, b"b": 1, b"ab": 2},
    merges=[],
    special_tokens=["ab"],
  )

  assert encoder.encode_word("ab") == [0, 1]
  assert encoder.encode_words(["ab"]) == [[0, 1]]
  assert encoder.encode("ab") == [2]


def test_unicode_encoder_rejects_mixed_fallback_byte_vocab_token() -> None:
  with pytest.raises(RuntimeError, match="only singleton fallback byte tokens are allowed"):
    BpeEncoder(
      unit="unicode",
      vocab={b"\x80": 0, b"a": 1, b"\x80a": 2},
      merges=[],
    )


def test_unicode_encoder_rejects_fallback_byte_merge() -> None:
  with pytest.raises(RuntimeError, match=r"Unicode merge 0 .* contains a fallback byte"):
    BpeEncoder(
      unit="unicode",
      vocab={b"\xc3": 0, b"\xa9": 1, b"\xc3\xa9": 2},
      merges=[(b"\xc3", b"\xa9")],
    )


def test_trainer_exposes_unit_and_singular_vocab() -> None:
  trainer = BpeTrainer(["<|endoftext|>"], unit="byte")

  assert trainer.unit == "byte"
  assert isinstance(trainer.vocab, dict)


def test_hot_pair_window_matches_exact_python_training() -> None:
  words = {"cab": 11, "eab": 9, "gab": 7, "abi": 5, "abj": 3, "abk": 1}

  def train(hot_pair_window_size: int | None) -> tuple[dict[bytes, int], int | None]:
    trainer = BpeTrainer([], unit="byte", hot_pair_window_size=hot_pair_window_size)
    trainer.add_words(words)
    trainer.train(vocab_size=260)
    trainer.validate_model()
    if hot_pair_window_size is None:
      assert trainer.hot_pair_window_stats is None
    else:
      assert trainer.hot_pair_window_stats is not None
      assert trainer.hot_pair_window_stats["hydration_scans"] >= 1
    return trainer.vocab, trainer.last_merge_freq

  assert train(2) == train(None)


def test_hot_pair_window_rejects_zero() -> None:
  with pytest.raises(ValueError, match="must be positive"):
    BpeTrainer([], hot_pair_window_size=0)


def test_unknown_unit_is_rejected() -> None:
  with pytest.raises(ValueError, match="Unknown unit"):
    BpeTrainer([], unit="characters")  # type: ignore[arg-type]


def test_gpt2_format_rejects_unicode_unit_without_creating_files(tmp_path: Path) -> None:
  trainer = BpeTrainer([], unit="unicode")
  vocab_file = tmp_path / "vocab.json"
  merges_file = tmp_path / "merges.txt"

  with pytest.raises(ValueError, match="not compatible"):
    trainer.save_files(vocab_file, merges_file, format="gpt2")

  assert not vocab_file.exists()
  assert not merges_file.exists()


@pytest.mark.parametrize("unit", ["byte", "unicode"])
def test_validate_model_rejects_duplicate_serialized_vocab(unit: str) -> None:
  # A one-character special token collides with the mandatory initial alphabet.
  trainer = BpeTrainer(["a"], unit=unit)  # type: ignore[arg-type]

  with pytest.raises(ValueError, match="duplicate vocabulary token a"):
    trainer.validate_model()


@pytest.mark.parametrize(
  ("unit", "word", "vocab_size"),
  [("byte", "ab", 257), ("unicode", "你好", 259)],
)
def test_train_and_validate_model_use_inclusive_bigram_cutoff(
  unit: str,
  word: str,
  vocab_size: int,
) -> None:
  trainer = BpeTrainer([], unit=unit, bigram_cutoff_freq=7)  # type: ignore[arg-type]
  trainer.add_words({word: 7})
  trainer.train(vocab_size=vocab_size)

  model = trainer.validate_model()

  assert model.last_merge_freq == 7


def test_manual_step_below_cutoff_is_rejected_when_saving(tmp_path: Path) -> None:
  trainer = BpeTrainer([], unit="byte", bigram_cutoff_freq=8)
  trainer.add_words({"ab": 7})
  trainer.init_training()
  trainer.step()
  vocab_path = tmp_path / "vocab.json"
  merges_path = tmp_path / "merges.txt"

  with pytest.raises(ValueError, match="must be at least.*cutoff frequency 8"):
    trainer.save_files(vocab_path, merges_path)

  assert not vocab_path.exists()
  assert not merges_path.exists()


@pytest.mark.parametrize("cutoff", [0, -1])
def test_trainer_rejects_non_positive_bigram_cutoff(cutoff: int) -> None:
  with pytest.raises(ValueError, match="bigram_cutoff_freq must be positive"):
    BpeTrainer([], bigram_cutoff_freq=cutoff)


@pytest.mark.parametrize(
  ("tie_break", "expected_tail"),
  [
    ("smallest_pair_id", ["你", "好", "你好"]),
    ("largest_content", ["好", "你", "你好"]),
  ],
)
def test_unicode_trainer_saves_only_loadable_merge_dependencies(
  tmp_path: Path,
  tie_break: str,
  expected_tail: list[str],
) -> None:
  trainer = BpeTrainer([], unit="unicode", tie_break=tie_break)  # type: ignore[arg-type]
  trainer.add_words({"你好": 1})

  for vocab_size, encoded_length in [(257, 4), (258, 2), (259, 1)]:
    # Repeated calls exercise rebuilding the candidate heap when training resumes.
    trainer.train(vocab_size=vocab_size)
    model = trainer.validate_model()
    assert isinstance(model, BpeModel)
    assert model.unit == "unicode"
    expected_last_merge_freq = None if vocab_size < 259 else 1
    assert trainer.last_merge_freq == expected_last_merge_freq
    assert model.last_merge_freq == expected_last_merge_freq

    tail = [
      token.decode("utf-8")
      for token, token_id in sorted(trainer.vocab.items(), key=lambda item: item[1])
      if token_id >= 256
    ]
    assert tail == expected_tail[:vocab_size - 256]

    vocab_file = tmp_path / f"vocab-{tie_break}-{vocab_size}.json"
    merges_file = tmp_path / f"merges-{tie_break}-{vocab_size}.txt"
    model.save_files(vocab_file, merges_file, format="unitoken")

    merge_lines = merges_file.read_text().splitlines()
    assert merge_lines == ([] if vocab_size < 259 else ["你 好 => 1"])

    encoder = BpeEncoder.load(
      unit="unicode",
      format="unitoken",
      vocab_file=vocab_file,
      merges_file=merges_file,
    )
    encoded = encoder.encode_word("你好")
    assert len(encoded) == encoded_length
    assert encoder.decode(encoded) == "你好"


def test_unicode_saved_merge_round_trips_ascii_byte_operand(tmp_path: Path) -> None:
  trainer = BpeTrainer([], unit="unicode")
  trainer.add_words({"a你": 1})
  trainer.train(vocab_size=258)
  model = trainer.validate_model()
  vocab_file = tmp_path / "vocab.json"
  merges_file = tmp_path / "merges.txt"

  model.save_vocab_json(vocab_file, format="unitoken")
  model.save_merges_txt(merges_file, format="unitoken")

  assert merges_file.read_text() == "a 你 => 1\n"
  encoder = BpeEncoder.load(
    unit="unicode",
    format="unitoken",
    vocab_file=vocab_file,
    merges_file=merges_file,
  )
  encoded = encoder.encode_word("a你")
  assert len(encoded) == 1
  assert encoder.decode(encoded) == "a你"


def test_validated_byte_model_saves_with_default_format(tmp_path: Path) -> None:
  trainer = BpeTrainer([], unit="byte")
  trainer.add_words({"ab": 1})
  trainer.train(vocab_size=257)
  model = trainer.validate_model()
  vocab_file = tmp_path / "vocab.json"
  merges_file = tmp_path / "merges.txt"

  model.save_files(vocab_file, merges_file)

  compat_vocab_file = tmp_path / "compat-vocab.json"
  compat_merges_file = tmp_path / "compat-merges.txt"
  trainer.save_files(compat_vocab_file, compat_merges_file)
  assert compat_vocab_file.read_bytes() == vocab_file.read_bytes()
  assert compat_merges_file.read_bytes() == merges_file.read_bytes()

  encoder = BpeEncoder.load(
    unit="byte",
    format="gpt2",
    vocab_file=vocab_file,
    merges_file=merges_file,
  )
  encoded = encoder.encode_word("ab")
  assert len(encoded) == 1
  assert encoder.decode(encoded) == "ab"

  disabled = BpeEncoder.load(
    unit="byte",
    format="gpt2",
    vocab_file=vocab_file,
    merges_file=merges_file,
    split_on_vocab_bigrams=False,
  )
  text = "ab" + "x" * 32
  assert encoder._encoder.pre_tokenizer().get_words(text) == {"ab": 1, "x": 32}
  assert disabled._encoder.pre_tokenizer().get_words(text) == {text: 1}
  assert encoder.encode(text) == disabled.encode(text)


def test_validated_model_is_an_immutable_trainer_snapshot() -> None:
  trainer = BpeTrainer([], unit="unicode")
  trainer.add_words({"你好": 1})
  trainer.train(vocab_size=257)
  model = trainer.validate_model()
  snapshot = model.vocab

  trainer.train(vocab_size=259)

  assert model.vocab == snapshot
  assert trainer.vocab != snapshot
  assert not hasattr(model, "train")


def test_source_counters_support_two_pass_replay_and_bounded_batches() -> None:
  class MemorySource:
    def __init__(self) -> None:
      self.scans = 0

    def scan(self):
      self.scans += 1
      yield "你好世界"
      yield "你好"

  source = MemorySource()
  pretokenizer = PreTokenizer([])
  bigram_counter = pretokenizer.bigram_counter()
  bigram_counter.add_source(source.scan(), max_records=1, max_bytes=8)
  selection = bigram_counter.select(top_k=1, min_freq=1)
  bigrams = selection.bigrams

  assert bigrams == ["你好"]
  assert selection.cutoff_freq == 2
  assert selection.max_excluded_freq == 1
  assert dict(bigram_counter.items())["你好"] == 2

  word_counter = pretokenizer.with_unicode_bigrams(bigrams).word_counter()
  word_counter.add_source(source.scan(), max_records=1, max_bytes=8)

  assert source.scans == 2
  assert word_counter.words() == {"世": 1, "你好": 2, "界": 1}


def test_file_bigram_selection_reports_frequency_boundary(tmp_path: Path) -> None:
  text = tmp_path / "corpus.txt"
  text.write_text("你好世界你好", encoding="utf-8")

  selection = PreTokenizer([]).select_unicode_bigrams_from_file(
    text,
    chunk_size=1024,
    boundary="utf8",
    top_k=1,
    min_freq=1,
  )

  assert selection.bigrams == ["你好"]
  assert selection.cutoff_freq == 2
  assert selection.max_excluded_freq == 1


def test_source_counter_merge() -> None:
  pretokenizer = PreTokenizer([], pat_str=r"[^\s]")
  left = pretokenizer.word_counter()
  left.add_text("ab")
  right = pretokenizer.word_counter()
  right.add_batch(["a", "c"])

  left.merge(right)

  assert left.words() == {"a": 2, "b": 1, "c": 1}


def test_source_counter_rejects_empty_batch_limits() -> None:
  counter = PreTokenizer([]).word_counter()

  with pytest.raises(ValueError, match="max_records"):
    counter.add_source(iter(["text"]), max_records=0)

  with pytest.raises(ValueError, match="max_bytes"):
    counter.add_source(iter(["text"]), max_bytes=0)

  for prefetch in [-1, 2]:
    advanced = False

    def source():
      nonlocal advanced
      advanced = True
      yield "text"

    with pytest.raises(ValueError, match="prefetch"):
      counter.add_source(source(), prefetch=prefetch)  # type: ignore[arg-type]
    assert not advanced


@pytest.mark.parametrize("prefetch", [0, 1])
def test_source_counter_prefetch_preserves_counts_and_source_thread(prefetch: int) -> None:
  caller_thread = threading.get_ident()
  source_threads: list[int] = []

  def source():
    for text in ["你好世界", "你好", "世界"]:
      source_threads.append(threading.get_ident())
      yield text

  pretokenizer = PreTokenizer([])
  bigram_counter = pretokenizer.bigram_counter()
  bigram_counter.add_source(source(), max_records=1, prefetch=prefetch)
  bigrams = bigram_counter.selected(top_k=2, min_freq=1)

  word_counter = pretokenizer.with_unicode_bigrams(bigrams).word_counter()
  word_counter.add_source(source(), max_records=1, prefetch=prefetch)

  assert source_threads == [caller_thread] * 6
  assert word_counter.words() == {"你好": 2, "世界": 2}


def test_source_counter_prefetch_acquires_iterator_once() -> None:
  caller_thread = threading.get_ident()

  class ThreadAffineIterator:
    def __init__(self) -> None:
      self.iter_calls = 0
      self.index = 0

    def __iter__(self):
      assert threading.get_ident() == caller_thread
      self.iter_calls += 1
      return self

    def __next__(self):
      assert threading.get_ident() == caller_thread
      if self.index == 2:
        raise StopIteration
      self.index += 1
      return "text"

  source = ThreadAffineIterator()
  counter = PreTokenizer([], pat_str=r"\S+").word_counter()
  counter.add_source(source, max_records=1)

  assert source.iter_calls == 1
  assert counter.words() == {"text": 2}


def test_source_counter_prefetch_matches_sync_for_boundaries_and_oversized_records() -> None:
  texts = ["", "ascii", "你好<eot>世界", "x" * 100]
  pretokenizer = PreTokenizer(["<eot>"], eot_token="<eot>", pat_str=r"\S+")

  sync = pretokenizer.word_counter()
  sync.add_source(iter(texts), max_records=2, max_bytes=5, prefetch=0)
  prefetched = pretokenizer.word_counter()
  prefetched.add_source(iter(texts), max_records=2, max_bytes=5, prefetch=1)

  assert prefetched.words() == sync.words()


@pytest.mark.parametrize("prefetch", [0, 1])
def test_source_counter_prefetch_preserves_iterator_error_boundary(prefetch: int) -> None:
  def source():
    yield "first"
    yield "unfinished"
    raise ValueError("source failed")

  counter = PreTokenizer([], pat_str=r"\S+").word_counter()

  with pytest.raises(ValueError, match="source failed"):
    counter.add_source(source(), max_records=1, prefetch=prefetch)

  assert counter.words() == {"first": 1}


@pytest.mark.parametrize("unit", ["byte", "unicode"])
def test_trainer_consumes_word_counter_without_changing_training(unit: str) -> None:
  pretokenizer = PreTokenizer([], pat_str=r"\S+")
  counter = pretokenizer.word_counter()
  counter.add_batch(["abab", "ab", "abab"])
  expected_words = counter.words()

  from_counter = BpeTrainer([], unit=unit)  # type: ignore[arg-type]
  from_counter.add_word_counter(counter)
  from_counter.train(vocab_size=from_counter.vocab_size + 2)

  from_mapping = BpeTrainer([], unit=unit)  # type: ignore[arg-type]
  from_mapping.add_words(expected_words)
  from_mapping.train(vocab_size=from_mapping.vocab_size + 2)

  assert counter.len == 0
  assert from_counter.vocab == from_mapping.vocab

  counter.add_text("new")
  assert counter.len == 1
  counter.clear()
  assert counter.len == 0


def test_word_counter_native_json_round_trip(tmp_path: Path) -> None:
  pretokenizer = PreTokenizer([], pat_str=r"\S+")
  counter = pretokenizer.word_counter()
  counter.add_batch(["你好", "hello", "你好"])
  path = tmp_path / "_words.json"

  counter.save(path)
  loaded = pretokenizer.load_word_counter(path)

  assert loaded.words() == {"hello": 1, "你好": 2}
