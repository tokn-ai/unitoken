unitoken
=======

[![CI](https://github.com/tokn-ai/unitoken/actions/workflows/ci.yml/badge.svg)](https://github.com/tokn-ai/unitoken/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/uni-tokenizer.svg)](https://pypi.org/project/uni-tokenizer/)
[![crates.io](https://img.shields.io/crates/v/unitoken.svg)](https://crates.io/crates/unitoken)
[![docs.rs](https://docs.rs/unitoken/badge.svg)](https://docs.rs/unitoken)

`unitoken` is a fast BPE tokenizer/trainer with a Rust core and optional Python bindings.

Install
-------

Rust:

```bash
cargo add unitoken
```

Python (wheels via PyPI):

```bash
pip install uni-tokenizer
```

Quickstart (Python)
-------------------

```python
from uni_tokenizer import BpeTrainer, BpeEncoder

trainer = BpeTrainer(["<|endoftext|>"], unit="byte")
trainer.add_words({"hello": 10, "world": 7})
trainer.train(vocab_size=256)
model = trainer.validate_model()
model.save("demo", format="gpt2")

enc = BpeEncoder.load("demo")
ids = enc.encode("hello")
```

Encoding uses model-vocabulary bigrams to partition long PAT words by default.
This does not change token ids, but it is not always profitable for byte
models. Disable it after benchmarking the target workload:

```python
enc = BpeEncoder.load("demo", split_on_vocab_bigrams=False)
```

For a model trained from a retained Unicode-bigram selection, configure its
inclusive cutoff on the trainer:

```python
trainer = BpeTrainer(
  [],
  unit="unicode",
  bigram_cutoff_freq=selection.cutoff_freq,
)
model = trainer.validate_model()
```

Automatic `train()` and `train_with_bbpe_fallback()` calls stop before a merge
below the cutoff. Manual `step()` calls remain unrestricted, while validation
rejects a final pair merge below the cutoff. Equality is valid because bigram
selection retains every tie at the cutoff.

Unicode BBPE fallback
---------------------

Unicode training can use part of its learned vocabulary for byte-BPE merges
inside Unicode scalars that are omitted from the direct Unicode alphabet:

```python
trainer = BpeTrainer(
  [],
  unit="unicode",
)
trainer.add_word_counter(word_counter)
trainer.train_with_bbpe_fallback(
  vocab_size=10_000,
  primary_vocab_ratio=0.9,
)
```

The ratio applies only to learned slots after special tokens and the mandatory
256-byte alphabet. Training first advances the primary Unicode trainer through
its configured share of learned slots, then freezes any still-unmaterialized
Unicode scalars. The fallback pass may use the remaining slots for byte merges
whose frequency reaches that primary boundary; unused fallback slots return to
primary pair training. The primary and fallback merge streams are combined by
frequency only after both phases finish. Fallback pseudo-words are isolated per
Unicode scalar, so the byte pass never learns across scalar boundaries.

`train_with_bbpe_fallback()` is only valid with `unit="unicode"`. Ordinary
`train()`, `init_training()`, and `step()` always perform normal Unicode BPE;
fallback is a separate, target-aware operation. A call that reserves fallback
slots must start before ordinary vocabulary growth because its phase boundary
depends on the requested vocabulary size. Once the fallback pass has run, the
trainer is finalized; create a new trainer for further training. A ratio of
`1.0` reserves no fallback slots, delegates to ordinary training, and remains
extendable. The resulting model is still a Unicode Unitoken model; encoding
behavior is carried by its merge rules, so no fallback option is needed when
loading it.

Streaming two-pass counting
---------------------------

For corpora that do not fit in memory, expose a replayable source whose
`scan()` method returns a fresh iterator of text records. Rust pulls and
processes bounded batches from each scan:

```python
from uni_tokenizer import BpeTrainer, PreTokenizer

pretokenizer = PreTokenizer([])

bigram_counter = pretokenizer.bigram_counter()
bigram_counter.add_source(source.scan())
bigrams = bigram_counter.selected(top_k=100_000, min_freq=16)

word_counter = pretokenizer.with_unicode_bigrams(bigrams).word_counter()
word_counter.add_source(source.scan())

trainer = BpeTrainer([], unit="byte")
trainer.add_word_counter(word_counter)
```

`add_source` defaults to at most 4,096 records or 64 MiB per batch. Override
`max_records` and `max_bytes` for the record sizes and worker memory available.
By default, it overlaps Python source iteration with Rust processing using one
bounded look-ahead batch; pass `prefetch=0` for synchronous processing.
Counters can also be merged, so separately counted corpus partitions can be
reduced before selecting bigrams or training. `add_word_counter` consumes the
native word inventory without constructing a Python dictionary; the counter is
empty and reusable afterward. `word_counter.words()` remains available for
small inventories, but copies the complete result into Python memory.

Bounded-memory BPE training
---------------------------

By default, the trainer retains occurrence sets for every discovered pair.
For large word inventories, `hot_pair_window_size` bounds persistent occurrence
sets while preserving exact global pair frequencies, winner selection, and
tie-breaking:

```python
trainer = BpeTrainer(
  [],
  unit="unicode",
  hot_pair_window_size=4096,
)
trainer.add_word_counter(word_counter)
trainer.train(vocab_size=10_000)
```

`4096` is a measured starting point, not a correctness setting. Smaller K uses
less memory but can require more full inventory scans when a cold pair wins.
Larger K retains more occurrence sets and generally reduces those scans. On a
cold winner, the trainer hydrates the exact current top K; newly created pairs
at or above the latest top-K frequency threshold are admitted immediately. If
resident pairs grow beyond 2K, they are pruned back to the exact top K.

On the 1 GiB FineWeb2 Chinese Unicode-bigram inventory used by the benchmark
suite (3,855,974 unique words, vocabulary size 10,000), one release run measured:

| occurrence mode | observed training peak RSS | total training | hydration scans |
|---|---:|---:|---:|
| exact (default) | 1,797 MiB | 5.58s | — |
| K=4096 | 1,649 MiB | 5.85s | 2 |

Both modes produced the same final merge frequency and model. Inspect
`trainer.hot_pair_window_stats` for hydration, pruning, resident-pair, and
occurrence-capacity diagnostics. Corpus shape and target vocabulary size affect
the best K, so benchmark representative inventories before changing the
default for a deployment.

Tiktoken-compatible API
-----------------------

`unitoken` also exposes a tiktoken-shaped Python API:

```python
from uni_tokenizer import Encoding

enc = Encoding.from_files(
  "demo",
  vocab_file="vocab.demo[u8].json",
  merges_file="merges.demo[u8].txt",
  special_tokens={"<|endoftext|>": 0},
)

ids = enc.encode("hello world")
text = enc.decode(ids)
```

The package also includes a `uni_tokenizer.tiktoken` namespace with `Encoding`,
`get_encoding`, `encoding_for_model`, `encoding_name_for_model`, and
`list_encoding_names`. Built-in registry names are limited to local unitoken
fixture models for now; use `Encoding.from_files(...)` for trained models.

Benchmark against tiktoken
--------------------------

Install the dev dependency and run:

```bash
uv pip install "tiktoken>=0.12.0"
python benchmarks/compare_tiktoken.py
```

The benchmark reports unitoken encode/decode timings and, when upstream
`tiktoken` is importable, matching upstream timings.

Benchmark training against Hugging Face
---------------------------------------

Install the dev dependency and run:

```bash
uv pip install "tokenizers>=0.22.1"
python benchmarks/compare_hf_training.py
```

The benchmark trains unitoken and Hugging Face `tokenizers` on the same
word-frequency fixture, checks that the learned byte-level BPE vocabularies
match, and reports median training speed.

For an end-to-end raw text comparison:

```bash
python benchmarks/compare_hf_training.py --text out/fineweb2_1GiB.txt --chunk-size 1048576 --boundary line --repeats 1
```

Raw text mode reports unitoken pretokenization and BPE training phases
separately, then compares the total against Hugging Face raw training. By
default, Hugging Face receives the same chunk boundaries as unitoken so vocab
parity is not affected by iterator boundary differences. Pass
`--hf-chunk-bytes` to force fixed byte chunks for Hugging Face.

Rust regression benchmark suites
--------------------------------

Complete benchmark profiles live in `benches/regression/config/`. A profile
can combine trainer, pretokenizer, and codec cases while keeping their inputs,
correctness hashes, run settings, and report names in one reviewable file:

```bash
cargo bench --bench regression -- suite smoke
cargo bench --bench regression -- suite 64mib
cargo bench --bench regression -- suite 1gib
```

`smoke.yml` uses checked-in fixtures, includes a 90% primary Unicode BBPE
trainer case plus a codec case for its pinned model, and is the profile run for
pull requests. Codec cases accept
`split_on_vocab_bigrams: false` for measured opt-out comparisons; reports and
encoder fingerprints record the selected value. The `64mib.yml` and `1gib.yml`
profiles compare ordinary and 90% primary BBPE training on the prepared
FineWeb2 Chinese word inventories under `out/data/`, so those local inputs must
exist before running them. Validate a profile without executing its cases with
`--check`:

```bash
cargo bench --bench regression -- suite 64mib --check
```

Run an unregistered profile with `--config`; relative config and input paths
are resolved from the repository root:

```bash
cargo bench --bench regression -- suite \
  --config benches/regression/config/smoke.yml \
  --output-dir /tmp/unitoken-regression
```

Report paths are relative to `--output-dir`. Pretokenizer outputs consumed by
later codec cases use logical artifact names, and validation requires every
artifact consumer to have exactly one producer in the same suite. The legacy
`smoke` subcommand remains a trainer-only shorthand whose cases now come from
`smoke.yml`; use `suite smoke` for the complete smoke pipeline.

Latest fixed-word trainer profile on the release build, using compressed
`_words.json` inventories and `vocab_size=10000`:

| dataset | unique words | occurrences | total train | train steps |
|---|---:|---:|---:|---:|
| FineWeb English 64MiB | 298,156 | 13,720,494 | 1.151s | 0.968s |
| FineWeb English 1GiB | 1,656,501 | 219,082,524 | 4.522s | 3.258s |
| FineWeb2 Chinese 64MiB | 1,803,009 | 5,774,521 | 26.681s | 20.416s |
| FineWeb2 Chinese bigram 64MiB | 606,153 | 15,901,831 | 3.702s | 3.034s |
| FineWeb2 Chinese bigram 1GiB | 3,855,974 | 249,919,657 | 20.197s | 14.169s |

The Chinese bigram rows use the unicode-bigram split inventory. The default
Chinese 1GiB inventory is intentionally omitted from this run; only the bigram
1GiB Chinese inventory was profiled.

Chunking supports explicit boundary modes:

- `auto`: split on the EOT token when present, otherwise line boundaries, then UTF-8 byte boundaries as a last resort.
- `eot`: split only on the EOT token.
- `line`: split on newline boundaries.
- `utf8`: split near byte boundaries while preserving valid UTF-8.

Use `--chunk-size BYTES` when you want target chunk size instead of a fixed
chunk count.

Prepare benchmark data
----------------------

To create a larger raw UTF-8 text sample from local FineWeb2 Parquet shards:

```bash
python benchmarks/create_fineweb2_sample.py --input-dir /path/to/fineweb2/10BT
```

This is a data-preparation step. Use the generated text with the CLI or a
separate benchmark that measures pretokenization/training on raw input.

Building from source
--------------------

This project uses `maturin` for the Python extension module.

```bash
maturin develop
```
