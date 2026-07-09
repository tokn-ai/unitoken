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

trainer = BpeTrainer(["<|endoftext|>"])  # first token is treated as EOT
trainer.add_words({"hello": 10, "world": 7})
trainer.train(vocab_size=256)
trainer.save("demo")

enc = BpeEncoder.load("demo")
ids = enc.encode_word("hello")
```

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
