unitoken
=======

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
python benchmarks/compare_hf_training.py --text out/fineweb2_1GiB.txt --repeats 1
```

Raw text mode reports unitoken pretokenization and BPE training phases
separately, then compares the total against Hugging Face raw training. By
default, Hugging Face receives the same chunk boundaries as unitoken so vocab
parity is not affected by iterator boundary differences. Pass
`--hf-chunk-bytes` to force fixed byte chunks for Hugging Face.

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
