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

The package also includes a `tiktoken` import shim with `Encoding`,
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
`tiktoken` is importable, matching upstream timings. It refuses to compare
against unitoken's own `tiktoken` shim so accidental self-comparisons are
visible.

Building from source
--------------------

This project uses `maturin` for the Python extension module.

```bash
maturin develop
```
