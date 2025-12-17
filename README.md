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

Building from source
--------------------

This project uses `maturin` for the Python extension module.

```bash
maturin develop
```
