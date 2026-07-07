# Benchmarks

Benchmark scripts live in this directory. Generated artifacts should go under
`out/` by artifact kind:

```text
out/
  data/          # sampled corpora and derived corpus inventories
  models/        # tokenizer vocab/merges artifacts
  encoded/       # encoded token arrays
  reports/       # smoke, metrics, and test reports
  benchmarks/    # benchmark measurement reports
```

Recommended benchmark report directories:

```text
out/benchmarks/training/
out/benchmarks/pretokenizer/
out/benchmarks/trainer/
out/benchmarks/tiktoken/
```

Example FineWeb2 sample:

```bash
python benchmarks/create_fineweb2_sample.py \
  --input-dir ~/NAS/ModelZoo/Corpus/FineWeb2/fineweb/sample/10BT \
  --output out/data/fineweb2/fineweb2_1GiB.txt \
  --json out/data/fineweb2/fineweb2_1GiB.sample.json
```

Example training comparison:

```bash
python benchmarks/compare_hf_training.py \
  --text out/data/fineweb2/fineweb2_1GiB.txt \
  --boundary eot \
  --vocab-size 10000 \
  --json out/benchmarks/training/fineweb2_1GiB.compare_hf.target_vocab_10000.json
```
