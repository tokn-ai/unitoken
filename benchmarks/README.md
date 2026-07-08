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

Benchmark report paths default to:

```text
out/benchmarks/{script_name}/{dataset_name}.{config_name}.{experiment_name}.vocab{N}.json
```

For reports that do not have a vocab size, the `.vocab{N}` segment is omitted.
`config_name` should be a short distinct key such as `default`, `eot16m`, or
`ubigram10k`; full configuration details belong inside the JSON report.

Training benchmarks use explicit contracts in their JSON metadata:

```text
fixed_words_unitoken_training_core_profile
fixed_words_unitoken_vs_hf_expanded_iterator
raw_text_unitoken_trainer_profile
raw_text_unitoken_vs_hf
```

Use `fixed_words_unitoken_training_core_profile` to isolate unitoken trainer
changes against a compressed `(word, frequency)` inventory. Hugging Face does
not receive that same compressed representation through the Python API; reports
with `hf_expanded_iterator` explicitly expand counts into repeated words.

Use `raw_text_unitoken_trainer_profile` to profile unitoken end-to-end training
from raw text. Use `raw_text_unitoken_vs_hf` for end-to-end implementation
comparisons against Hugging Face.

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
  --dataset-name fineweb2_1GiB \
  --config-name eot16m \
  --experiment-name baseline_release
```

Example fixed-words unitoken trainer profile:

```bash
python benchmarks/profile_training_core.py \
  --words out/data/fineweb2/cmn_Hani/fineweb2_cmn_Hani_1GiB.unicode_bigram_top10k_min16/_words.json \
  --vocab-size 1000 \
  --dataset-name cmn_Hani_1GiB \
  --config-name ubigram10k \
  --experiment-name baseline_release
```

Example raw-text unitoken trainer profile:

```bash
python benchmarks/profile_trainer.py \
  --text out/data/fineweb2/fineweb2_1GiB.txt \
  --boundary eot \
  --vocab-size 10000 \
  --dataset-name fineweb2_1GiB \
  --config-name eot16m \
  --experiment-name baseline_release
```
