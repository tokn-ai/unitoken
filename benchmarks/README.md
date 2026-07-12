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
fixed_words_exact_hot_window_simulation_v1
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

Count and persist an exact two-pass word inventory directly from Parquet:

```bash
python benchmarks/count_parquet_source.py \
  --input-dir ~/NAS/ModelZoo/Corpus/FineWeb2/fineweb-2/data/cmn_Hani/train \
  --size-bytes 5368709120 \
  --json out/benchmarks/count_parquet_source/cmn_Hani_5GiB.json
```

The generated `_words.json` is written directly by Rust without constructing a
Python dictionary. Reload it with `pretokenizer.load_word_counter(path)` and
pass it to `trainer.add_word_counter(counter)` for native training ingestion.

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

Simulate bounded occurrence windows without changing exact merge selection:

```bash
cargo run --release --features analysis --bin analyze_hot_window -- \
  --words out/data/fineweb2/cmn_Hani/fineweb2_cmn_Hani_1GiB.unicode_bigram_top10k_min16/_words.json \
  --unit unicode \
  --vocab-size 10000 \
  --window-sizes 1024,4096,16384,65536 \
  --policy threshold-no-evict \
  --dataset-name cmn_Hani_1GiB \
  --config-name unicode \
  --experiment-name threshold_no_evict
```

The simulator uses the exact trainer as an oracle. On a cold winner, it
hydrates the current top-K pairs in one inventory scan. The
`threshold-no-evict` policy also admits a newly created pair immediately when
its frequency reaches the least frequency in the most recent top-K refill,
using the complete occurrence set emitted by that merge. The cutoff remains
fixed until another refill. Residents are not evicted, so K is a refill size
rather than a memory bound. Use `--policy replace-top-k` for
the original fixed-window baseline. Reports separate shared heap inspection
time from per-window hydration time and include payload-only memory estimates
for the full exact occurrence state and each simulated window.

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
