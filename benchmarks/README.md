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

Rust regression benchmark
-------------------------

The authoritative core regression benchmark is the harness-free Rust target
in `benches/regression/`. It runs with Cargo's optimized `bench` profile and
spawns a fresh child process for every exact or bounded occurrence-storage
sample. Running it without a subcommand executes the checked-in byte and
Unicode smoke cases:

```bash
cargo bench --bench regression
```

Run a pinned word-frequency inventory at independent vocabulary checkpoints:

```bash
cargo bench --bench regression -- trainer \
  --words /path/to/_words.json \
  --unit unicode \
  --vocab-sizes 300,10000,100000 \
  --hot-pair-window-sizes 4096 \
  --rayon-threads 8 \
  --repeats 3 \
  --output out/benchmarks/regression/cmn_hani.json
```

Each report uses the versioned `unitoken_trainer_regression_v1` contract and
contains raw phase durations, phase-boundary RSS, 5 ms sampled peaks for trainer
construction and training, the cumulative whole-process high-water mark,
bounded-window statistics, and canonical SHA-256 fingerprints. The sampled
training peak starts after the JSON inventory has been transferred into the
trainer, so a transient inventory-loading high-water mark cannot mask it.
Fingerprints are computed from length-prefixed semantic model and final-word
state data rather than formatted vocabulary or merge files.
The parent writes the report even when a child or correctness gate fails, then
returns a nonzero status so CI retains an inspectable failure artifact.

Correctness gates require every run to reach its target, validate successfully,
remain deterministic across repeats, and produce identical exact/K models and
final word states. The checked-in smoke cases also pin golden semantic model
and inventory hashes, catching common-mode code changes and accidental fixture
changes. Pass `--expected-input-sha256` and `--expected-model-sha256` to add the
same guards to a custom run. Determinism is reported as `null` with one sample;
use `--repeats 2` or more to evaluate that gate. For a Unicode-bigram inventory,
pass `--bigram-cutoff-freq`; model validation then strictly requires
`last_merge_freq > bigram_cutoff_freq`, so equality fails.

Timing and RSS remain measurements rather than correctness gates. Compare them
only across reports produced on matching hardware, operating systems, thread
counts, and corpus artifacts. Reports record the CPU and hardware model,
logical CPU count, total memory, Rust version, and benchmark binary hash to make
that check explicit. Python may still prepare external Parquet or SQL inputs and
plot reports, but it is not in the measured Rust trainer path.

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

Profile the production bounded occurrence window against exact occurrence
storage:

```bash
python benchmarks/profile_training_core.py \
  --words out/data/fineweb2/cmn_Hani/fineweb2_cmn_Hani_1GiB.unicode_bigram_top10k_min16/_words.json \
  --unit unicode \
  --vocab-size 10000 \
  --hot-pair-window-size 4096 \
  --dataset-name cmn_Hani_1GiB \
  --config-name unicode \
  --experiment-name hot4096
```

K=4096 is the current measured starting point for this inventory shape, not a
correctness requirement or universal optimum. Exact storage remains the API
default. Smaller windows retain fewer occurrence sets but may trigger more
inventory hydration scans; larger windows trade memory for fewer scans.

On a cold winner, the trainer hydrates the exact current top-K pairs in one
inventory scan. Newly created pairs at or above the latest top-K frequency
threshold are admitted with complete occurrence sets. Crossing 2K resident
pairs releases occurrence sets back to K using the configured exact tie-break.
Reports include hydration and prune counters, resident occurrence-set capacity,
phase-level RSS, and the final merge frequency guard.

The final 1 GiB Unicode-bigram run (3,855,974 unique words, vocabulary size
10,000) measured 1,797 MiB observed training peak RSS and 5.58s in exact mode,
versus 1,649 MiB and 5.85s at K=4096. The bounded run used two hydration scans,
peaked at 4,847 resident pairs, required no batch prune, and matched the exact
final merge frequency of 4,183.

Unicode-bigram selection reports record `cutoff_freq`, the least retained
frequency after including cutoff ties, and `max_excluded_freq`. Training
reports record `final_merge_freq`. When selection and training happen in the
same experiment, `final_merge_above_bigram_cutoff` is the strict configuration
guard: equality or a lower final merge frequency is reported as a failed guard,
but the benchmark still completes so the failure can be inspected.

Saved word inventories may carry a sibling `_words.manifest.json`. The
manifest records source identity, pretokenizer settings, Unicode-bigram
selection boundaries, and inventory statistics. Fixed-word training reports
load this sidecar automatically and evaluate the frequency guard without
relying on directory names.

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
