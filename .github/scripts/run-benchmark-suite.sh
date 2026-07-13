#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 <checkout> <output-directory>" >&2
  exit 2
fi

checkout=$1
output_dir=$2

mkdir -p "$output_dir"
cd "$checkout"

# Keep base/head ordering from turning fixture page-cache state into a PR delta.
find fixtures -maxdepth 1 -type f -exec sha256sum {} + >/dev/null

cargo bench --bench regression --no-run
cargo bench --bench regression -- smoke \
  --repeats 2 \
  --output "$output_dir/trainer.json"
cargo bench --bench regression -- pretokenizer \
  --text fixtures/TinyStories_all_data_zh_1M-sample.txt \
  --name ci_zh_pretokenizer \
  --chunk-size 1048576 \
  --unicode-bigram-top-k 100 \
  --unicode-bigram-min-freq 2 \
  --unicode-bigram-mixed-boundary split \
  --unicode-bigrams-output "$output_dir/unicode_bigrams.json" \
  --repeats 2 \
  --expected-input-sha256 c298b1680c4378091ad9e39126ac0858d78e547f3744d1a30442c12adac8e9f3 \
  --expected-bigrams-sha256 2e88788add465df26296cd56b0369b9d1056850e5e7fe5584c19354b191288f7 \
  --expected-inventory-sha256 c749404bcf209b877e75215c94ce297c7aa4f7511e7ee3eb7dd6ae2ee71735cf \
  --output "$output_dir/pretokenizer.json"
cargo bench --bench regression -- codec \
  --text fixtures/tinystories_sample_5M.txt \
  --vocab fixtures/vocab.tinystories_sample_5M.json \
  --merges fixtures/merges.tinystories_sample_5M.txt \
  --unit byte \
  --format gpt2 \
  --name ci_en_codec \
  --chunks 8 \
  --repeats 2 \
  --expected-input-sha256 7cc2577b9e1f9ed703b13ca651103bf421c91912fcbcb2d024213858e0981d87 \
  --expected-vocab-sha256 ad20f9939e447a91cba3875775bcdeec208c0849187d16870d50aec95b499492 \
  --expected-merges-sha256 948f039e415c8448e5223c02fc22d4b4a2f4e31b64005e7119da3d1ca10ada72 \
  --expected-token-count 1424317 \
  --expected-token-sha256 59054b2420d287b3f243e9fb3eda7fc827cc83f49450ef272c438600bc1bf6e2 \
  --output "$output_dir/codec-byte.json"
cargo bench --bench regression -- codec \
  --text fixtures/TinyStories_all_data_zh_1M-sample.txt \
  --vocab fixtures/vocab.TinyStories_all_data_zh_1M-sample.uni.json \
  --merges fixtures/merges.TinyStories_all_data_zh_1M-sample.uni.txt \
  --unit unicode \
  --format unitoken \
  --name ci_zh_codec \
  --chunks 8 \
  --unicode-bigrams "$output_dir/unicode_bigrams.json" \
  --unicode-bigram-mixed-boundary split \
  --repeats 2 \
  --expected-input-sha256 c298b1680c4378091ad9e39126ac0858d78e547f3744d1a30442c12adac8e9f3 \
  --expected-vocab-sha256 8931292e5594186561164f5da2c8557e1e99ae41b647e548c3ef996b08d233d6 \
  --expected-merges-sha256 2c347f27fb94b61afddfb82a0d910c1142d4076f8d74175c30aa77d9b2c24297 \
  --expected-token-count 886572 \
  --expected-token-sha256 532a8595665ca19acc8e74b423258c09314d22316730498dd41348d860381180 \
  --output "$output_dir/codec-unicode.json"
