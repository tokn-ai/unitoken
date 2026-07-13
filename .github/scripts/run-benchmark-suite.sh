#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 <checkout> <output-directory> <suite-config>" >&2
  exit 2
fi

checkout=$1
output_dir=$2
suite_config=$3

mkdir -p "$output_dir"
cd "$checkout"

# Keep base/head ordering from turning fixture page-cache state into a PR delta.
find fixtures -maxdepth 1 -type f -exec sha256sum {} + >/dev/null

cargo bench --bench regression --no-run

# Both revisions consume the candidate config so a config-only PR still
# compares identical work. Inputs remain relative to each checkout.
cargo bench --bench regression -- suite \
  --config "$suite_config" \
  --output-dir "$output_dir"
