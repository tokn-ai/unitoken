#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 <baseline-output-directory> <candidate-output-directory>" >&2
  exit 2
fi

baseline_dir=$1
candidate_dir=$2
max_report_bytes=$((2 * 1024 * 1024))

validate_report() {
  local revision=$1
  local output_dir=$2
  local relative_path=$3
  local expected_contract=$4
  local report="$output_dir/$relative_path"

  if [[ -L "$report" || ! -f "$report" ]]; then
    echo "$revision benchmark report is missing or not a regular file: $relative_path" >&2
    return 1
  fi

  local report_size
  report_size=$(wc -c < "$report")
  if (( report_size > max_report_bytes )); then
    echo "$revision benchmark report exceeds 2 MiB: $relative_path" >&2
    return 1
  fi

  if ! jq -e --arg contract "$expected_contract" '
    type == "object"
      and .schema_version == 1
      and .contract == $contract
      and (.samples | type == "array")
      and .gates.passed == true
  ' "$report" >/dev/null; then
    echo "$revision benchmark report has an invalid structure or failed gates: $relative_path" >&2
    return 1
  fi
}

status=0
for revision in baseline candidate; do
  if [[ "$revision" == baseline ]]; then
    output_dir=$baseline_dir
  else
    output_dir=$candidate_dir
  fi

  validate_report "$revision" "$output_dir" trainer.json \
    unitoken_trainer_regression_v1 || status=1
  validate_report "$revision" "$output_dir" pretokenizer.json \
    unitoken_pretokenizer_regression_v1 || status=1
  validate_report "$revision" "$output_dir" codec-byte.json \
    unitoken_codec_regression_v1 || status=1
  validate_report "$revision" "$output_dir" codec-unicode.json \
    unitoken_codec_regression_v1 || status=1
done

exit "$status"
