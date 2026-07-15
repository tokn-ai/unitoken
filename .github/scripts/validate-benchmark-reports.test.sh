#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
validator="$script_dir/validate-benchmark-reports.sh"
temporary_dir=$(mktemp -d "${TMPDIR:-/tmp}/unitoken-report-contract.XXXXXX")
trap 'rm -rf "$temporary_dir"' EXIT
max_report_bytes=$((2 * 1024 * 1024))

write_report() {
  local output_dir=$1
  local relative_path=$2
  local contract=$3
  local passed=${4:-true}

  mkdir -p "$output_dir"
  jq -n \
    --arg contract "$contract" \
    --argjson passed "$passed" \
    '{
      schema_version: 1,
      contract: $contract,
      samples: [],
      gates: { passed: $passed }
    }' > "$output_dir/$relative_path"
}

write_complete_report_set() {
  local output_dir=$1

  write_report "$output_dir" trainer.json unitoken_trainer_regression_v1
  write_report "$output_dir" pretokenizer.json unitoken_pretokenizer_regression_v1
  write_report "$output_dir" codec-byte.json unitoken_codec_regression_v1
  write_report "$output_dir" codec-unicode.json unitoken_codec_regression_v1
}

pad_report_to_size() {
  local report=$1
  local target_size=$2
  local current_size
  local padding

  current_size=$(wc -c < "$report")
  padding=$((target_size - current_size))
  if (( padding < 0 )); then
    echo "cannot shrink $report to $target_size bytes" >&2
    exit 1
  fi
  if (( padding > 0 )); then
    dd if=/dev/zero bs="$padding" count=1 2>/dev/null \
      | tr '\000' ' ' >> "$report"
  fi
}

expect_failure() {
  local expected_message=$1
  local output
  shift

  if output=$("$@" 2>&1); then
    echo "expected command to fail: $*" >&2
    exit 1
  fi
  if [[ "$output" != *"$expected_message"* ]]; then
    echo "failure did not contain '$expected_message': $output" >&2
    exit 1
  fi
}

baseline="$temporary_dir/baseline"
candidate="$temporary_dir/candidate"
write_complete_report_set "$baseline"
write_complete_report_set "$candidate"

bash "$validator" "$baseline" "$candidate"

mv "$candidate/trainer.json" "$candidate/trainer-renamed.json"
expect_failure "candidate benchmark report is missing or not a regular file: trainer.json" \
  bash "$validator" "$baseline" "$candidate"
mv "$candidate/trainer-renamed.json" "$candidate/trainer.json"

mv "$candidate/pretokenizer.json" "$candidate/pretokenizer-target.json"
ln -s pretokenizer-target.json "$candidate/pretokenizer.json"
expect_failure "candidate benchmark report is missing or not a regular file: pretokenizer.json" \
  bash "$validator" "$baseline" "$candidate"
rm "$candidate/pretokenizer.json"
mv "$candidate/pretokenizer-target.json" "$candidate/pretokenizer.json"

pad_report_to_size "$candidate/trainer.json" "$max_report_bytes"
bash "$validator" "$baseline" "$candidate"
printf ' ' >> "$candidate/trainer.json"
expect_failure "candidate benchmark report exceeds 2 MiB: trainer.json" \
  bash "$validator" "$baseline" "$candidate"
write_report "$candidate" trainer.json unitoken_trainer_regression_v1

write_report "$candidate/extra" trainer.json unitoken_trainer_regression_v1
bash "$validator" "$baseline" "$candidate"

write_report "$candidate" codec-byte.json unitoken_codec_regression_v1 false
expect_failure "candidate benchmark report has an invalid structure or failed gates: codec-byte.json" \
  bash "$validator" "$baseline" "$candidate"

write_report "$candidate" codec-byte.json unitoken_pretokenizer_regression_v1
expect_failure "candidate benchmark report has an invalid structure or failed gates: codec-byte.json" \
  bash "$validator" "$baseline" "$candidate"

echo "benchmark report contract tests passed"
