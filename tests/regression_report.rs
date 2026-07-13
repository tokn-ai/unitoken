#![allow(dead_code)]

#[path = "../benches/regression/config.rs"]
mod config;
#[path = "../benches/regression/fingerprint.rs"]
mod fingerprint;
#[path = "../benches/regression/report.rs"]
mod report;
#[path = "../benches/regression/util.rs"]
mod util;

use std::{collections::BTreeMap, fs, path::PathBuf};

use config::{CaseConfig, CaseRequest, InitialAlphabetName, OccurrenceVariant, TieBreakName, Unit};
use fingerprint::{
  ModelFingerprints, fingerprint_model, fingerprint_token_ids, fingerprint_unicode_bigrams,
  fingerprint_word_counts, sha256_hex,
};
use report::{
  CaseMeasurement, CaseOutcome, EnvironmentReport, InputReport, MemoryReport, RunStatus, SuiteReport, TimingReport,
  TrainingCounts,
};
use util::FileIdentity;

const INPUT_SHA256: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const MODEL_SHA256: &str = "2222222222222222222222222222222222222222222222222222222222222222";
const OTHER_MODEL_SHA256: &str = "3333333333333333333333333333333333333333333333333333333333333333";

#[test]
fn semantic_fingerprint_is_stable_and_unit_scoped() {
  use unitoken::bpe::{BpeTrainer, CharIdx, Character, Idx};

  let mut byte = BpeTrainer::<u8, Idx>::from_words([("ab", 7)], &[]);
  byte.train_until(257).unwrap();
  let byte_model = byte.validate_model().unwrap();
  let byte_fingerprint = fingerprint_model(&byte_model, &byte.words).unwrap();
  assert_eq!(byte_fingerprint, fingerprint_model(&byte_model, &byte.words).unwrap(),);

  let mut unicode = BpeTrainer::<Character, CharIdx>::from_words([("ab", 7)], &[]);
  unicode.train_until(257).unwrap();
  let unicode_model = unicode.validate_model().unwrap();
  let unicode_fingerprint = fingerprint_model(&unicode_model, &unicode.words).unwrap();

  assert_ne!(byte_fingerprint.model_sha256, unicode_fingerprint.model_sha256);
  assert_eq!(
    sha256_hex(b"abc"),
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
  );
}

#[test]
fn unicode_bigram_fingerprint_is_canonical_and_stable() {
  let left = ahash::AHashSet::from_iter([('你', '好'), ('世', '界')]);
  let right = ahash::AHashSet::from_iter([('世', '界'), ('你', '好')]);
  let fingerprint = fingerprint_unicode_bigrams(&left, Some(7), Some(6));
  assert_eq!(
    fingerprint,
    fingerprint_unicode_bigrams(&right, Some(7), Some(6)),
  );
  assert_eq!(
    fingerprint,
    "8a1119f324ec0ac8e187eaa67f4778b8d62bfb2038a86b90f870a9f358dd354a",
  );
  assert_eq!(
    fingerprint_unicode_bigrams(&left, None, None),
    "aeef9e5e16bbf6657816d4cf7c472fd7aad3f1486f4f224ed3ce7ee5500884a4",
  );
  assert_eq!(
    fingerprint_unicode_bigrams(&left, Some(0), Some(0)),
    "ea792ed37e186f3d94190a90a1c2feb06a09be1b8fd8a9b0a17be3eae93b0af9",
  );
  assert_ne!(
    fingerprint_unicode_bigrams(&left, None, None),
    fingerprint_unicode_bigrams(&left, Some(0), Some(0)),
  );
}

#[test]
fn word_count_fingerprint_is_stable() {
  let words = BTreeMap::from([("hello".to_string(), 3), ("world".to_string(), 2)]);
  assert_eq!(
    fingerprint_word_counts(&words),
    "51dac90b89ff08bb5448d61ec4ca2a0afd5ee8e087d24e3a8332894059e26684",
  );
}

#[test]
fn token_id_fingerprint_is_raw_u32_le() {
  assert_eq!(
    fingerprint_token_ids(&[1, 2, 3]),
    "4636993d3e1da4e9d6b8f87b79e8f7c6d018580d52661950eabc3845c5897a4d",
  );
  assert_eq!(
    fingerprint_token_ids(&[1, 2, 3]),
    sha256_hex(&[1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]),
  );
}

#[test]
fn file_identity_detects_changes_during_a_benchmark() {
  let path = std::env::temp_dir().join(format!(
    "unitoken-regression-file-identity-{}",
    std::process::id(),
  ));
  fs::write(&path, b"before").unwrap();
  let identity = FileIdentity::capture(&path).unwrap();
  fs::write(&path, b"after-change").unwrap();
  assert!(identity.ensure_unchanged(&path).is_err());
  fs::remove_file(path).unwrap();
}

#[test]
fn singleton_samples_leave_determinism_unchecked() {
  let case = case_config(Some(INPUT_SHA256), Some(MODEL_SHA256), None);
  let report = suite(vec![
    completed(&case, OccurrenceVariant::exact(), 0, MODEL_SHA256, 8),
    completed(&case, OccurrenceVariant::bounded(4096), 0, MODEL_SHA256, 8),
  ]);

  assert_eq!(report.gates.samples_deterministic, None);
  assert_eq!(report.gates.inputs_match_expected, Some(true));
  assert_eq!(report.gates.exact_matches_golden, Some(true));
  assert_eq!(report.gates.bounded_matches_exact, Some(true));
  assert!(report.gates.passed);
}

#[test]
fn repeated_model_change_fails_determinism_and_parity() {
  let case = case_config(Some(INPUT_SHA256), None, None);
  let report = suite(vec![
    completed(&case, OccurrenceVariant::exact(), 0, MODEL_SHA256, 8),
    completed(&case, OccurrenceVariant::exact(), 1, OTHER_MODEL_SHA256, 8),
    completed(&case, OccurrenceVariant::bounded(4096), 0, MODEL_SHA256, 8),
    completed(&case, OccurrenceVariant::bounded(4096), 1, MODEL_SHA256, 8),
  ]);

  assert_eq!(report.gates.samples_deterministic, Some(false));
  assert_eq!(report.gates.bounded_matches_exact, Some(false));
  assert!(!report.gates.passed);
}

#[test]
fn failed_child_cannot_pass_comparison_gates() {
  let case = case_config(Some(INPUT_SHA256), None, None);
  let bounded_request = request(&case, OccurrenceVariant::bounded(4096), 0);
  let report = suite(vec![
    completed(&case, OccurrenceVariant::exact(), 0, MODEL_SHA256, 8),
    CaseOutcome::failed(bounded_request, "child_process", "terminated"),
  ]);

  assert!(!report.gates.all_runs_completed);
  assert_eq!(report.gates.samples_deterministic, Some(false));
  assert_eq!(report.gates.bounded_matches_exact, Some(false));
  assert!(!report.gates.passed);
}

#[test]
fn golden_model_mismatch_fails_the_suite() {
  let case = case_config(Some(INPUT_SHA256), Some(OTHER_MODEL_SHA256), None);
  let report = suite(vec![
    completed(&case, OccurrenceVariant::exact(), 0, MODEL_SHA256, 8),
    completed(&case, OccurrenceVariant::bounded(4096), 0, MODEL_SHA256, 8),
  ]);

  assert_eq!(report.gates.exact_matches_golden, Some(false));
  assert!(!report.gates.passed);
}

#[test]
fn cutoff_equality_passes_for_an_otherwise_valid_measurement() {
  let case = case_config(Some(INPUT_SHA256), None, Some(8));
  let report = suite(vec![
    completed(&case, OccurrenceVariant::exact(), 0, MODEL_SHA256, 8),
    completed(&case, OccurrenceVariant::bounded(4096), 0, MODEL_SHA256, 8),
  ]);

  assert_eq!(report.gates.final_merge_at_or_above_bigram_cutoff, Some(true));
  assert!(report.gates.passed);
}

fn suite(samples: Vec<CaseOutcome>) -> SuiteReport {
  SuiteReport::new(
    "test".to_string(),
    0,
    EnvironmentReport {
      git_commit: None,
      git_dirty: None,
      rustc: None,
      os: "test".to_string(),
      arch: "test".to_string(),
      cpu_model: None,
      hardware_model: None,
      logical_cpus: 1,
      total_memory_bytes: None,
      profile: "test".to_string(),
      debug_assertions: true,
      benchmark_binary_sha256: None,
    },
    samples,
  )
}

fn case_config(
  expected_input_sha256: Option<&str>,
  expected_model_sha256: Option<&str>,
  bigram_cutoff_freq: Option<i64>,
) -> CaseConfig {
  CaseConfig {
    name: "case".to_string(),
    words_path: PathBuf::from("fixture.json"),
    unit: Unit::Byte,
    initial_alphabet: InitialAlphabetName::RawBytes,
    tie_break: TieBreakName::SmallestPairId,
    parallel_merge_min_occurs_in: None,
    target_vocab_size: 300,
    special_tokens: Vec::new(),
    bucket_size: 100,
    bigram_cutoff_freq,
    expected_input_sha256: expected_input_sha256.map(str::to_string),
    expected_model_sha256: expected_model_sha256.map(str::to_string),
    rayon_threads: 1,
  }
}

fn request(case: &CaseConfig, variant: OccurrenceVariant, sample_index: usize) -> CaseRequest {
  CaseRequest {
    case: case.clone(),
    variant,
    sample_index,
  }
}

fn completed(
  case: &CaseConfig,
  variant: OccurrenceVariant,
  sample_index: usize,
  model_sha256: &str,
  last_merge_freq: i64,
) -> CaseOutcome {
  let request = request(case, variant, sample_index);
  CaseOutcome {
    case_id: request.id(),
    request,
    status: RunStatus::Completed,
    measurement: Some(measurement(model_sha256, last_merge_freq, case.bigram_cutoff_freq)),
    error: None,
  }
}

fn measurement(model_sha256: &str, last_merge_freq: i64, bigram_cutoff_freq: Option<i64>) -> CaseMeasurement {
  CaseMeasurement {
    input: InputReport {
      path: PathBuf::from("fixture.json"),
      bytes: 10,
      sha256: INPUT_SHA256.to_string(),
      unique_words: 1,
      weighted_occurrences: 1,
    },
    actual_rayon_threads: 1,
    counts: TrainingCounts {
      initial_vocab_size: 256,
      final_vocab_size: 300,
      step_count: 44,
      merge_count: 44,
      last_merge_freq: Some(last_merge_freq),
    },
    fingerprints: ModelFingerprints {
      vocab_sha256: model_sha256.to_string(),
      merges_sha256: model_sha256.to_string(),
      model_sha256: model_sha256.to_string(),
      word_state_sha256: model_sha256.to_string(),
    },
    timing: TimingReport {
      inventory_load_ns: 1,
      build_trainer_ns: 1,
      init_training_ns: 1,
      training_steps_ns: 1,
      validate_model_ns: 1,
      fingerprint_ns: 1,
      core_training_ns: 3,
    },
    memory: MemoryReport {
      current_rss_source: None,
      peak_rss_source: None,
      current_after_inventory_load_bytes: None,
      current_after_trainer_build_bytes: None,
      current_after_init_training_bytes: None,
      current_after_training_bytes: None,
      peak_after_inventory_load_bytes: None,
      peak_after_trainer_build_bytes: None,
      peak_after_init_training_bytes: None,
      sampled_peak_during_trainer_build_bytes: None,
      sampled_peak_during_training_bytes: None,
      rss_sample_interval_ms: None,
      process_peak_rss_through_training_bytes: None,
    },
    step_buckets: Vec::new(),
    model_valid: true,
    target_vocab_reached: true,
    final_merge_at_or_above_bigram_cutoff: bigram_cutoff_freq.map(|cutoff| last_merge_freq >= cutoff),
    hot_pair_window: None,
  }
}
