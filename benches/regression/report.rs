use std::{collections::BTreeMap, path::PathBuf};

use serde::{Deserialize, Serialize};

use crate::{
  config::{CaseRequest, OccurrenceMode},
  fingerprint::ModelFingerprints,
};

pub const CONTRACT: &str = "unitoken_trainer_regression_v1";
pub const SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct InputReport {
  pub path: PathBuf,
  pub bytes: u64,
  pub sha256: String,
  pub unique_words: usize,
  pub weighted_occurrences: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TrainingCounts {
  pub initial_vocab_size: usize,
  pub final_vocab_size: usize,
  pub step_count: usize,
  pub merge_count: usize,
  pub last_merge_freq: Option<i64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StepBucket {
  pub start_vocab_size: usize,
  pub end_vocab_size: usize,
  pub step_count: usize,
  pub duration_ns: u64,
  pub current_rss_bytes: Option<u64>,
  pub process_peak_rss_bytes: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TimingReport {
  pub inventory_load_ns: u64,
  pub build_trainer_ns: u64,
  pub init_training_ns: u64,
  pub training_steps_ns: u64,
  pub validate_model_ns: u64,
  pub fingerprint_ns: u64,
  pub core_training_ns: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MemoryReport {
  pub current_rss_source: Option<String>,
  pub peak_rss_source: Option<String>,
  pub current_after_inventory_load_bytes: Option<u64>,
  pub current_after_trainer_build_bytes: Option<u64>,
  pub current_after_init_training_bytes: Option<u64>,
  pub current_after_training_bytes: Option<u64>,
  pub peak_after_inventory_load_bytes: Option<u64>,
  pub peak_after_trainer_build_bytes: Option<u64>,
  pub peak_after_init_training_bytes: Option<u64>,
  pub sampled_peak_during_trainer_build_bytes: Option<u64>,
  pub sampled_peak_during_training_bytes: Option<u64>,
  pub rss_sample_interval_ms: Option<u64>,
  pub process_peak_rss_through_training_bytes: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HotPairWindowReport {
  pub hydration_scans: u64,
  pub hydrated_word_entries: u64,
  pub batch_prunes: u64,
  pub prune_evictions: u64,
  pub peak_resident_pairs: usize,
  pub final_resident_pairs: usize,
  pub resident_occurrence_capacity_entries: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CaseMeasurement {
  pub input: InputReport,
  pub actual_rayon_threads: usize,
  pub counts: TrainingCounts,
  pub fingerprints: ModelFingerprints,
  pub timing: TimingReport,
  pub memory: MemoryReport,
  pub step_buckets: Vec<StepBucket>,
  pub model_valid: bool,
  pub target_vocab_reached: bool,
  pub final_merge_above_bigram_cutoff: Option<bool>,
  pub hot_pair_window: Option<HotPairWindowReport>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
  Completed,
  Failed,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RunFailure {
  pub phase: String,
  pub message: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CaseOutcome {
  pub case_id: String,
  pub request: CaseRequest,
  pub status: RunStatus,
  pub measurement: Option<CaseMeasurement>,
  pub error: Option<RunFailure>,
}

impl CaseOutcome {
  pub fn completed(request: CaseRequest, measurement: CaseMeasurement) -> Self {
    Self {
      case_id: request.id(),
      request,
      status: RunStatus::Completed,
      measurement: Some(measurement),
      error: None,
    }
  }

  pub fn failed(request: CaseRequest, phase: impl Into<String>, message: impl Into<String>) -> Self {
    Self {
      case_id: request.id(),
      request,
      status: RunStatus::Failed,
      measurement: None,
      error: Some(RunFailure {
        phase: phase.into(),
        message: message.into(),
      }),
    }
  }
}

#[derive(Clone, Debug, Serialize)]
pub struct EnvironmentReport {
  pub git_commit: Option<String>,
  pub git_dirty: Option<bool>,
  pub rustc: Option<String>,
  pub os: String,
  pub arch: String,
  pub cpu_model: Option<String>,
  pub hardware_model: Option<String>,
  pub logical_cpus: usize,
  pub total_memory_bytes: Option<u64>,
  pub profile: String,
  pub debug_assertions: bool,
  pub benchmark_binary_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ComparisonReport {
  pub case_name: String,
  pub bounded_hot_pair_window_size: usize,
  pub exact_case_ids: Vec<String>,
  pub bounded_case_ids: Vec<String>,
  pub input_sha256_match: bool,
  pub vocab_sha256_match: bool,
  pub merges_sha256_match: bool,
  pub model_sha256_match: bool,
  pub word_state_sha256_match: bool,
  pub counts_match: bool,
  pub last_merge_freq_match: bool,
  pub passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct GateReport {
  pub all_runs_completed: bool,
  pub target_vocab_reached: bool,
  pub models_valid: bool,
  pub samples_deterministic: Option<bool>,
  pub bounded_matches_exact: Option<bool>,
  pub inputs_match_expected: Option<bool>,
  pub exact_matches_golden: Option<bool>,
  pub final_merge_above_bigram_cutoff: Option<bool>,
  pub passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct SuiteReport {
  pub schema_version: u32,
  pub contract: String,
  pub suite_name: String,
  pub generated_at_unix_seconds: u64,
  pub environment: EnvironmentReport,
  pub samples: Vec<CaseOutcome>,
  pub comparisons: Vec<ComparisonReport>,
  pub gates: GateReport,
}

impl SuiteReport {
  pub fn new(
    suite_name: String,
    generated_at_unix_seconds: u64,
    environment: EnvironmentReport,
    samples: Vec<CaseOutcome>,
  ) -> Self {
    let comparisons = compare_exact_and_bounded(&samples);
    let gates = evaluate_gates(&samples, &comparisons);
    Self {
      schema_version: SCHEMA_VERSION,
      contract: CONTRACT.to_string(),
      suite_name,
      generated_at_unix_seconds,
      environment,
      samples,
      comparisons,
      gates,
    }
  }
}

fn compare_exact_and_bounded(samples: &[CaseOutcome]) -> Vec<ComparisonReport> {
  let mut by_case = BTreeMap::<&str, Vec<&CaseOutcome>>::new();
  for sample in samples {
    by_case.entry(&sample.request.case.name).or_default().push(sample);
  }

  let mut comparisons = Vec::new();
  for (case_name, case_samples) in by_case {
    let exact = case_samples
      .iter()
      .copied()
      .filter(|sample| sample.request.variant.occurrence_mode == OccurrenceMode::Exact)
      .collect::<Vec<_>>();
    let mut bounded_by_size = BTreeMap::<usize, Vec<&CaseOutcome>>::new();
    for sample in case_samples
      .iter()
      .copied()
      .filter(|sample| sample.request.variant.occurrence_mode == OccurrenceMode::Bounded)
    {
      if let Some(size) = sample.request.variant.hot_pair_window_size {
        bounded_by_size.entry(size).or_default().push(sample);
      }
    }

    for (window_size, bounded) in bounded_by_size {
      let exact_measurements = exact
        .iter()
        .filter_map(|sample| sample.measurement.as_ref())
        .collect::<Vec<_>>();
      let bounded_measurements = bounded
        .iter()
        .filter_map(|sample| sample.measurement.as_ref())
        .collect::<Vec<_>>();
      let input_sha256_match = all_cross_match(&exact_measurements, &bounded_measurements, |measurement| {
        &measurement.input.sha256
      });
      let vocab_sha256_match = all_cross_match(&exact_measurements, &bounded_measurements, |measurement| {
        &measurement.fingerprints.vocab_sha256
      });
      let merges_sha256_match = all_cross_match(&exact_measurements, &bounded_measurements, |measurement| {
        &measurement.fingerprints.merges_sha256
      });
      let model_sha256_match = all_cross_match(&exact_measurements, &bounded_measurements, |measurement| {
        &measurement.fingerprints.model_sha256
      });
      let word_state_sha256_match = all_cross_match(&exact_measurements, &bounded_measurements, |measurement| {
        &measurement.fingerprints.word_state_sha256
      });
      let counts_match = all_cross_match(&exact_measurements, &bounded_measurements, |measurement| {
        &measurement.counts
      });
      let last_merge_freq_match = all_cross_match(&exact_measurements, &bounded_measurements, |measurement| {
        &measurement.counts.last_merge_freq
      });
      let completed = exact.len() == exact_measurements.len()
        && bounded.len() == bounded_measurements.len()
        && !exact_measurements.is_empty()
        && !bounded_measurements.is_empty();
      let passed = completed
        && input_sha256_match
        && vocab_sha256_match
        && merges_sha256_match
        && model_sha256_match
        && word_state_sha256_match
        && counts_match
        && last_merge_freq_match;
      comparisons.push(ComparisonReport {
        case_name: case_name.to_string(),
        bounded_hot_pair_window_size: window_size,
        exact_case_ids: exact.iter().map(|sample| sample.case_id.clone()).collect(),
        bounded_case_ids: bounded.iter().map(|sample| sample.case_id.clone()).collect(),
        input_sha256_match,
        vocab_sha256_match,
        merges_sha256_match,
        model_sha256_match,
        word_state_sha256_match,
        counts_match,
        last_merge_freq_match,
        passed,
      });
    }
  }
  comparisons
}

fn all_cross_match<'a, T, F>(exact: &'a [&CaseMeasurement], bounded: &'a [&CaseMeasurement], value: F) -> bool
where
  T: PartialEq + ?Sized + 'a,
  F: Fn(&'a CaseMeasurement) -> &'a T,
{
  let Some(first) = exact.first().map(|measurement| value(measurement)) else {
    return false;
  };
  exact
    .iter()
    .chain(bounded.iter())
    .all(|measurement| value(measurement) == first)
}

fn evaluate_gates(samples: &[CaseOutcome], comparisons: &[ComparisonReport]) -> GateReport {
  let completed = samples
    .iter()
    .filter_map(|sample| sample.measurement.as_ref())
    .collect::<Vec<_>>();
  let all_runs_completed = !samples.is_empty()
    && samples.iter().all(|sample| sample.status == RunStatus::Completed)
    && completed.len() == samples.len();
  let target_vocab_reached = all_runs_completed && completed.iter().all(|measurement| measurement.target_vocab_reached);
  let models_valid = all_runs_completed && completed.iter().all(|measurement| measurement.model_valid);
  let samples_deterministic = deterministic_within_variants(samples);
  let bounded_matches_exact = (!comparisons.is_empty()).then(|| comparisons.iter().all(|comparison| comparison.passed));
  let inputs_match_expected = optional_gate(
    samples,
    |sample| sample.request.case.expected_input_sha256.is_some(),
    |sample, measurement| {
      sample
        .request
        .case
        .expected_input_sha256
        .as_ref()
        .is_some_and(|expected| measurement.input.sha256.eq_ignore_ascii_case(expected))
    },
  );
  let exact_matches_golden = optional_gate(
    samples,
    |sample| {
      sample.request.variant.occurrence_mode == OccurrenceMode::Exact
        && sample.request.case.expected_model_sha256.is_some()
    },
    |sample, measurement| {
      sample
        .request
        .case
        .expected_model_sha256
        .as_ref()
        .is_some_and(|expected| measurement.fingerprints.model_sha256.eq_ignore_ascii_case(expected))
    },
  );
  let final_merge_above_bigram_cutoff = optional_gate(
    samples,
    |sample| sample.request.case.bigram_cutoff_freq.is_some(),
    |sample, measurement| {
      sample
        .request
        .case
        .bigram_cutoff_freq
        .map(|cutoff| {
          measurement
            .counts
            .last_merge_freq
            .is_some_and(|frequency| frequency > cutoff)
        })
        .unwrap_or(false)
    },
  );
  let passed = all_runs_completed
    && target_vocab_reached
    && models_valid
    && samples_deterministic != Some(false)
    && bounded_matches_exact != Some(false)
    && inputs_match_expected != Some(false)
    && exact_matches_golden != Some(false)
    && final_merge_above_bigram_cutoff != Some(false);

  GateReport {
    all_runs_completed,
    target_vocab_reached,
    models_valid,
    samples_deterministic,
    bounded_matches_exact,
    inputs_match_expected,
    exact_matches_golden,
    final_merge_above_bigram_cutoff,
    passed,
  }
}

fn deterministic_within_variants(samples: &[CaseOutcome]) -> Option<bool> {
  let mut grouped = BTreeMap::<(String, String), Vec<&CaseMeasurement>>::new();
  for sample in samples {
    let Some(measurement) = sample.measurement.as_ref() else {
      return Some(false);
    };
    grouped
      .entry((sample.request.case.name.clone(), sample.request.variant.label()))
      .or_default()
      .push(measurement);
  }
  if grouped.is_empty() || grouped.values().any(|measurements| measurements.len() < 2) {
    return None;
  }
  Some(grouped.values().all(|measurements| {
    let Some(first) = measurements.first() else {
      return false;
    };
    measurements.iter().all(|measurement| {
      measurement.input.sha256 == first.input.sha256
        && measurement.counts == first.counts
        && measurement.fingerprints == first.fingerprints
    })
  }))
}

fn optional_gate<C, F>(samples: &[CaseOutcome], configured: C, check: F) -> Option<bool>
where
  C: Fn(&CaseOutcome) -> bool,
  F: Fn(&CaseOutcome, &CaseMeasurement) -> bool,
{
  let mut saw_gate = false;
  let mut passed = true;
  for sample in samples {
    if !configured(sample) {
      continue;
    }
    saw_gate = true;
    let Some(measurement) = sample.measurement.as_ref() else {
      passed = false;
      continue;
    };
    passed &= check(sample, measurement);
  }
  saw_gate.then_some(passed)
}
