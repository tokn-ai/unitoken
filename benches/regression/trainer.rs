use std::{
  collections::BTreeSet,
  path::{Path, PathBuf},
  process::ExitStatus,
};

use clap::Args as ClapArgs;
use unitoken::pretokenizer::DEFAULT_EOT;

use crate::common::{
  config::Unit,
  environment::{default_suite_report_path, environment_report, resolve_threads},
  process::{run_isolated_protocol, run_protocol_child, validate_outcome_shape, write_json_atomic},
  report::RunStatus,
  util::{format_bytes, now_seconds, resolve_path_for_comparison},
};

use self::{
  config::{CaseConfig, CaseRequest, InitialAlphabetName, OccurrenceVariant, TieBreakName},
  report::{CaseOutcome, SuiteReport},
};

pub(crate) mod config {
  use std::{collections::BTreeSet, path::PathBuf};

  use clap::ValueEnum;
  use serde::{Deserialize, Serialize};

  use crate::common::config::Unit;

  #[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
  #[serde(rename_all = "snake_case")]
  #[value(rename_all = "snake_case")]
  pub enum InitialAlphabetName {
    RawBytes,
    ByteLevel,
  }

  #[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
  #[serde(rename_all = "snake_case")]
  #[value(rename_all = "snake_case")]
  pub enum TieBreakName {
    SmallestPairId,
    LargestContent,
  }

  #[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
  #[serde(rename_all = "snake_case")]
  pub enum OccurrenceMode {
    Exact,
    Bounded,
  }

  #[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
  pub struct OccurrenceVariant {
    pub occurrence_mode: OccurrenceMode,
    pub hot_pair_window_size: Option<usize>,
  }

  impl OccurrenceVariant {
    pub fn exact() -> Self {
      Self {
        occurrence_mode: OccurrenceMode::Exact,
        hot_pair_window_size: None,
      }
    }

    pub fn bounded(hot_pair_window_size: usize) -> Self {
      Self {
        occurrence_mode: OccurrenceMode::Bounded,
        hot_pair_window_size: Some(hot_pair_window_size),
      }
    }

    pub fn label(&self) -> String {
      match self.occurrence_mode {
        OccurrenceMode::Exact => "exact".to_string(),
        OccurrenceMode::Bounded => format!(
          "k{}",
          self.hot_pair_window_size.expect("bounded variants have a window size"),
        ),
      }
    }

    pub fn validate(&self) -> Result<(), String> {
      match (self.occurrence_mode, self.hot_pair_window_size) {
        (OccurrenceMode::Exact, None) => Ok(()),
        (OccurrenceMode::Bounded, Some(size)) if size > 0 => Ok(()),
        (OccurrenceMode::Exact, Some(_)) => Err("exact occurrence mode cannot set hot_pair_window_size".to_string()),
        (OccurrenceMode::Bounded, None | Some(0)) => {
          Err("bounded occurrence mode requires a positive hot_pair_window_size".to_string())
        }
        (OccurrenceMode::Bounded, Some(_)) => unreachable!("positive window size matched earlier"),
      }
    }
  }

  #[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
  pub struct CaseConfig {
    pub name: String,
    pub words_path: PathBuf,
    pub unit: Unit,
    pub initial_alphabet: InitialAlphabetName,
    pub tie_break: TieBreakName,
    pub parallel_merge_min_occurs_in: Option<usize>,
    pub target_vocab_size: usize,
    pub special_tokens: Vec<String>,
    pub bucket_size: usize,
    pub bigram_cutoff_freq: Option<i64>,
    pub expected_input_sha256: Option<String>,
    pub expected_model_sha256: Option<String>,
    pub rayon_threads: usize,
  }

  impl CaseConfig {
    pub fn validate(&self) -> Result<(), String> {
      if self.name.trim().is_empty() {
        return Err("case name cannot be empty".to_string());
      }
      if self.bucket_size == 0 {
        return Err(format!("case {} has a zero bucket_size", self.name));
      }
      if self.rayon_threads == 0 {
        return Err(format!("case {} has zero Rayon threads", self.name));
      }
      if self.parallel_merge_min_occurs_in == Some(0) {
        return Err(format!("case {} has a zero parallel_merge_min_occurs_in", self.name,));
      }
      if self.bigram_cutoff_freq.is_some_and(|cutoff| cutoff <= 0) {
        return Err(format!("case {} has a non-positive bigram cutoff", self.name));
      }
      validate_sha256(
        &self.name,
        "expected_input_sha256",
        self.expected_input_sha256.as_deref(),
      )?;
      validate_sha256(
        &self.name,
        "expected_model_sha256",
        self.expected_model_sha256.as_deref(),
      )?;
      let minimum_vocab_size = 256usize.saturating_add(self.special_tokens.len());
      if self.target_vocab_size < minimum_vocab_size {
        return Err(format!(
          "case {} targets vocabulary {}, below the initial vocabulary {}",
          self.name, self.target_vocab_size, minimum_vocab_size,
        ));
      }
      let unique_special_tokens = self.special_tokens.iter().collect::<BTreeSet<_>>();
      if unique_special_tokens.len() != self.special_tokens.len() {
        return Err(format!("case {} contains duplicate special tokens", self.name));
      }
      Ok(())
    }
  }

  fn validate_sha256(case_name: &str, field: &str, value: Option<&str>) -> Result<(), String> {
    let Some(value) = value else {
      return Ok(());
    };
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
      return Err(format!(
        "case {case_name} has an invalid {field}; expected 64 hexadecimal characters",
      ));
    }
    Ok(())
  }

  #[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
  pub struct CaseRequest {
    pub case: CaseConfig,
    pub variant: OccurrenceVariant,
    pub sample_index: usize,
  }

  impl CaseRequest {
    pub fn id(&self) -> String {
      format!("{}__{}__r{}", self.case.name, self.variant.label(), self.sample_index,)
    }

    pub fn validate(&self) -> Result<(), String> {
      self.case.validate()?;
      self.variant.validate()
    }
  }
}

pub(crate) mod report {
  use std::{collections::BTreeMap, path::PathBuf};

  use serde::{Deserialize, Serialize};

  use crate::common::{
    fingerprint::ModelFingerprints,
    report::{EnvironmentReport, RunFailure, RunStatus},
  };

  use super::config::{CaseRequest, OccurrenceMode};

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
    pub final_merge_at_or_above_bigram_cutoff: Option<bool>,
    pub hot_pair_window: Option<HotPairWindowReport>,
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
    pub final_merge_at_or_above_bigram_cutoff: Option<bool>,
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
    let final_merge_at_or_above_bigram_cutoff = optional_gate(
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
              .is_none_or(|frequency| frequency >= cutoff)
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
      && final_merge_at_or_above_bigram_cutoff != Some(false);

    GateReport {
      all_runs_completed,
      target_vocab_reached,
      models_valid,
      samples_deterministic,
      bounded_matches_exact,
      inputs_match_expected,
      exact_matches_golden,
      final_merge_at_or_above_bigram_cutoff,
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
}

mod runner {
  use std::{
    fmt::Display,
    fs,
    time::Instant,
  };

  use ordermap::OrderMap;
  use unitoken::{
    bpe::{
      BpeTrainer, BpeTrainerConfig, CharIdx, CharSplit, CharToIdx, Character, Freq, HasChar, Idx, IdxLike,
      InitialAlphabet, TieBreak, Word, utils::WordDebugExt,
    },
    traits::{CanStrToWord, CanToWord, CanTrain, Train},
  };

  use crate::common::{
    fingerprint::{CanonicalId, CanonicalUnit, fingerprint_model, sha256_hex},
    rss,
    util::duration_ns,
  };

  use crate::common::config::Unit;

  use super::{
    config::{CaseRequest, InitialAlphabetName, TieBreakName},
    report::{
      CaseMeasurement, CaseOutcome, HotPairWindowReport, InputReport, MemoryReport, StepBucket, TimingReport,
      TrainingCounts,
    },
  };

  struct LoadedInventory {
    words: OrderMap<String, Freq>,
    input: InputReport,
    load_ns: u64,
    current_rss_bytes: Option<u64>,
    peak_rss_bytes: Option<u64>,
  }

  #[derive(Debug)]
  struct CaseError {
    phase: &'static str,
    message: String,
  }

  impl CaseError {
    fn new(phase: &'static str, message: impl Into<String>) -> Self {
      Self {
        phase,
        message: message.into(),
      }
    }

    fn from_error(phase: &'static str, error: impl Display) -> Self {
      Self::new(phase, error.to_string())
    }
  }

  pub fn execute_case(request: CaseRequest) -> CaseOutcome {
    if let Err(error) = request.validate() {
      return CaseOutcome::failed(request, "configuration", error);
    }

    let pool = match rayon::ThreadPoolBuilder::new()
      .num_threads(request.case.rayon_threads)
      .build()
    {
      Ok(pool) => pool,
      Err(error) => {
        return CaseOutcome::failed(request, "rayon_pool", error.to_string());
      }
    };
    match pool.install(|| execute_case_inner(&request)) {
      Ok(measurement) => CaseOutcome::completed(request, measurement),
      Err(error) => CaseOutcome::failed(request, error.phase, error.message),
    }
  }

  fn execute_case_inner(request: &CaseRequest) -> Result<CaseMeasurement, CaseError> {
    let inventory = load_inventory(request)?;
    match request.case.unit {
      Unit::Byte => run_training::<u8, Idx>(request, inventory),
      Unit::Unicode => run_training::<Character, CharIdx>(request, inventory),
    }
  }

  fn load_inventory(request: &CaseRequest) -> Result<LoadedInventory, CaseError> {
    let started = Instant::now();
    let bytes = fs::read(&request.case.words_path).map_err(|error| CaseError::from_error("inventory_load", error))?;
    let sha256 = sha256_hex(&bytes);
    if let Some(expected) = request.case.expected_input_sha256.as_ref()
      && !expected.eq_ignore_ascii_case(&sha256)
    {
      return Err(CaseError::new(
        "inventory_validate",
        format!("inventory SHA-256 {sha256} does not match expected {expected}"),
      ));
    }
    let words = serde_json::from_slice::<OrderMap<String, Freq>>(&bytes)
      .map_err(|error| CaseError::from_error("inventory_parse", error))?;
    if words.is_empty() {
      return Err(CaseError::new("inventory_validate", "word inventory is empty"));
    }

    let mut weighted_occurrences = 0u64;
    for (word, frequency) in &words {
      if *frequency <= 0 {
        return Err(CaseError::new(
          "inventory_validate",
          format!("word {word:?} has non-positive frequency {frequency}"),
        ));
      }
      weighted_occurrences = weighted_occurrences
        .checked_add(*frequency as u64)
        .ok_or_else(|| CaseError::new("inventory_validate", "weighted occurrence count overflowed"))?;
    }

    let input = InputReport {
      path: fs::canonicalize(&request.case.words_path).unwrap_or_else(|_| request.case.words_path.clone()),
      bytes: bytes.len() as u64,
      sha256,
      unique_words: words.len(),
      weighted_occurrences,
    };
    Ok(LoadedInventory {
      words,
      input,
      load_ns: duration_ns(started.elapsed()),
      current_rss_bytes: rss::current_rss_bytes(),
      peak_rss_bytes: rss::process_peak_rss_bytes(),
    })
  }

  fn run_training<C, I>(request: &CaseRequest, inventory: LoadedInventory) -> Result<CaseMeasurement, CaseError>
  where
    C: CanStrToWord + CanToWord<u8> + CanonicalUnit + CharSplit + CharToIdx<I> + Clone + Ord + Send + Sync + 'static,
    I: CanonicalId + HasChar<C> + IdxLike,
    Word<C>: WordDebugExt,
    BpeTrainer<C, I>: CanTrain<C, I>,
  {
    let LoadedInventory {
      words,
      input,
      load_ns,
      current_rss_bytes: current_after_inventory_load_bytes,
      peak_rss_bytes: peak_after_inventory_load_bytes,
    } = inventory;
    let config = BpeTrainerConfig {
      initial_alphabet: match request.case.initial_alphabet {
        InitialAlphabetName::RawBytes => InitialAlphabet::RawBytes,
        InitialAlphabetName::ByteLevel => InitialAlphabet::ByteLevel,
      },
      tie_break: match request.case.tie_break {
        TieBreakName::SmallestPairId => TieBreak::SmallestPairId,
        TieBreakName::LargestContent => TieBreak::LargestContent,
      },
      parallel_merge_min_occurs_in: request.case.parallel_merge_min_occurs_in,
      hot_pair_window_size: request.variant.hot_pair_window_size,
      bigram_cutoff_freq: request.case.bigram_cutoff_freq,
    };

    let build_rss_sampler = rss::RssSampler::start();
    let started = Instant::now();
    let mut trainer = BpeTrainer::<C, I>::from_words_with_config(words, &request.case.special_tokens, config);
    let build_trainer_ns = duration_ns(started.elapsed());
    let initial_vocab_size = trainer.vocab_size();
    let current_after_trainer_build_bytes = rss::current_rss_bytes();
    let peak_after_trainer_build_bytes = rss::process_peak_rss_bytes();
    let sampled_peak_during_trainer_build_bytes = build_rss_sampler.map(rss::RssSampler::finish);

    let training_rss_sampler = rss::RssSampler::start();
    let started = Instant::now();
    trainer.init_training();
    let init_training_ns = duration_ns(started.elapsed());
    let current_after_init_training_bytes = rss::current_rss_bytes();
    let peak_after_init_training_bytes = rss::process_peak_rss_bytes();
    if let Some(sampler) = training_rss_sampler.as_ref() {
      sampler.observe();
    }

    let mut step_count = 0usize;
    let mut training_steps_ns = 0u64;
    let mut step_buckets = Vec::new();
    while trainer.vocab_size() < request.case.target_vocab_size {
      let start_vocab_size = trainer.vocab_size();
      let started = Instant::now();
      let mut bucket_steps = 0usize;
      while bucket_steps < request.case.bucket_size && trainer.vocab_size() < request.case.target_vocab_size {
        trainer
          .step()
          .map_err(|error| CaseError::from_error("training_steps", error))?;
        bucket_steps += 1;
        step_count += 1;
      }
      let bucket_ns = duration_ns(started.elapsed());
      training_steps_ns = training_steps_ns.saturating_add(bucket_ns);
      step_buckets.push(StepBucket {
        start_vocab_size,
        end_vocab_size: trainer.vocab_size(),
        step_count: bucket_steps,
        duration_ns: bucket_ns,
        current_rss_bytes: rss::current_rss_bytes(),
        process_peak_rss_bytes: rss::process_peak_rss_bytes(),
      });
      if let Some(sampler) = training_rss_sampler.as_ref() {
        sampler.observe();
      }
    }

    let final_vocab_size = trainer.vocab_size();
    if final_vocab_size != request.case.target_vocab_size {
      return Err(CaseError::new(
        "target_vocab",
        format!(
          "training reached vocabulary {final_vocab_size}, expected exactly {}",
          request.case.target_vocab_size,
        ),
      ));
    }
    let current_after_training_bytes = rss::current_rss_bytes();
    let process_peak_rss_through_training_bytes = rss::process_peak_rss_bytes();
    let sampled_peak_during_training_bytes = training_rss_sampler.map(rss::RssSampler::finish);
    let rss_sample_interval_ms = (sampled_peak_during_trainer_build_bytes.is_some()
      || sampled_peak_during_training_bytes.is_some())
    .then_some(rss::SAMPLE_INTERVAL.as_millis() as u64);
    let last_merge_freq = trainer.last_merge_freq();
    let hot_pair_window = trainer.hot_pair_window_stats().map(|stats| HotPairWindowReport {
      hydration_scans: stats.hydration_scans,
      hydrated_word_entries: stats.hydrated_word_entries,
      batch_prunes: stats.batch_prunes,
      prune_evictions: stats.prune_evictions,
      peak_resident_pairs: stats.peak_resident_pairs,
      final_resident_pairs: trainer.hot_resident_pairs(),
      resident_occurrence_capacity_entries: trainer.hot_occurrence_capacity(),
    });
    let counts = TrainingCounts {
      initial_vocab_size,
      final_vocab_size,
      step_count,
      merge_count: trainer.merges.len(),
      last_merge_freq,
    };

    let started = Instant::now();
    let model = trainer.validate_model()
    .map_err(|error| CaseError::from_error("validate_model", error))?;
    let validate_model_ns = duration_ns(started.elapsed());

    let started = Instant::now();
    let fingerprints = fingerprint_model(&model, &trainer.words).map_err(|error| CaseError::new("fingerprint", error))?;
    let fingerprint_ns = duration_ns(started.elapsed());
    let final_merge_at_or_above_bigram_cutoff = request
      .case
      .bigram_cutoff_freq
      .map(|cutoff| last_merge_freq.is_none_or(|frequency| frequency >= cutoff));
    let core_training_ns = build_trainer_ns
      .saturating_add(init_training_ns)
      .saturating_add(training_steps_ns);

    Ok(CaseMeasurement {
      input,
      actual_rayon_threads: rayon::current_num_threads(),
      counts,
      fingerprints,
      timing: TimingReport {
        inventory_load_ns: load_ns,
        build_trainer_ns,
        init_training_ns,
        training_steps_ns,
        validate_model_ns,
        fingerprint_ns,
        core_training_ns,
      },
      memory: MemoryReport {
        current_rss_source: rss::current_rss_source().map(str::to_string),
        peak_rss_source: rss::peak_rss_source().map(str::to_string),
        current_after_inventory_load_bytes,
        current_after_trainer_build_bytes,
        current_after_init_training_bytes,
        current_after_training_bytes,
        peak_after_inventory_load_bytes,
        peak_after_trainer_build_bytes,
        peak_after_init_training_bytes,
        sampled_peak_during_trainer_build_bytes,
        sampled_peak_during_training_bytes,
        rss_sample_interval_ms,
        process_peak_rss_through_training_bytes,
      },
      step_buckets,
      model_valid: true,
      target_vocab_reached: final_vocab_size == request.case.target_vocab_size,
      final_merge_at_or_above_bigram_cutoff,
      hot_pair_window,
    })
  }
}

#[derive(Clone, Debug, ClapArgs)]
pub struct SuiteOptions {
  /// Repeat every exact/bounded variant to check determinism.
  #[arg(long, default_value_t = 1)]
  repeats: usize,
  /// Bounded occurrence-window sizes to compare with exact mode.
  #[arg(long, value_delimiter = ',', default_value = "4096")]
  hot_pair_window_sizes: Vec<usize>,
  /// Rayon worker threads. Defaults to available parallelism, resolved once.
  #[arg(long)]
  rayon_threads: Option<usize>,
  /// Merge steps per timing/RSS observation bucket.
  #[arg(long, default_value_t = 500)]
  bucket_size: usize,
  /// JSON report path. Defaults below out/benchmarks/regression/.
  #[arg(long)]
  output: Option<PathBuf>,
}

impl Default for SuiteOptions {
  fn default() -> Self {
    Self {
      repeats: 1,
      hot_pair_window_sizes: vec![4096],
      rayon_threads: None,
      bucket_size: 500,
      output: None,
    }
  }
}

#[derive(Clone, Debug, ClapArgs)]
pub struct Args {
  /// Stable JSON object mapping pretokenized words to positive frequencies.
  #[arg(long)]
  words: PathBuf,
  /// Benchmark case prefix. Defaults to the inventory file stem.
  #[arg(long)]
  name: Option<String>,
  #[arg(long, value_enum)]
  unit: Unit,
  /// Independent vocabulary checkpoints, each run in a fresh process.
  #[arg(long, value_delimiter = ',', default_value = "10000")]
  vocab_sizes: Vec<usize>,
  #[arg(long, value_enum, default_value = "raw_bytes")]
  initial_alphabet: InitialAlphabetName,
  #[arg(long, value_enum, default_value = "smallest_pair_id")]
  tie_break: TieBreakName,
  #[arg(long)]
  parallel_merge_min_occurs_in: Option<usize>,
  /// Require the final pair merge frequency to be at least this value.
  #[arg(long)]
  bigram_cutoff_freq: Option<i64>,
  /// Optional golden semantic model hash. Requires one vocabulary checkpoint.
  #[arg(long)]
  expected_model_sha256: Option<String>,
  /// Optional SHA-256 of the exact inventory bytes.
  #[arg(long)]
  expected_input_sha256: Option<String>,
  /// Reserved special token. Repeat to configure more than one.
  #[arg(long = "special-token")]
  special_tokens: Vec<String>,
  #[command(flatten)]
  suite: SuiteOptions,
}


pub fn run_smoke(options: SuiteOptions) -> Result<(), String> {
  let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
  let rayon_threads = resolve_threads(options.rayon_threads)?;
  let cases = [
    (
      "smoke_en_byte_v300",
      "fixtures/_words.tinystories_sample_5M.json",
      Unit::Byte,
      300usize,
      "20b257111ca6e5ce81ee0d0e78924b9987db13029d7d006e4eb981cca151c9f4",
      "fa65e898d4cec1be5b78732ec4738b20213856a2de73bba5ca34366d347e91c0",
    ),
    (
      "smoke_en_byte_v1000",
      "fixtures/_words.tinystories_sample_5M.json",
      Unit::Byte,
      1000usize,
      "20b257111ca6e5ce81ee0d0e78924b9987db13029d7d006e4eb981cca151c9f4",
      "197a9f7d6ec3630370b1a30e0392b0f2fbcd2de1d36ee4d05884f01f2a877be9",
    ),
    (
      "smoke_zh_unicode_v300",
      "fixtures/_words.TinyStories_all_data_zh_1M-sample.json",
      Unit::Unicode,
      300usize,
      "ffb74990eb0b04ca0986a24ead7acf63e5483df7afb68c65ad2c397497a67c6a",
      "b3f2e74a4b169244774d71cd289d246847d4a56e585411436c1e4c44219e7b3a",
    ),
    (
      "smoke_zh_unicode_v1000",
      "fixtures/_words.TinyStories_all_data_zh_1M-sample.json",
      Unit::Unicode,
      1000usize,
      "ffb74990eb0b04ca0986a24ead7acf63e5483df7afb68c65ad2c397497a67c6a",
      "34dcb3aeb65c2220f50158d594defb73f1d5649b296c0020220266ba70f1d9e1",
    ),
  ]
  .into_iter()
  .map(
    |(name, relative_path, unit, target_vocab_size, expected_input_sha256, expected_model_sha256)| CaseConfig {
      name: name.to_string(),
      words_path: manifest_dir.join(relative_path),
      unit,
      initial_alphabet: InitialAlphabetName::RawBytes,
      tie_break: TieBreakName::SmallestPairId,
      parallel_merge_min_occurs_in: None,
      target_vocab_size,
      special_tokens: vec![DEFAULT_EOT.to_string()],
      bucket_size: options.bucket_size,
      bigram_cutoff_freq: None,
      expected_input_sha256: Some(expected_input_sha256.to_string()),
      expected_model_sha256: Some(expected_model_sha256.to_string()),
      rayon_threads,
    },
  )
  .collect::<Vec<_>>();
  run_suite("smoke", cases, options)
}

pub fn run(args: Args) -> Result<(), String> {
  if args.vocab_sizes.is_empty() {
    return Err("at least one --vocab-sizes value is required".to_string());
  }
  if args.expected_model_sha256.is_some() && args.vocab_sizes.len() != 1 {
    return Err("--expected-model-sha256 requires exactly one vocabulary checkpoint".to_string());
  }
  let rayon_threads = resolve_threads(args.suite.rayon_threads)?;
  let case_prefix = args.name.unwrap_or_else(|| {
    args
      .words
      .file_stem()
      .and_then(|name| name.to_str())
      .unwrap_or("trainer")
      .to_string()
  });
  let special_tokens = if args.special_tokens.is_empty() {
    vec![DEFAULT_EOT.to_string()]
  } else {
    args.special_tokens
  };
  let mut seen_vocab_sizes = BTreeSet::new();
  let mut cases = Vec::new();
  for target_vocab_size in args.vocab_sizes {
    if !seen_vocab_sizes.insert(target_vocab_size) {
      return Err(format!("duplicate vocabulary checkpoint {target_vocab_size}"));
    }
    cases.push(CaseConfig {
      name: format!("{case_prefix}_{}_v{target_vocab_size}", args.unit.as_str()),
      words_path: args.words.clone(),
      unit: args.unit,
      initial_alphabet: args.initial_alphabet,
      tie_break: args.tie_break,
      parallel_merge_min_occurs_in: args.parallel_merge_min_occurs_in,
      target_vocab_size,
      special_tokens: special_tokens.clone(),
      bucket_size: args.suite.bucket_size,
      bigram_cutoff_freq: args.bigram_cutoff_freq,
      expected_input_sha256: args.expected_input_sha256.clone(),
      expected_model_sha256: args.expected_model_sha256.clone(),
      rayon_threads,
    });
  }
  run_suite(&case_prefix, cases, args.suite)
}

fn run_suite(suite_name: &str, cases: Vec<CaseConfig>, mut options: SuiteOptions) -> Result<(), String> {
  validate_suite_options(&mut options)?;
  for case in &cases {
    case.validate()?;
  }
  let input_paths = cases
    .iter()
    .map(|case| case.words_path.clone())
    .collect::<Vec<_>>();
  if let Some(output) = options.output.as_deref() {
    validate_trainer_output_path(output, &input_paths)?;
  }
  let requests = build_requests(cases, &options);
  let outcomes = run_isolated_cases(&requests)?;
  let environment = environment_report();
  let generated_at_unix_seconds = now_seconds();
  let report = SuiteReport::new(suite_name.to_string(), generated_at_unix_seconds, environment, outcomes);
  let output = options
    .output
    .unwrap_or_else(|| default_report_path(suite_name, &report));
  validate_trainer_output_path(&output, &input_paths)?;
  write_json_atomic(&output, &report)?;
  print_summary(&output, &report);
  if report.gates.passed {
    Ok(())
  } else {
    Err(format!(
      "regression correctness gates failed; inspect {}",
      output.display()
    ))
  }
}

fn validate_trainer_output_path(output: &Path, inputs: &[PathBuf]) -> Result<(), String> {
  let output = resolve_path_for_comparison(output)?;
  for input in inputs {
    if output == resolve_path_for_comparison(input)? {
      return Err("report output cannot overwrite the word inventory".to_string());
    }
  }
  Ok(())
}

fn validate_suite_options(options: &mut SuiteOptions) -> Result<(), String> {
  if options.repeats == 0 {
    return Err("--repeats must be positive".to_string());
  }
  if options.bucket_size == 0 {
    return Err("--bucket-size must be positive".to_string());
  }
  if options.hot_pair_window_sizes.is_empty() {
    return Err("at least one hot-pair window size is required".to_string());
  }
  if options.hot_pair_window_sizes.contains(&0) {
    return Err("hot-pair window sizes must be positive".to_string());
  }
  options.hot_pair_window_sizes.sort_unstable();
  options.hot_pair_window_sizes.dedup();
  Ok(())
}

fn build_requests(cases: Vec<CaseConfig>, options: &SuiteOptions) -> Vec<CaseRequest> {
  let mut requests = Vec::new();
  for case in cases {
    let bounded = options
      .hot_pair_window_sizes
      .iter()
      .copied()
      .map(OccurrenceVariant::bounded)
      .collect::<Vec<_>>();
    for sample_index in 0..options.repeats {
      let mut variants = Vec::with_capacity(bounded.len() + 1);
      if sample_index % 2 == 0 {
        variants.push(OccurrenceVariant::exact());
        variants.extend(bounded.iter().cloned());
      } else {
        variants.extend(bounded.iter().cloned());
        variants.push(OccurrenceVariant::exact());
      }
      requests.extend(variants.into_iter().map(|variant| CaseRequest {
        case: case.clone(),
        variant,
        sample_index,
      }));
    }
  }
  requests
}



pub fn run_child(request_path: &Path, result_path: &Path) -> Result<bool, String> {
  run_protocol_child(request_path, result_path, runner::execute_case, |outcome| {
    outcome.measurement.is_some()
  })
}

fn run_isolated_cases(requests: &[CaseRequest]) -> Result<Vec<CaseOutcome>, String> {
  run_isolated_protocol(
    "case",
    requests,
    CaseRequest::id,
    validate_child_outcome,
    |request, message| CaseOutcome::failed(request, "child_process", message),
  )
}

fn validate_child_outcome(
  request: &CaseRequest,
  status: &ExitStatus,
  outcome: CaseOutcome,
) -> Result<CaseOutcome, String> {
  if outcome.request != *request || outcome.case_id != request.id() {
    return Err("child result does not match its request".to_string());
  }
  validate_outcome_shape(
    status,
    outcome.status == RunStatus::Completed,
    outcome.measurement.is_some(),
    outcome.error.is_some(),
  )?;
  Ok(outcome)
}

fn default_report_path(suite_name: &str, report: &SuiteReport) -> PathBuf {
  default_suite_report_path(suite_name, &report.environment)
}

fn print_summary(path: &Path, report: &SuiteReport) {
  println!("regression report: {}", path.display());
  for sample in &report.samples {
    match (&sample.status, &sample.measurement, &sample.error) {
      (RunStatus::Completed, Some(measurement), _) => println!(
        "  {:<45} train={:>9.3} ms train_peak_rss={:<12} model={}",
        sample.case_id,
        measurement.timing.core_training_ns as f64 / 1_000_000.0,
        format_bytes(measurement.memory.sampled_peak_during_training_bytes),
        &measurement.fingerprints.model_sha256[..12],
      ),
      (_, _, Some(error)) => println!("  {:<45} FAILED [{}] {}", sample.case_id, error.phase, error.message,),
      _ => println!("  {:<45} FAILED [invalid child report]", sample.case_id),
    }
  }
  println!("correctness gates passed: {}", report.gates.passed);
}
