use std::{
  collections::BTreeSet,
  fs,
  path::{Path, PathBuf},
  process::ExitStatus,
  time::Instant,
};

use clap::{Args as ClapArgs, ValueEnum};
use serde::{Deserialize, Serialize};
use unitoken::{
  bpe::Freq,
  pretokenizer::{BoundaryMode, ChunkHint, ChunkOptions, PreTokenizer},
};

use crate::common::{
  config::UnicodeBigramMixedBoundaryName as MixedBoundaryName,
  environment::{default_suite_report_path, environment_report, resolve_threads},
  fingerprint::{fingerprint_unicode_bigrams, fingerprint_word_counts, sha256_file, sha256_hex},
  process::{run_isolated_protocol, run_protocol_child, validate_outcome_shape, write_json_atomic},
  report::{EnvironmentReport, RunFailure, RunStatus},
  rss,
  util::{
    FileIdentity, duration_ms, duration_ns, file_stem, format_bytes, now_seconds, resolve_path_for_comparison,
    short_hash, throughput_mib, validate_sha256,
  },
};

pub const CONTRACT: &str = "unitoken_pretokenizer_regression_v1";
pub const SCHEMA_VERSION: u32 = 1;
const DEFAULT_UNICODE_BIGRAM_MIN_FREQ: Freq = 16;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
#[value(rename_all = "snake_case")]
pub enum BoundaryName {
  Auto,
  Eot,
  Line,
  Utf8,
}

impl BoundaryName {
  fn core(self) -> BoundaryMode {
    match self {
      Self::Auto => BoundaryMode::Auto,
      Self::Eot => BoundaryMode::Eot,
      Self::Line => BoundaryMode::Line,
      Self::Utf8 => BoundaryMode::Utf8,
    }
  }

  fn as_str(self) -> &'static str {
    match self {
      Self::Auto => "auto",
      Self::Eot => "eot",
      Self::Line => "line",
      Self::Utf8 => "utf8",
    }
  }
}

#[derive(Clone, Debug, ClapArgs)]
pub struct Args {
  /// Raw UTF-8 corpus file.
  #[arg(long)]
  pub text: PathBuf,
  /// Benchmark case name. Defaults to the corpus file stem.
  #[arg(long)]
  pub name: Option<String>,
  /// Approximate bytes per parallel file chunk.
  #[arg(long, default_value_t = 16 * 1024 * 1024)]
  pub chunk_size: u64,
  #[arg(long, value_enum, default_value = "auto")]
  pub boundary: BoundaryName,
  /// Enable the Unicode-bigram first pass and retain this many candidates plus ties.
  #[arg(long)]
  pub unicode_bigram_top_k: Option<usize>,
  #[arg(long, default_value_t = DEFAULT_UNICODE_BIGRAM_MIN_FREQ)]
  pub unicode_bigram_min_freq: Freq,
  #[arg(long, value_enum, default_value = "keep")]
  pub unicode_bigram_mixed_boundary: MixedBoundaryName,
  /// Write the selected bigrams as a canonical JSON array for codec replay.
  #[arg(long)]
  pub unicode_bigrams_output: Option<PathBuf>,
  /// Custom pretokenizer regex. Defaults to the library pattern.
  #[arg(long)]
  pub pat_str: Option<String>,
  /// Reserved special token. Repeat to configure more than one.
  #[arg(long = "special-token")]
  pub special_tokens: Vec<String>,
  /// File chunk boundary token. Defaults to the first special token or the library EOT.
  #[arg(long)]
  pub eot_token: Option<String>,
  #[arg(long, default_value_t = 1)]
  pub repeats: usize,
  #[arg(long)]
  pub rayon_threads: Option<usize>,
  #[arg(long)]
  pub expected_input_sha256: Option<String>,
  #[arg(long)]
  pub expected_bigrams_sha256: Option<String>,
  #[arg(long)]
  pub expected_inventory_sha256: Option<String>,
  #[arg(long)]
  pub output: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct UnicodeBigramConfig {
  pub top_k: usize,
  pub min_freq: Freq,
  pub mixed_boundary: MixedBoundaryName,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PretokenizerCaseConfig {
  pub name: String,
  pub text_path: PathBuf,
  pub chunk_size: u64,
  pub boundary: BoundaryName,
  pub unicode_bigrams: Option<UnicodeBigramConfig>,
  pub unicode_bigrams_output: Option<PathBuf>,
  pub pat_str: Option<String>,
  pub special_tokens: Vec<String>,
  pub eot_token: String,
  pub rayon_threads: usize,
  pub expected_input_sha256: Option<String>,
  pub expected_bigrams_sha256: Option<String>,
  pub expected_inventory_sha256: Option<String>,
}

impl PretokenizerCaseConfig {
  fn validate(&self) -> Result<(), String> {
    if self.name.trim().is_empty() {
      return Err("case name cannot be empty".to_string());
    }
    if self.chunk_size == 0 {
      return Err("chunk_size must be positive".to_string());
    }
    if self.rayon_threads == 0 {
      return Err("rayon_threads must be positive".to_string());
    }
    if self.eot_token.is_empty() {
      return Err("eot_token cannot be empty".to_string());
    }
    if self.special_tokens.iter().any(String::is_empty) {
      return Err("special tokens cannot be empty".to_string());
    }
    if self.special_tokens.iter().collect::<BTreeSet<_>>().len() != self.special_tokens.len() {
      return Err("special tokens cannot contain duplicates".to_string());
    }
    if let Some(config) = &self.unicode_bigrams {
      if config.top_k == 0 {
        return Err("unicode_bigram_top_k must be positive".to_string());
      }
      if config.min_freq <= 0 {
        return Err("unicode_bigram_min_freq must be positive".to_string());
      }
    } else {
      if self.expected_bigrams_sha256.is_some() {
        return Err("expected_bigrams_sha256 requires unicode_bigram_top_k".to_string());
      }
      if self.unicode_bigrams_output.is_some() {
        return Err("unicode_bigrams_output requires unicode_bigram_top_k".to_string());
      }
    }
    for (field, value) in [
      (
        "expected_input_sha256",
        self.expected_input_sha256.as_deref(),
      ),
      (
        "expected_bigrams_sha256",
        self.expected_bigrams_sha256.as_deref(),
      ),
      (
        "expected_inventory_sha256",
        self.expected_inventory_sha256.as_deref(),
      ),
    ] {
      validate_sha256(field, value)?;
    }
    Ok(())
  }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PretokenizerRequest {
  pub case: PretokenizerCaseConfig,
  pub sample_index: usize,
}

impl PretokenizerRequest {
  pub fn id(&self) -> String {
    format!("{}__r{}", self.case.name, self.sample_index)
  }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RawCorpusReport {
  pub path: PathBuf,
  pub bytes: u64,
  pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct UnicodeBigramReport {
  pub selected: usize,
  pub cutoff_freq: Option<Freq>,
  pub max_excluded_freq: Option<Freq>,
  pub boundary_is_valid: bool,
  pub artifact: Option<UnicodeBigramArtifactReport>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct UnicodeBigramArtifactReport {
  pub path: PathBuf,
  pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WordInventoryReport {
  pub unique_words: usize,
  pub weighted_occurrences: u64,
  pub singleton_words: usize,
  pub unique_word_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PretokenizerFingerprints {
  pub input_sha256: String,
  pub bigrams_sha256: Option<String>,
  pub inventory_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PretokenizerTiming {
  pub bigram_pass_ns: Option<u64>,
  pub bigram_artifact_ns: Option<u64>,
  pub configure_ns: u64,
  pub word_pass_ns: u64,
  pub fingerprint_ns: u64,
  pub input_hash_ns: u64,
  pub core_pretokenizer_ns: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PretokenizerMemory {
  pub current_rss_source: Option<String>,
  pub peak_rss_source: Option<String>,
  pub current_after_bigram_pass_bytes: Option<u64>,
  pub current_after_word_pass_bytes: Option<u64>,
  pub sampled_peak_during_bigram_pass_bytes: Option<u64>,
  pub sampled_peak_during_word_pass_bytes: Option<u64>,
  pub process_peak_rss_through_core_bytes: Option<u64>,
  pub rss_sample_interval_ms: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PretokenizerMeasurement {
  pub input: RawCorpusReport,
  pub actual_rayon_threads: usize,
  pub unicode_bigrams: Option<UnicodeBigramReport>,
  pub inventory: WordInventoryReport,
  pub fingerprints: PretokenizerFingerprints,
  pub timing: PretokenizerTiming,
  pub memory: PretokenizerMemory,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PretokenizerOutcome {
  pub case_id: String,
  pub request: PretokenizerRequest,
  pub status: RunStatus,
  pub measurement: Option<PretokenizerMeasurement>,
  pub error: Option<RunFailure>,
}

impl PretokenizerOutcome {
  fn completed(request: PretokenizerRequest, measurement: PretokenizerMeasurement) -> Self {
    Self {
      case_id: request.id(),
      request,
      status: RunStatus::Completed,
      measurement: Some(measurement),
      error: None,
    }
  }

  fn failed(
    request: PretokenizerRequest,
    phase: impl Into<String>,
    message: impl Into<String>,
  ) -> Self {
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
pub struct PretokenizerGates {
  pub all_runs_completed: bool,
  pub selections_valid: bool,
  pub samples_deterministic: Option<bool>,
  pub input_matches_expected: Option<bool>,
  pub bigrams_match_expected: Option<bool>,
  pub inventory_matches_expected: Option<bool>,
  pub passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct PretokenizerSuiteReport {
  pub schema_version: u32,
  pub contract: String,
  pub suite_name: String,
  pub generated_at_unix_seconds: u64,
  pub environment: EnvironmentReport,
  pub samples: Vec<PretokenizerOutcome>,
  pub gates: PretokenizerGates,
}

pub fn run(args: Args) -> Result<(), String> {
  let environment = environment_report();
  let repeats = args.repeats;
  if repeats == 0 {
    return Err("--repeats must be positive".to_string());
  }
  if args.unicode_bigram_min_freq <= 0 {
    return Err("--unicode-bigram-min-freq must be positive".to_string());
  }
  if args.unicode_bigram_top_k.is_none() {
    if args.unicode_bigram_min_freq != DEFAULT_UNICODE_BIGRAM_MIN_FREQ {
      return Err("--unicode-bigram-min-freq requires --unicode-bigram-top-k".to_string());
    }
    if args.unicode_bigram_mixed_boundary != MixedBoundaryName::Keep {
      return Err("--unicode-bigram-mixed-boundary requires --unicode-bigram-top-k".to_string());
    }
    if args.unicode_bigrams_output.is_some() {
      return Err("--unicode-bigrams-output requires --unicode-bigram-top-k".to_string());
    }
  }
  let rayon_threads = resolve_threads(args.rayon_threads)?;
  let name = args
    .name
    .unwrap_or_else(|| file_stem(&args.text, "pretokenizer"));
  let special_tokens = if args.special_tokens.is_empty() {
    vec![unitoken::pretokenizer::DEFAULT_EOT.to_string()]
  } else {
    args.special_tokens
  };
  let eot_token = args
    .eot_token
    .or_else(|| special_tokens.first().cloned())
    .unwrap_or_else(|| unitoken::pretokenizer::DEFAULT_EOT.to_string());
  let case = PretokenizerCaseConfig {
    name: name.clone(),
    text_path: args.text,
    chunk_size: args.chunk_size,
    boundary: args.boundary,
    unicode_bigrams: args.unicode_bigram_top_k.map(|top_k| UnicodeBigramConfig {
      top_k,
      min_freq: args.unicode_bigram_min_freq,
      mixed_boundary: args.unicode_bigram_mixed_boundary,
    }),
    unicode_bigrams_output: args.unicode_bigrams_output,
    pat_str: args.pat_str,
    special_tokens,
    eot_token,
    rayon_threads,
    expected_input_sha256: args.expected_input_sha256,
    expected_bigrams_sha256: args.expected_bigrams_sha256,
    expected_inventory_sha256: args.expected_inventory_sha256,
  };
  case.validate()?;
  if let Some(report_output) = args.output.as_deref() {
    validate_output_paths(
      report_output,
      &case.text_path,
      case.unicode_bigrams_output.as_deref(),
    )?;
  }
  let requests = (0..repeats)
    .map(|sample_index| PretokenizerRequest {
      case: case.clone(),
      sample_index,
    })
    .collect::<Vec<_>>();
  let outcomes = run_isolated_protocol(
    "pretokenizer-case",
    &requests,
    PretokenizerRequest::id,
    validate_child_outcome,
    |request, message| PretokenizerOutcome::failed(request, "child_process", message),
  )?;
  let gates = evaluate_gates(&outcomes);
  let report_name = pretokenizer_report_name(&case, repeats, &outcomes)?;
  let report = PretokenizerSuiteReport {
    schema_version: SCHEMA_VERSION,
    contract: CONTRACT.to_string(),
    suite_name: name.clone(),
    generated_at_unix_seconds: now_seconds(),
    environment,
    samples: outcomes,
    gates,
  };
  let output = args
    .output
    .unwrap_or_else(|| default_suite_report_path(&report_name, &report.environment));
  validate_output_paths(
    &output,
    &case.text_path,
    case.unicode_bigrams_output.as_deref(),
  )?;
  write_json_atomic(&output, &report)?;
  print_summary(&output, &report);
  if report.gates.passed {
    Ok(())
  } else {
    Err(format!(
      "pretokenizer correctness gates failed; inspect {}",
      output.display()
    ))
  }
}

pub fn run_child(request: &Path, result: &Path) -> Result<bool, String> {
  run_protocol_child(request, result, execute, |outcome| {
    outcome.measurement.is_some()
  })
}

fn execute(request: PretokenizerRequest) -> PretokenizerOutcome {
  if let Err(error) = request.case.validate() {
    return PretokenizerOutcome::failed(request, "configuration", error);
  }
  let pool = match rayon::ThreadPoolBuilder::new()
    .num_threads(request.case.rayon_threads)
    .build()
  {
    Ok(pool) => pool,
    Err(error) => return PretokenizerOutcome::failed(request, "rayon_pool", error.to_string()),
  };
  match pool.install(|| execute_inner(&request)) {
    Ok(measurement) => PretokenizerOutcome::completed(request, measurement),
    Err((phase, message)) => PretokenizerOutcome::failed(request, phase, message),
  }
}

fn execute_inner(
  request: &PretokenizerRequest,
) -> Result<PretokenizerMeasurement, (&'static str, String)> {
  let config = &request.case;
  let input_identity = FileIdentity::capture(&config.text_path)
    .map_err(|error| ("input_metadata", error))?;
  let options = ChunkOptions {
    hint: ChunkHint::Size(config.chunk_size),
    boundary: config.boundary.core(),
  };
  let started = Instant::now();
  let mut base = PreTokenizer::try_new(
    &config.special_tokens,
    Some(&config.eot_token),
    config.pat_str.as_deref(),
  )
  .map_err(|error| ("configure", error.to_string()))?;
  base.metrics = false;
  let mut configure_ns = duration_ns(started.elapsed());

  let mut bigram_pass_ns = None;
  let mut current_after_bigram_pass_bytes = None;
  let mut sampled_peak_during_bigram_pass_bytes = None;
  let mut bigram_report = None;
  let mut selection_metadata = None;
  let mut bigram_artifact_ns = None;
  let frozen = if let Some(bigram_config) = &config.unicode_bigrams {
    let sampler = rss::RssSampler::start();
    let started = Instant::now();
    let selection = base
      .build_unicode_bigram_selection_from_file_with_options(
        &config.text_path,
        options,
        bigram_config.top_k,
        bigram_config.min_freq,
      )
      .map_err(|error| ("bigram_pass", error.to_string()))?;
    bigram_pass_ns = Some(duration_ns(started.elapsed()));
    current_after_bigram_pass_bytes = rss::current_rss_bytes();
    sampled_peak_during_bigram_pass_bytes = sampler.map(rss::RssSampler::finish);
    let boundary_is_valid = unicode_bigram_selection_is_valid(
      selection.bigrams.len(),
      selection.cutoff_freq,
      selection.max_excluded_freq,
    );
    let artifact = if let Some(path) = &config.unicode_bigrams_output {
      let started = Instant::now();
      let artifact = write_unicode_bigram_artifact(
        path,
        &config.text_path,
        &selection.bigrams,
      )
      .map_err(|error| ("bigram_artifact", error))?;
      bigram_artifact_ns = Some(duration_ns(started.elapsed()));
      Some(artifact)
    } else {
      None
    };
    bigram_report = Some(UnicodeBigramReport {
      selected: selection.bigrams.len(),
      cutoff_freq: selection.cutoff_freq,
      max_excluded_freq: selection.max_excluded_freq,
      boundary_is_valid,
      artifact,
    });
    selection_metadata = Some((selection.cutoff_freq, selection.max_excluded_freq));
    let started = Instant::now();
    let frozen = base
      .with_unicode_bigrams(selection.bigrams)
      .with_unicode_bigram_mixed_boundary(bigram_config.mixed_boundary.core());
    configure_ns = configure_ns.saturating_add(duration_ns(started.elapsed()));
    frozen
  } else {
    base
  };

  let word_sampler = rss::RssSampler::start();
  let started = Instant::now();
  let words = frozen
    .get_words_from_file_with_options(&config.text_path, options)
    .map_err(|error| ("word_pass", error.to_string()))?;
  let word_pass_ns = duration_ns(started.elapsed());
  let current_after_word_pass_bytes = rss::current_rss_bytes();
  let sampled_peak_during_word_pass_bytes = word_sampler.map(rss::RssSampler::finish);
  let process_peak_rss_through_core_bytes = rss::process_peak_rss_bytes();

  let mut weighted_occurrences = 0u64;
  let mut singleton_words = 0usize;
  let mut unique_word_bytes = 0u64;
  for (word, frequency) in &words {
    if *frequency <= 0 {
      return Err((
        "inventory_validate",
        format!("word {word:?} has non-positive frequency {frequency}"),
      ));
    }
    weighted_occurrences = weighted_occurrences
      .checked_add(*frequency as u64)
      .ok_or_else(|| {
        (
          "inventory_validate",
          "weighted occurrence count overflowed".to_string(),
        )
      })?;
    singleton_words += usize::from(*frequency == 1);
    unique_word_bytes = unique_word_bytes
      .checked_add(word.len() as u64)
      .ok_or_else(|| {
        (
          "inventory_validate",
          "unique word byte count overflowed".to_string(),
        )
      })?;
  }

  let started = Instant::now();
  let bigrams_sha256 = frozen.unicode_bigrams.as_ref().map(|bigrams| {
    let (cutoff, excluded) = selection_metadata.unwrap_or((None, None));
    fingerprint_unicode_bigrams(bigrams, cutoff, excluded)
  });
  let inventory_sha256 = fingerprint_word_counts(&words);
  let fingerprint_ns = duration_ns(started.elapsed());

  let started = Instant::now();
  let input_sha256 = sha256_file(&config.text_path).map_err(|error| ("input_hash", error))?;
  let input_hash_ns = duration_ns(started.elapsed());
  input_identity
    .ensure_unchanged(&config.text_path)
    .map_err(|error| ("input_changed", error))?;

  Ok(PretokenizerMeasurement {
    input: RawCorpusReport {
      path: fs::canonicalize(&config.text_path).unwrap_or_else(|_| config.text_path.clone()),
      bytes: input_identity.len(),
      sha256: input_sha256.clone(),
    },
    actual_rayon_threads: rayon::current_num_threads(),
    unicode_bigrams: bigram_report,
    inventory: WordInventoryReport {
      unique_words: words.len(),
      weighted_occurrences,
      singleton_words,
      unique_word_bytes,
    },
    fingerprints: PretokenizerFingerprints {
      input_sha256,
      bigrams_sha256,
      inventory_sha256,
    },
    timing: PretokenizerTiming {
      bigram_pass_ns,
      bigram_artifact_ns,
      configure_ns,
      word_pass_ns,
      fingerprint_ns,
      input_hash_ns,
      core_pretokenizer_ns: bigram_pass_ns
        .unwrap_or_default()
        .saturating_add(word_pass_ns),
    },
    memory: PretokenizerMemory {
      current_rss_source: rss::current_rss_source().map(str::to_string),
      peak_rss_source: rss::peak_rss_source().map(str::to_string),
      current_after_bigram_pass_bytes,
      current_after_word_pass_bytes,
      sampled_peak_during_bigram_pass_bytes,
      sampled_peak_during_word_pass_bytes,
      process_peak_rss_through_core_bytes,
      rss_sample_interval_ms: (sampled_peak_during_bigram_pass_bytes.is_some()
        || sampled_peak_during_word_pass_bytes.is_some())
      .then_some(rss::SAMPLE_INTERVAL.as_millis() as u64),
    },
  })
}

fn validate_child_outcome(
  request: &PretokenizerRequest,
  status: &ExitStatus,
  outcome: PretokenizerOutcome,
) -> Result<PretokenizerOutcome, String> {
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

fn evaluate_gates(samples: &[PretokenizerOutcome]) -> PretokenizerGates {
  let completed = samples
    .iter()
    .filter_map(|sample| sample.measurement.as_ref())
    .collect::<Vec<_>>();
  let all_runs_completed = !samples.is_empty()
    && completed.len() == samples.len()
    && samples
      .iter()
      .all(|sample| sample.status == RunStatus::Completed);
  let selections_valid = all_runs_completed
    && completed.iter().all(|measurement| {
      measurement
        .unicode_bigrams
        .as_ref()
        .is_none_or(|selection| selection.boundary_is_valid)
    });
  let samples_deterministic = if !all_runs_completed || completed.len() < 2 {
    None
  } else {
    let first = &completed[0].fingerprints;
    Some(
      completed
        .iter()
        .all(|measurement| measurement.fingerprints == *first),
    )
  };
  let input_matches_expected = expected_gate(
    samples,
    |case| case.expected_input_sha256.as_deref(),
    |m| Some(m.fingerprints.input_sha256.as_str()),
  );
  let bigrams_match_expected = expected_gate(
    samples,
    |case| case.expected_bigrams_sha256.as_deref(),
    |m| m.fingerprints.bigrams_sha256.as_deref(),
  );
  let inventory_matches_expected = expected_gate(
    samples,
    |case| case.expected_inventory_sha256.as_deref(),
    |m| Some(m.fingerprints.inventory_sha256.as_str()),
  );
  let passed = all_runs_completed
    && selections_valid
    && samples_deterministic != Some(false)
    && input_matches_expected != Some(false)
    && bigrams_match_expected != Some(false)
    && inventory_matches_expected != Some(false);
  PretokenizerGates {
    all_runs_completed,
    selections_valid,
    samples_deterministic,
    input_matches_expected,
    bigrams_match_expected,
    inventory_matches_expected,
    passed,
  }
}

fn expected_gate<'a>(
  samples: &'a [PretokenizerOutcome],
  expected: impl Fn(&'a PretokenizerCaseConfig) -> Option<&'a str>,
  actual: impl Fn(&'a PretokenizerMeasurement) -> Option<&'a str>,
) -> Option<bool> {
  let configured = samples
    .iter()
    .any(|sample| expected(&sample.request.case).is_some());
  configured.then(|| {
    samples.iter().all(|sample| {
      let Some(expected) = expected(&sample.request.case) else {
        return true;
      };
      sample
        .measurement
        .as_ref()
        .and_then(&actual)
        .is_some_and(|actual| actual.eq_ignore_ascii_case(expected))
    })
  })
}

fn pretokenizer_report_name(
  case: &PretokenizerCaseConfig,
  repeats: usize,
  samples: &[PretokenizerOutcome],
) -> Result<String, String> {
  let config_bytes = serde_json::to_vec(case)
    .map_err(|error| format!("cannot fingerprint pretokenizer configuration: {error}"))?;
  let config_sha256 = sha256_hex(&config_bytes);
  let bigram_key = case
    .unicode_bigrams
    .as_ref()
    .map(|config| {
      format!(
        "bigram-k{}-min{}-{}",
        config.top_k,
        config.min_freq,
        config.mixed_boundary.as_str(),
      )
    })
    .unwrap_or_else(|| "bigram-off".to_string());
  let input_key = samples
    .iter()
    .find_map(|sample| sample.measurement.as_ref())
    .map(|measurement| &measurement.fingerprints.input_sha256[..8])
    .unwrap_or("failed");
  Ok(format!(
    "pretokenizer.{}.{}.{}.chunk{}.t{}.r{}.{}.{}",
    case.name,
    case.boundary.as_str(),
    bigram_key,
    case.chunk_size,
    case.rayon_threads,
    repeats,
    &config_sha256[..8],
    input_key,
  ))
}

fn write_unicode_bigram_artifact(
  output: &Path,
  input: &Path,
  bigrams: &ahash::AHashSet<(char, char)>,
) -> Result<UnicodeBigramArtifactReport, String> {
  let resolved_output = resolve_path_for_comparison(output)?;
  let resolved_input = fs::canonicalize(input)
    .map_err(|error| format!("cannot resolve {}: {error}", input.display()))?;
  if resolved_output == resolved_input {
    return Err("unicode bigram output cannot overwrite the input corpus".to_string());
  }
  let mut sorted = bigrams.iter().copied().collect::<Vec<_>>();
  sorted.sort_unstable();
  let strings = sorted
    .into_iter()
    .map(|(left, right)| String::from_iter([left, right]))
    .collect::<Vec<_>>();
  write_json_atomic(output, &strings)?;
  let sha256 = sha256_file(output)?;
  Ok(UnicodeBigramArtifactReport {
    path: fs::canonicalize(output).unwrap_or(resolved_output),
    sha256,
  })
}

fn validate_output_paths(
  report: &Path,
  input: &Path,
  unicode_bigrams: Option<&Path>,
) -> Result<(), String> {
  let report = resolve_path_for_comparison(report)?;
  let input = resolve_path_for_comparison(input)?;
  if report == input {
    return Err("report output cannot overwrite the input corpus".to_string());
  }
  if let Some(unicode_bigrams) = unicode_bigrams {
    let unicode_bigrams = resolve_path_for_comparison(unicode_bigrams)?;
    if report == unicode_bigrams {
      return Err("report output and unicode bigram output must be different files".to_string());
    }
  }
  Ok(())
}

fn print_summary(path: &Path, report: &PretokenizerSuiteReport) {
  println!("pretokenizer regression report: {}", path.display());
  for sample in &report.samples {
    if let Some(measurement) = &sample.measurement {
      let bytes = measurement.input.bytes;
      println!(
        "  {} input={:.1} MiB core={:.3} ms process_peak={}",
        sample.case_id,
        bytes as f64 / 1024.0 / 1024.0,
        duration_ms(measurement.timing.core_pretokenizer_ns),
        format_bytes(measurement.memory.process_peak_rss_through_core_bytes),
      );
      match (
        measurement.timing.bigram_pass_ns,
        measurement.unicode_bigrams.as_ref(),
      ) {
        (Some(elapsed_ns), Some(selection)) => {
          println!(
            "    bigram {:>9.3} ms {:>8.1} MiB/s peak={:<10} selected={} cutoff={} excluded={}",
            duration_ms(elapsed_ns),
            throughput_mib(bytes, elapsed_ns),
            format_bytes(measurement.memory.sampled_peak_during_bigram_pass_bytes),
            selection.selected,
            format_freq(selection.cutoff_freq),
            format_freq(selection.max_excluded_freq),
          );
          if let Some(artifact) = &selection.artifact {
            println!(
              "      artifact={} hash={} write_and_hash={:.3} ms",
              artifact.path.display(),
              short_hash(&artifact.sha256),
              duration_ms(measurement.timing.bigram_artifact_ns.unwrap_or_default()),
            );
          }
        }
        _ => println!("    bigram disabled"),
      }
      println!(
        "    words  {:>9.3} ms {:>8.1} MiB/s peak={:<10} unique={} occurrences={} singletons={} unique_bytes={}",
        duration_ms(measurement.timing.word_pass_ns),
        throughput_mib(bytes, measurement.timing.word_pass_ns),
        format_bytes(measurement.memory.sampled_peak_during_word_pass_bytes),
        measurement.inventory.unique_words,
        measurement.inventory.weighted_occurrences,
        measurement.inventory.singleton_words,
        measurement.inventory.unique_word_bytes,
      );
      println!(
        "    aux configure={:.3} ms fingerprint={:.3} ms input_hash={:.3} ms",
        duration_ms(measurement.timing.configure_ns),
        duration_ms(measurement.timing.fingerprint_ns),
        duration_ms(measurement.timing.input_hash_ns),
      );
      println!(
        "    hashes input={} bigrams={} inventory={}",
        short_hash(&measurement.fingerprints.input_sha256),
        measurement
          .fingerprints
          .bigrams_sha256
          .as_deref()
          .map(short_hash)
          .unwrap_or("disabled"),
        short_hash(&measurement.fingerprints.inventory_sha256),
      );
    } else if let Some(error) = &sample.error {
      println!(
        "  {:<36} FAILED [{}] {}",
        sample.case_id, error.phase, error.message
      );
    }
  }
  println!(
    "gates completed={} selections={} deterministic={:?} input_expected={:?} bigrams_expected={:?} inventory_expected={:?}",
    report.gates.all_runs_completed,
    report.gates.selections_valid,
    report.gates.samples_deterministic,
    report.gates.input_matches_expected,
    report.gates.bigrams_match_expected,
    report.gates.inventory_matches_expected,
  );
  println!("correctness gates passed: {}", report.gates.passed);
}

fn unicode_bigram_selection_is_valid(
  selected: usize,
  cutoff_freq: Option<Freq>,
  max_excluded_freq: Option<Freq>,
) -> bool {
  let selection_shape_is_valid = (selected == 0) == cutoff_freq.is_none();
  let frequencies_are_positive = cutoff_freq.is_none_or(|frequency| frequency > 0)
    && max_excluded_freq.is_none_or(|frequency| frequency > 0);
  let boundary_is_ordered = match (cutoff_freq, max_excluded_freq) {
    (Some(cutoff), Some(excluded)) => excluded < cutoff,
    _ => true,
  };
  selection_shape_is_valid && frequencies_are_positive && boundary_is_ordered
}

fn format_freq(frequency: Option<Freq>) -> String {
  frequency
    .map(|frequency| frequency.to_string())
    .unwrap_or_else(|| "none".to_string())
}
