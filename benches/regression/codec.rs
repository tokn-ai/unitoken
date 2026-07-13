use std::{
  collections::BTreeSet,
  fs,
  io::{BufReader, BufWriter, Read, Write},
  path::{Path, PathBuf},
  process::ExitStatus,
  time::Instant,
};

use clap::{Args as ClapArgs, ValueEnum};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use unitoken::{
  bpe::{BpeEncoder, Character, Idx, encoder::BpeBuilder},
  pretokenizer::{PreTokenizer, UnicodeBigramMixedBoundary, parse_unicode_bigrams},
  spec::{gpt2::Gpt2Spec, unitoken::UnitokenSpec},
  traits::{CanEncode, Encode},
};

use crate::common::{
  config::{UnicodeBigramMixedBoundaryName, Unit},
  environment::{default_suite_report_path, environment_report, resolve_threads},
  fingerprint::{fingerprint_token_ids, sha256_file, sha256_hex, to_hex},
  process::{TemporaryDirectory, run_isolated_protocol, run_protocol_child, validate_outcome_shape, write_json_atomic},
  report::{EnvironmentReport, RunFailure, RunStatus},
  rss,
  util::{
    FileIdentity, duration_ms, duration_ns, file_stem, format_bytes, now_seconds, resolve_path_for_comparison,
    short_hash, throughput_mib, validate_sha256,
  },
};

pub const CONTRACT: &str = "unitoken_codec_regression_v1";
pub const SCHEMA_VERSION: u32 = 1;
pub(crate) const DEFAULT_CHUNKS: usize = 1024;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
#[value(rename_all = "snake_case")]
pub enum ModelFormat {
  Gpt2,
  Unitoken,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum CodecPhase {
  Encode,
  Decode,
}

impl CodecPhase {
  fn label(self) -> &'static str {
    match self {
      Self::Encode => "encode",
      Self::Decode => "decode",
    }
  }
}

#[derive(Clone, Debug, ClapArgs)]
pub struct Args {
  /// Raw UTF-8 corpus file to encode and round-trip.
  #[arg(long)]
  pub text: PathBuf,
  #[arg(long)]
  pub vocab: PathBuf,
  #[arg(long)]
  pub merges: PathBuf,
  #[arg(long, value_enum)]
  pub unit: Unit,
  #[arg(long, value_enum)]
  pub format: ModelFormat,
  /// Benchmark case name. Defaults to the corpus file stem.
  #[arg(long)]
  pub name: Option<String>,
  /// Requested parallel file chunks; aligned boundaries may deduplicate.
  #[arg(long, default_value_t = DEFAULT_CHUNKS)]
  pub chunks: usize,
  /// Custom pretokenizer regex. Defaults to the library pattern.
  #[arg(long)]
  pub pat_str: Option<String>,
  /// JSON array of selected Unicode bigram strings required by a premerged model.
  #[arg(long)]
  pub unicode_bigrams: Option<PathBuf>,
  #[arg(long, value_enum, default_value = "keep")]
  pub unicode_bigram_mixed_boundary: UnicodeBigramMixedBoundaryName,
  /// Reserved special token. Repeat to configure more than one.
  #[arg(long = "special-token")]
  pub special_tokens: Vec<String>,
  #[arg(long, default_value_t = 1)]
  pub repeats: usize,
  #[arg(long)]
  pub rayon_threads: Option<usize>,
  #[arg(long)]
  pub expected_input_sha256: Option<String>,
  #[arg(long)]
  pub expected_vocab_sha256: Option<String>,
  #[arg(long)]
  pub expected_merges_sha256: Option<String>,
  #[arg(long)]
  pub expected_token_count: Option<usize>,
  #[arg(long)]
  pub expected_token_sha256: Option<String>,
  #[arg(long)]
  pub output: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CodecCaseConfig {
  pub name: String,
  pub text_path: PathBuf,
  pub vocab_path: PathBuf,
  pub merges_path: PathBuf,
  pub unit: Unit,
  pub format: ModelFormat,
  pub requested_chunks: usize,
  pub pat_str: Option<String>,
  pub unicode_bigrams_path: Option<PathBuf>,
  pub unicode_bigram_mixed_boundary: UnicodeBigramMixedBoundaryName,
  pub special_tokens: Vec<String>,
  pub rayon_threads: usize,
  pub expected_input_sha256: Option<String>,
  pub expected_vocab_sha256: Option<String>,
  pub expected_merges_sha256: Option<String>,
  pub expected_token_count: Option<usize>,
  pub expected_token_sha256: Option<String>,
}

impl CodecCaseConfig {
  pub(crate) fn validate(&self) -> Result<(), String> {
    if self.name.trim().is_empty() {
      return Err("case name cannot be empty".to_string());
    }
    if self.requested_chunks == 0 {
      return Err("chunks must be positive".to_string());
    }
    if self.rayon_threads == 0 {
      return Err("rayon_threads must be positive".to_string());
    }
    if self.special_tokens.is_empty() {
      return Err("at least one special token is required".to_string());
    }
    if self.special_tokens.iter().any(String::is_empty) {
      return Err("special tokens cannot be empty".to_string());
    }
    if self.special_tokens.iter().collect::<BTreeSet<_>>().len() != self.special_tokens.len() {
      return Err("special tokens cannot contain duplicates".to_string());
    }
    PreTokenizer::try_new(
      &self.special_tokens,
      self.special_tokens.first().map(String::as_str),
      self.pat_str.as_deref(),
    )
    .map_err(|error| format!("invalid pretokenizer configuration: {error}"))?;
    match (self.format, self.unit) {
      (ModelFormat::Gpt2, Unit::Byte)
      | (ModelFormat::Unitoken, Unit::Byte | Unit::Unicode) => {}
      (ModelFormat::Gpt2, Unit::Unicode) => {
        return Err("gpt2 format is not compatible with unicode units".to_string());
      }
    }
    if self.unicode_bigrams_path.is_some() && self.unit != Unit::Unicode {
      return Err("unicode_bigrams is only compatible with unicode units".to_string());
    }
    if self.unicode_bigram_mixed_boundary != UnicodeBigramMixedBoundaryName::Keep
      && self.unit != Unit::Unicode
    {
      return Err("unicode_bigram_mixed_boundary is only configurable with unicode units".to_string());
    }
    for (field, value) in [
      ("expected_input_sha256", self.expected_input_sha256.as_deref()),
      ("expected_vocab_sha256", self.expected_vocab_sha256.as_deref()),
      ("expected_merges_sha256", self.expected_merges_sha256.as_deref()),
      ("expected_token_sha256", self.expected_token_sha256.as_deref()),
    ] {
      validate_sha256(field, value)?;
    }
    Ok(())
  }
}

struct ModelFileIdentities {
  vocab: FileIdentity,
  merges: FileIdentity,
  unicode_bigrams: Option<FileIdentity>,
}

impl ModelFileIdentities {
  fn capture(config: &CodecCaseConfig) -> Result<Self, String> {
    Ok(Self {
      vocab: FileIdentity::capture(&config.vocab_path)?,
      merges: FileIdentity::capture(&config.merges_path)?,
      unicode_bigrams: config
        .unicode_bigrams_path
        .as_deref()
        .map(FileIdentity::capture)
        .transpose()?,
    })
  }

  fn ensure_unchanged(&self, config: &CodecCaseConfig) -> Result<(), String> {
    self.vocab.ensure_unchanged(&config.vocab_path)?;
    self.merges.ensure_unchanged(&config.merges_path)?;
    if let (Some(identity), Some(path)) = (
      &self.unicode_bigrams,
      config.unicode_bigrams_path.as_deref(),
    ) {
      identity.ensure_unchanged(path)?;
    }
    Ok(())
  }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct CodecRequest {
  case: CodecCaseConfig,
  sample_index: usize,
  phase: CodecPhase,
  token_path: PathBuf,
}

impl CodecRequest {
  fn id(&self) -> String {
    format!("{}__{}__r{}", self.case.name, self.phase.label(), self.sample_index)
  }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CodecInputReport {
  pub text_bytes: u64,
  pub input_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CodecModelReport {
  pub vocab_size: usize,
  pub merge_count: usize,
  pub special_token_count: usize,
  pub vocab_sha256: String,
  pub merges_sha256: String,
  pub unicode_bigrams_file_sha256: Option<String>,
  pub unicode_bigrams_semantic_sha256: Option<String>,
  pub unicode_bigram_count: Option<usize>,
  pub unicode_bigram_mixed_boundary: UnicodeBigramMixedBoundaryName,
  pub resolved_pat_str: String,
  pub end_of_text: String,
  pub encoder_config_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CodecTiming {
  pub model_load_ns: u64,
  pub model_fingerprint_ns: u64,
  pub input_hash_ns: Option<u64>,
  pub token_load_ns: Option<u64>,
  pub encode_ns: Option<u64>,
  pub decode_ns: Option<u64>,
  pub fingerprint_ns: u64,
  pub artifact_write_ns: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CodecMemory {
  pub current_rss_source: Option<String>,
  pub peak_rss_source: Option<String>,
  pub current_after_model_load_bytes: Option<u64>,
  pub current_before_phase_bytes: Option<u64>,
  pub current_after_phase_bytes: Option<u64>,
  pub sampled_peak_during_phase_bytes: Option<u64>,
  pub process_peak_rss_through_phase_bytes: Option<u64>,
  pub rss_sample_interval_ms: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EncodeMeasurement {
  pub input: CodecInputReport,
  pub model: CodecModelReport,
  pub actual_rayon_threads: usize,
  pub token_count: usize,
  pub token_sha256: String,
  pub timing: CodecTiming,
  pub memory: CodecMemory,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DecodeMeasurement {
  pub model: CodecModelReport,
  pub token_count: usize,
  pub token_sha256: String,
  pub decoded_bytes: u64,
  pub decoded_sha256: String,
  pub timing: CodecTiming,
  pub memory: CodecMemory,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
enum CodecMeasurement {
  Encode(EncodeMeasurement),
  Decode(DecodeMeasurement),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CodecStageOutcome {
  case_id: String,
  request: CodecRequest,
  status: RunStatus,
  measurement: Option<CodecMeasurement>,
  error: Option<RunFailure>,
}

impl CodecStageOutcome {
  fn completed(request: CodecRequest, measurement: CodecMeasurement) -> Self {
    Self {
      case_id: request.id(),
      request,
      status: RunStatus::Completed,
      measurement: Some(measurement),
      error: None,
    }
  }

  fn failed(request: CodecRequest, phase: impl Into<String>, message: impl Into<String>) -> Self {
    Self {
      case_id: request.id(),
      request,
      status: RunStatus::Failed,
      measurement: None,
      error: Some(RunFailure { phase: phase.into(), message: message.into() }),
    }
  }
}

#[derive(Clone, Debug, Serialize)]
pub struct CodecSample {
  pub case_id: String,
  pub sample_index: usize,
  pub encode: Option<EncodeMeasurement>,
  pub decode: Option<DecodeMeasurement>,
  pub errors: Vec<RunFailure>,
  pub model_consistent: bool,
  pub roundtrip_valid: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct CodecGates {
  pub all_runs_completed: bool,
  pub models_consistent: bool,
  pub roundtrips_valid: bool,
  pub samples_deterministic: Option<bool>,
  pub input_matches_expected: Option<bool>,
  pub model_files_match_expected: Option<bool>,
  pub tokens_match_expected: Option<bool>,
  pub passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct CodecSuiteReport {
  pub schema_version: u32,
  pub contract: String,
  pub suite_name: String,
  pub generated_at_unix_seconds: u64,
  pub environment: EnvironmentReport,
  pub config: CodecCaseConfig,
  pub samples: Vec<CodecSample>,
  pub gates: CodecGates,
}

pub fn run(args: Args) -> Result<(), String> {
  let repeats = args.repeats;
  if repeats == 0 {
    return Err("--repeats must be positive".to_string());
  }
  let rayon_threads = resolve_threads(args.rayon_threads)?;
  let name = args.name.unwrap_or_else(|| file_stem(&args.text, "codec"));
  let special_tokens = if args.special_tokens.is_empty() {
    vec![unitoken::pretokenizer::DEFAULT_EOT.to_string()]
  } else {
    args.special_tokens
  };
  let config = CodecCaseConfig {
    name: name.clone(),
    text_path: args.text,
    vocab_path: args.vocab,
    merges_path: args.merges,
    unit: args.unit,
    format: args.format,
    requested_chunks: args.chunks,
    pat_str: args.pat_str,
    unicode_bigrams_path: args.unicode_bigrams,
    unicode_bigram_mixed_boundary: args.unicode_bigram_mixed_boundary,
    special_tokens,
    rayon_threads,
    expected_input_sha256: args.expected_input_sha256,
    expected_vocab_sha256: args.expected_vocab_sha256,
    expected_merges_sha256: args.expected_merges_sha256,
    expected_token_count: args.expected_token_count,
    expected_token_sha256: args.expected_token_sha256,
  };
  run_config(config, repeats, args.output)
}

pub(crate) fn run_config(
  config: CodecCaseConfig,
  repeats: usize,
  output: Option<PathBuf>,
) -> Result<(), String> {
  let environment = environment_report();
  if repeats == 0 {
    return Err("repeats must be positive".to_string());
  }
  config.validate()?;
  if let Some(report_output) = output.as_deref() {
    validate_codec_output_path(report_output, &config)?;
  }
  let artifact_dir = TemporaryDirectory::create("unitoken-codec")?;
  let mut samples = Vec::with_capacity(repeats);
  for sample_index in 0..repeats {
    let token_path = artifact_dir.path().join(format!("tokens-{sample_index}.bin"));
    let encode_request = CodecRequest {
      case: config.clone(),
      sample_index,
      phase: CodecPhase::Encode,
      token_path: token_path.clone(),
    };
    let encode_outcome = run_stage(&encode_request)?;
    let decode_outcome = if encode_outcome.status == RunStatus::Completed {
      let request = CodecRequest {
        case: config.clone(),
        sample_index,
        phase: CodecPhase::Decode,
        token_path: token_path.clone(),
      };
      Some(run_stage(&request)?)
    } else {
      None
    };
    samples.push(combine_sample(sample_index, encode_outcome, decode_outcome));
    let _ = fs::remove_file(token_path);
  }
  let gates = evaluate_gates(&config, &samples);
  let report_name = codec_report_name(&config, repeats, &samples)?;
  let report = CodecSuiteReport {
    schema_version: SCHEMA_VERSION,
    contract: CONTRACT.to_string(),
    suite_name: config.name.clone(),
    generated_at_unix_seconds: now_seconds(),
    environment,
    config,
    samples,
    gates,
  };
  let output = output.unwrap_or_else(|| default_suite_report_path(&report_name, &report.environment));
  validate_codec_output_path(&output, &report.config)?;
  write_json_atomic(&output, &report)?;
  print_summary(&output, &report);
  if report.gates.passed {
    Ok(())
  } else {
    Err(format!("codec correctness gates failed; inspect {}", output.display()))
  }
}

pub fn run_child(request: &Path, result: &Path) -> Result<bool, String> {
  run_protocol_child(request, result, execute, |outcome| outcome.measurement.is_some())
}

fn run_stage(request: &CodecRequest) -> Result<CodecStageOutcome, String> {
  let outcomes = run_isolated_protocol(
    "codec-case",
    std::slice::from_ref(request),
    CodecRequest::id,
    validate_child_outcome,
    |request, message| CodecStageOutcome::failed(request, "child_process", message),
  )?;
  outcomes.into_iter().next().ok_or_else(|| "codec child produced no outcome".to_string())
}

fn execute(request: CodecRequest) -> CodecStageOutcome {
  if let Err(error) = request.case.validate() {
    return CodecStageOutcome::failed(request, "configuration", error);
  }
  let pool = match rayon::ThreadPoolBuilder::new()
    .num_threads(request.case.rayon_threads)
    .build()
  {
    Ok(pool) => pool,
    Err(error) => return CodecStageOutcome::failed(request, "rayon_pool", error.to_string()),
  };
  match pool.install(|| execute_inner(&request)) {
    Ok(measurement) => CodecStageOutcome::completed(request, measurement),
    Err((phase, message)) => CodecStageOutcome::failed(request, phase, message),
  }
}

fn execute_inner(request: &CodecRequest) -> Result<CodecMeasurement, (&'static str, String)> {
  let model_files = ModelFileIdentities::capture(&request.case)
    .map_err(|error| ("model_metadata", error))?;
  match request.case.unit {
    Unit::Byte => {
      let (encoder, load_ns, current_after_model_load_bytes) = load_byte_encoder(&request.case)?;
      run_phase(
        request,
        encoder,
        &model_files,
        load_ns,
        current_after_model_load_bytes,
      )
    }
    Unit::Unicode => {
      let (encoder, load_ns, current_after_model_load_bytes) = load_unicode_encoder(&request.case)?;
      run_phase(
        request,
        encoder,
        &model_files,
        load_ns,
        current_after_model_load_bytes,
      )
    }
  }
}

fn load_byte_encoder(config: &CodecCaseConfig) -> Result<(BpeEncoder<u8>, u64, Option<u64>), (&'static str, String)> {
  let started = Instant::now();
  let builder = BpeBuilder::new()
    .load_vocab_file(&config.vocab_path, match config.format {
      ModelFormat::Gpt2 => &Gpt2Spec as &dyn unitoken::spec::Spec<u8, Idx>,
      ModelFormat::Unitoken => &UnitokenSpec,
    })
    .map_err(|error| ("model_load", error.to_string()))?
    .load_merges_file(&config.merges_path, match config.format {
      ModelFormat::Gpt2 => &Gpt2Spec as &dyn unitoken::spec::Spec<u8, Idx>,
      ModelFormat::Unitoken => &UnitokenSpec,
    })
    .map_err(|error| ("model_load", error.to_string()))?
    .set_special_tokens(Some(config.special_tokens.clone()))
    .set_pat_str(config.pat_str.clone());
  let encoder = match config.format {
    ModelFormat::Gpt2 => builder.build(&Gpt2Spec),
    ModelFormat::Unitoken => builder.build(&UnitokenSpec),
  }.map_err(|error| ("model_build", error.to_string()))?;
  Ok((encoder, duration_ns(started.elapsed()), rss::current_rss_bytes()))
}

fn load_unicode_encoder(config: &CodecCaseConfig) -> Result<(BpeEncoder<Character>, u64, Option<u64>), (&'static str, String)> {
  let started = Instant::now();
  let mut encoder = BpeBuilder::new()
    .load_vocab_file::<Character, _>(&config.vocab_path, &UnitokenSpec)
    .map_err(|error| ("model_load", error.to_string()))?
    .load_merges_file::<Character, _>(&config.merges_path, &UnitokenSpec)
    .map_err(|error| ("model_load", error.to_string()))?
    .set_special_tokens(Some(config.special_tokens.clone()))
    .set_pat_str(config.pat_str.clone())
    .build(&UnitokenSpec)
    .map_err(|error| ("model_build", error.to_string()))?;
  let mixed_boundary = match config.unicode_bigram_mixed_boundary {
    UnicodeBigramMixedBoundaryName::Keep => UnicodeBigramMixedBoundary::Keep,
    UnicodeBigramMixedBoundaryName::Split => UnicodeBigramMixedBoundary::Split,
  };
  encoder.pre_tokenizer = encoder
    .pre_tokenizer
    .clone()
    .with_unicode_bigram_mixed_boundary(mixed_boundary);
  if let Some(path) = &config.unicode_bigrams_path {
    let bytes = fs::read(path).map_err(|error| ("unicode_bigrams_load", error.to_string()))?;
    let strings = serde_json::from_slice::<Vec<String>>(&bytes)
      .map_err(|error| ("unicode_bigrams_parse", error.to_string()))?;
    let bigrams = parse_unicode_bigrams(&strings).map_err(|error| ("unicode_bigrams_parse", error.to_string()))?;
    encoder.pre_tokenizer = encoder
      .pre_tokenizer
      .clone()
      .with_unicode_bigrams(bigrams);
  }
  Ok((encoder, duration_ns(started.elapsed()), rss::current_rss_bytes()))
}

fn build_model_report<C>(
  config: &CodecCaseConfig,
  encoder: &BpeEncoder<C>,
) -> Result<CodecModelReport, (&'static str, String)> {
  let vocab_sha256 = sha256_file(&config.vocab_path).map_err(|error| ("model_fingerprint", error))?;
  let merges_sha256 = sha256_file(&config.merges_path).map_err(|error| ("model_fingerprint", error))?;
  let unicode_bigrams_file_sha256 = config
    .unicode_bigrams_path
    .as_deref()
    .map(sha256_file)
    .transpose()
    .map_err(|error| ("model_fingerprint", error))?;
  let unicode_bigram_mixed_boundary = match encoder.pre_tokenizer.unicode_bigram_mixed_boundary {
    UnicodeBigramMixedBoundary::Keep => UnicodeBigramMixedBoundaryName::Keep,
    UnicodeBigramMixedBoundary::Split => UnicodeBigramMixedBoundaryName::Split,
  };
  let unicode_bigrams_semantic_sha256 = encoder
    .pre_tokenizer
    .unicode_bigrams
    .as_ref()
    .map(|bigrams| fingerprint_unicode_bigram_config(bigrams, unicode_bigram_mixed_boundary));
  let unicode_bigram_count = encoder
    .pre_tokenizer
    .unicode_bigrams
    .as_ref()
    .map(|bigrams| bigrams.len());
  let resolved_pat_str = encoder.pre_tokenizer.re_pat.as_str().to_string();
  let end_of_text = encoder.pre_tokenizer.end_of_text.clone();

  let mut digest = Sha256::new();
  update_hash_bytes(&mut digest, b"unitoken:codec_encoder_config:v1");
  update_hash_bytes(&mut digest, config.unit.as_str().as_bytes());
  update_hash_bytes(&mut digest, model_format_name(config.format).as_bytes());
  update_hash_bytes(&mut digest, vocab_sha256.as_bytes());
  update_hash_bytes(&mut digest, merges_sha256.as_bytes());
  update_hash_option(&mut digest, unicode_bigrams_file_sha256.as_deref());
  update_hash_option(&mut digest, unicode_bigrams_semantic_sha256.as_deref());
  update_hash_bytes(&mut digest, resolved_pat_str.as_bytes());
  update_hash_bytes(&mut digest, end_of_text.as_bytes());
  update_hash_bytes(
    &mut digest,
    unicode_bigram_mixed_boundary.as_str().as_bytes(),
  );
  digest.update((config.special_tokens.len() as u64).to_le_bytes());
  for token in &config.special_tokens {
    update_hash_bytes(&mut digest, token.as_bytes());
  }
  digest.update((encoder.special_tokens.len() as u64).to_le_bytes());
  for (token, id) in &encoder.special_tokens {
    update_hash_bytes(&mut digest, token.as_bytes());
    digest.update(id.to_le_bytes());
  }
  let encoder_config_sha256 = to_hex(&digest.finalize());

  Ok(CodecModelReport {
    vocab_size: encoder.vocab.len(),
    merge_count: encoder.merges.len(),
    special_token_count: encoder.special_tokens.len(),
    vocab_sha256,
    merges_sha256,
    unicode_bigrams_file_sha256,
    unicode_bigrams_semantic_sha256,
    unicode_bigram_count,
    unicode_bigram_mixed_boundary,
    resolved_pat_str,
    end_of_text,
    encoder_config_sha256,
  })
}

fn fingerprint_unicode_bigram_config(
  bigrams: &ahash::AHashSet<(char, char)>,
  mixed_boundary: UnicodeBigramMixedBoundaryName,
) -> String {
  let mut sorted = bigrams.iter().copied().collect::<Vec<_>>();
  sorted.sort_unstable();
  let mut digest = Sha256::new();
  update_hash_bytes(&mut digest, b"unitoken:codec_unicode_bigrams:v1");
  update_hash_bytes(&mut digest, mixed_boundary.as_str().as_bytes());
  digest.update((sorted.len() as u64).to_le_bytes());
  for (left, right) in sorted {
    digest.update((left as u32).to_le_bytes());
    digest.update((right as u32).to_le_bytes());
  }
  to_hex(&digest.finalize())
}

fn update_hash_option(digest: &mut Sha256, value: Option<&str>) {
  match value {
    Some(value) => {
      digest.update([1]);
      update_hash_bytes(digest, value.as_bytes());
    }
    None => digest.update([0]),
  }
}

fn update_hash_bytes(digest: &mut Sha256, bytes: &[u8]) {
  digest.update((bytes.len() as u64).to_le_bytes());
  digest.update(bytes);
}

fn model_format_name(format: ModelFormat) -> &'static str {
  match format {
    ModelFormat::Gpt2 => "gpt2",
    ModelFormat::Unitoken => "unitoken",
  }
}

fn run_phase<C>(
  request: &CodecRequest,
  encoder: BpeEncoder<C>,
  model_files: &ModelFileIdentities,
  model_load_ns: u64,
  current_after_model_load_bytes: Option<u64>,
) -> Result<CodecMeasurement, (&'static str, String)>
where
  BpeEncoder<C>: CanEncode<C, Idx>,
  C: Clone,
{
  let started = Instant::now();
  let model = build_model_report(&request.case, &encoder)?;
  let model_fingerprint_ns = duration_ns(started.elapsed());
  model_files
    .ensure_unchanged(&request.case)
    .map_err(|error| ("model_changed", error))?;
  match request.phase {
    CodecPhase::Encode => run_encode(
      request,
      &encoder,
      model,
      model_load_ns,
      model_fingerprint_ns,
      current_after_model_load_bytes,
    ),
    CodecPhase::Decode => run_decode(
      request,
      &encoder,
      model,
      model_load_ns,
      model_fingerprint_ns,
      current_after_model_load_bytes,
    ),
  }
}

fn run_encode<C>(
  request: &CodecRequest,
  encoder: &BpeEncoder<C>,
  model: CodecModelReport,
  model_load_ns: u64,
  model_fingerprint_ns: u64,
  current_after_model_load_bytes: Option<u64>,
) -> Result<CodecMeasurement, (&'static str, String)>
where
  BpeEncoder<C>: CanEncode<C, Idx>,
{
  let input_identity = FileIdentity::capture(&request.case.text_path)
    .map_err(|error| ("input_metadata", error))?;
  let current_before_phase_bytes = rss::current_rss_bytes();
  let sampler = rss::RssSampler::start();
  let started = Instant::now();
  let ids = encoder
    .encode_file(&request.case.text_path, request.case.requested_chunks)
    .map_err(|error| ("encode", error.to_string()))?;
  let encode_ns = duration_ns(started.elapsed());
  let current_after_phase_bytes = rss::current_rss_bytes();
  let sampled_peak_during_phase_bytes = sampler.map(rss::RssSampler::finish);
  let process_peak_rss_through_phase_bytes = rss::process_peak_rss_bytes();

  let started = Instant::now();
  let token_sha256 = fingerprint_token_ids(&ids);
  let fingerprint_ns = duration_ns(started.elapsed());

  let started = Instant::now();
  let input_sha256 = sha256_file(&request.case.text_path).map_err(|error| ("input_hash", error))?;
  let input_hash_ns = duration_ns(started.elapsed());
  input_identity
    .ensure_unchanged(&request.case.text_path)
    .map_err(|error| ("input_changed", error))?;

  let started = Instant::now();
  write_token_ids(&request.token_path, &ids).map_err(|error| ("token_artifact_write", error))?;
  let artifact_write_ns = duration_ns(started.elapsed());
  Ok(CodecMeasurement::Encode(EncodeMeasurement {
    input: CodecInputReport {
      text_bytes: input_identity.len(),
      input_sha256,
    },
    model,
    actual_rayon_threads: rayon::current_num_threads(),
    token_count: ids.len(),
    token_sha256,
    timing: CodecTiming {
      model_load_ns,
      model_fingerprint_ns,
      input_hash_ns: Some(input_hash_ns),
      token_load_ns: None,
      encode_ns: Some(encode_ns),
      decode_ns: None,
      fingerprint_ns,
      artifact_write_ns: Some(artifact_write_ns),
    },
    memory: phase_memory(
      current_after_model_load_bytes,
      current_before_phase_bytes,
      current_after_phase_bytes,
      sampled_peak_during_phase_bytes,
      process_peak_rss_through_phase_bytes,
    ),
  }))
}

fn run_decode<C>(
  request: &CodecRequest,
  encoder: &BpeEncoder<C>,
  model: CodecModelReport,
  model_load_ns: u64,
  model_fingerprint_ns: u64,
  current_after_model_load_bytes: Option<u64>,
) -> Result<CodecMeasurement, (&'static str, String)>
where
  BpeEncoder<C>: CanEncode<C, Idx>,
  C: Clone,
{
  let started = Instant::now();
  let ids = read_token_ids(&request.token_path).map_err(|error| ("token_artifact_load", error))?;
  let token_load_ns = duration_ns(started.elapsed());
  let current_before_phase_bytes = rss::current_rss_bytes();
  let sampler = rss::RssSampler::start();
  let started = Instant::now();
  let decoded = encoder.decode(&ids).map_err(|error| ("decode", error.to_string()))?;
  let decode_ns = duration_ns(started.elapsed());
  let current_after_phase_bytes = rss::current_rss_bytes();
  let sampled_peak_during_phase_bytes = sampler.map(rss::RssSampler::finish);
  let process_peak_rss_through_phase_bytes = rss::process_peak_rss_bytes();
  let started = Instant::now();
  let token_sha256 = fingerprint_token_ids(&ids);
  let decoded_sha256 = sha256_hex(decoded.as_bytes());
  let fingerprint_ns = duration_ns(started.elapsed());
  Ok(CodecMeasurement::Decode(DecodeMeasurement {
    model,
    token_count: ids.len(),
    token_sha256,
    decoded_bytes: u64::try_from(decoded.len()).map_err(|_| ("decode", "decoded byte count does not fit u64".to_string()))?,
    decoded_sha256,
    timing: CodecTiming {
      model_load_ns,
      model_fingerprint_ns,
      input_hash_ns: None,
      token_load_ns: Some(token_load_ns),
      encode_ns: None,
      decode_ns: Some(decode_ns),
      fingerprint_ns,
      artifact_write_ns: None,
    },
    memory: phase_memory(
      current_after_model_load_bytes,
      current_before_phase_bytes,
      current_after_phase_bytes,
      sampled_peak_during_phase_bytes,
      process_peak_rss_through_phase_bytes,
    ),
  }))
}

fn phase_memory(
  current_after_model_load_bytes: Option<u64>,
  current_before_phase_bytes: Option<u64>,
  current_after_phase_bytes: Option<u64>,
  sampled_peak_during_phase_bytes: Option<u64>,
  process_peak_rss_through_phase_bytes: Option<u64>,
) -> CodecMemory {
  CodecMemory {
    current_rss_source: rss::current_rss_source().map(str::to_string),
    peak_rss_source: rss::peak_rss_source().map(str::to_string),
    current_after_model_load_bytes,
    current_before_phase_bytes,
    current_after_phase_bytes,
    sampled_peak_during_phase_bytes,
    process_peak_rss_through_phase_bytes,
    rss_sample_interval_ms: sampled_peak_during_phase_bytes.is_some().then_some(rss::SAMPLE_INTERVAL.as_millis() as u64),
  }
}

fn combine_sample(
  sample_index: usize,
  encode_outcome: CodecStageOutcome,
  decode_outcome: Option<CodecStageOutcome>,
) -> CodecSample {
  let case_name = encode_outcome.request.case.name.clone();
  let mut errors = Vec::new();
  let encode = match encode_outcome.measurement {
    Some(CodecMeasurement::Encode(measurement)) => Some(measurement),
    _ => {
      if let Some(error) = encode_outcome.error { errors.push(error) }
      None
    }
  };
  let decode = match decode_outcome {
    Some(CodecStageOutcome { measurement: Some(CodecMeasurement::Decode(measurement)), .. }) => Some(measurement),
    Some(outcome) => {
      if let Some(error) = outcome.error { errors.push(error) }
      None
    }
    None => None,
  };
  let model_consistent = encode
    .as_ref()
    .zip(decode.as_ref())
    .is_some_and(|(encode, decode)| encode.model == decode.model);
  let roundtrip_valid = encode.as_ref().zip(decode.as_ref()).is_some_and(|(encode, decode)| {
    encode.token_count == decode.token_count
      && encode.token_sha256 == decode.token_sha256
      && encode.input.text_bytes == decode.decoded_bytes
      && encode.input.input_sha256 == decode.decoded_sha256
  });
  CodecSample {
    case_id: format!("{case_name}__r{sample_index}"),
    sample_index,
    encode,
    decode,
    errors,
    model_consistent,
    roundtrip_valid,
  }
}

fn evaluate_gates(config: &CodecCaseConfig, samples: &[CodecSample]) -> CodecGates {
  let all_runs_completed = !samples.is_empty()
    && samples.iter().all(|sample| sample.encode.is_some() && sample.decode.is_some() && sample.errors.is_empty());
  let models_consistent = all_runs_completed && samples.iter().all(|sample| sample.model_consistent);
  let roundtrips_valid = all_runs_completed && samples.iter().all(|sample| sample.roundtrip_valid);
  let samples_deterministic = if samples.len() < 2 || !all_runs_completed {
    None
  } else {
    let first = samples[0].encode.as_ref().unwrap();
    Some(samples.iter().all(|sample| {
      sample.encode.as_ref().is_some_and(|encode| {
        encode.input == first.input
          && encode.model == first.model
          && encode.token_count == first.token_count
          && encode.token_sha256 == first.token_sha256
          && sample.decode.as_ref().is_some_and(|decode| {
            decode.model == first.model
              && decode.token_count == first.token_count
              && decode.token_sha256 == first.token_sha256
              && decode.decoded_bytes == first.input.text_bytes
              && decode.decoded_sha256 == first.input.input_sha256
          })
      })
    }))
  };
  let input_matches_expected = config.expected_input_sha256.as_ref().map(|expected| {
    samples.iter().all(|sample| sample.encode.as_ref().is_some_and(|encode| encode.input.input_sha256.eq_ignore_ascii_case(expected)))
  });
  let model_files_match_expected = (config.expected_vocab_sha256.is_some() || config.expected_merges_sha256.is_some()).then(|| {
    samples.iter().all(|sample| sample.encode.as_ref().is_some_and(|encode| {
      config.expected_vocab_sha256.as_ref().is_none_or(|expected| encode.model.vocab_sha256.eq_ignore_ascii_case(expected))
        && config.expected_merges_sha256.as_ref().is_none_or(|expected| encode.model.merges_sha256.eq_ignore_ascii_case(expected))
    }))
  });
  let tokens_match_expected = (config.expected_token_count.is_some() || config.expected_token_sha256.is_some()).then(|| {
    samples.iter().all(|sample| sample.encode.as_ref().is_some_and(|encode| {
      config.expected_token_count.is_none_or(|expected| encode.token_count == expected)
        && config.expected_token_sha256.as_ref().is_none_or(|expected| encode.token_sha256.eq_ignore_ascii_case(expected))
    }))
  });
  let passed = all_runs_completed
    && models_consistent
    && roundtrips_valid
    && samples_deterministic != Some(false)
    && input_matches_expected != Some(false)
    && model_files_match_expected != Some(false)
    && tokens_match_expected != Some(false);
  CodecGates {
    all_runs_completed,
    models_consistent,
    roundtrips_valid,
    samples_deterministic,
    input_matches_expected,
    model_files_match_expected,
    tokens_match_expected,
    passed,
  }
}

fn validate_child_outcome(
  request: &CodecRequest,
  status: &ExitStatus,
  outcome: CodecStageOutcome,
) -> Result<CodecStageOutcome, String> {
  if outcome.request != *request || outcome.case_id != request.id() {
    return Err("child result does not match its request".to_string());
  }
  let measurement_matches_phase = match (&request.phase, &outcome.measurement) {
    (CodecPhase::Encode, Some(CodecMeasurement::Encode(_)))
    | (CodecPhase::Decode, Some(CodecMeasurement::Decode(_)))
    | (_, None) => true,
    _ => false,
  };
  if !measurement_matches_phase {
    return Err("child result measurement does not match its requested phase".to_string());
  }
  validate_outcome_shape(
    status,
    outcome.status == RunStatus::Completed,
    outcome.measurement.is_some(),
    outcome.error.is_some(),
  )?;
  Ok(outcome)
}

fn write_token_ids(path: &Path, ids: &[Idx]) -> Result<(), String> {
  let file = fs::File::create(path).map_err(|error| format!("cannot create {}: {error}", path.display()))?;
  let mut writer = BufWriter::new(file);
  for id in ids {
    writer.write_all(&id.to_le_bytes()).map_err(|error| format!("cannot write {}: {error}", path.display()))?;
  }
  writer.flush().map_err(|error| format!("cannot finish {}: {error}", path.display()))
}

fn read_token_ids(path: &Path) -> Result<Vec<Idx>, String> {
  let file = fs::File::open(path).map_err(|error| format!("cannot open {}: {error}", path.display()))?;
  let bytes_len = file
    .metadata()
    .map_err(|error| format!("cannot inspect {}: {error}", path.display()))?
    .len();
  let id_bytes = u64::try_from(std::mem::size_of::<Idx>()).expect("Idx size fits u64");
  if bytes_len % id_bytes != 0 {
    return Err(format!("{} has a partial token id", path.display()));
  }
  let token_count = usize::try_from(bytes_len / id_bytes)
    .map_err(|_| format!("{} contains too many token ids", path.display()))?;
  let mut reader = BufReader::new(file);
  let mut ids = Vec::with_capacity(token_count);
  let mut buffer = vec![0u8; 1024 * 1024];
  let mut remaining = bytes_len;
  while remaining > 0 {
    let read_len = usize::try_from(remaining.min(buffer.len() as u64)).expect("buffer length fits usize");
    reader
      .read_exact(&mut buffer[..read_len])
      .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    ids.extend(
      buffer[..read_len]
        .chunks_exact(std::mem::size_of::<Idx>())
        .map(|bytes| Idx::from_le_bytes(bytes.try_into().expect("token chunk has Idx width"))),
    );
    remaining -= read_len as u64;
  }
  Ok(ids)
}

fn codec_report_name(
  config: &CodecCaseConfig,
  repeats: usize,
  samples: &[CodecSample],
) -> Result<String, String> {
  let config_bytes = serde_json::to_vec(config)
    .map_err(|error| format!("cannot fingerprint codec configuration: {error}"))?;
  let config_sha256 = sha256_hex(&config_bytes);
  let model_key = samples
    .iter()
    .find_map(|sample| sample.encode.as_ref())
    .map(|encode| &encode.model.encoder_config_sha256[..8])
    .unwrap_or("failed");
  let input_key = samples
    .iter()
    .find_map(|sample| sample.encode.as_ref())
    .map(|encode| &encode.input.input_sha256[..8])
    .unwrap_or("failed");
  Ok(format!(
    "codec.{}.{}.{}.chunks{}.t{}.r{}.{}.{}.{}",
    config.name,
    config.unit.as_str(),
    model_format_name(config.format),
    config.requested_chunks,
    config.rayon_threads,
    repeats,
    &config_sha256[..8],
    input_key,
    model_key,
  ))
}

fn validate_codec_output_path(
  report: &Path,
  config: &CodecCaseConfig,
) -> Result<(), String> {
  let report = resolve_path_for_comparison(report)?;
  let mut inputs = vec![
    ("input corpus", config.text_path.as_path()),
    ("vocabulary", config.vocab_path.as_path()),
    ("merges", config.merges_path.as_path()),
  ];
  if let Some(path) = config.unicode_bigrams_path.as_deref() {
    inputs.push(("unicode bigram artifact", path));
  }
  for (label, path) in inputs {
    if report == resolve_path_for_comparison(path)? {
      return Err(format!("report output cannot overwrite the {label}"));
    }
  }
  Ok(())
}

fn print_summary(path: &Path, report: &CodecSuiteReport) {
  println!("codec regression report: {}", path.display());
  for sample in &report.samples {
    if let (Some(encode), Some(decode)) = (&sample.encode, &sample.decode) {
      println!(
        "  {} input={:.1} MiB tokens={} vocab={} merges={} model={}",
        sample.case_id,
        encode.input.text_bytes as f64 / 1024.0 / 1024.0,
        encode.token_count,
        encode.model.vocab_size,
        encode.model.merge_count,
        short_hash(&encode.model.encoder_config_sha256),
      );
      println!(
        "    encode {:>9.3} ms {:>8.1} MiB/s peak={} load={:.3} ms model_hash={:.3} ms",
        duration_ms(encode.timing.encode_ns.unwrap_or_default()),
        throughput_mib(encode.input.text_bytes, encode.timing.encode_ns.unwrap_or_default()),
        format_bytes(encode.memory.sampled_peak_during_phase_bytes),
        duration_ms(encode.timing.model_load_ns),
        duration_ms(encode.timing.model_fingerprint_ns),
      );
      println!(
        "    decode {:>9.3} ms {:>8.1} MiB/s peak={} load={:.3} ms token_load={:.3} ms",
        duration_ms(decode.timing.decode_ns.unwrap_or_default()),
        throughput_mib(decode.decoded_bytes, decode.timing.decode_ns.unwrap_or_default()),
        format_bytes(decode.memory.sampled_peak_during_phase_bytes),
        duration_ms(decode.timing.model_load_ns),
        duration_ms(decode.timing.token_load_ns.unwrap_or_default()),
      );
      println!(
        "    valid model={} roundtrip={} hashes input={} tokens={} config={}",
        sample.model_consistent,
        sample.roundtrip_valid,
        short_hash(&encode.input.input_sha256),
        short_hash(&encode.token_sha256),
        short_hash(&encode.model.encoder_config_sha256),
      );
    } else {
      for error in &sample.errors {
        println!("  {:<32} FAILED [{}] {}", sample.case_id, error.phase, error.message);
      }
    }
  }
  println!(
    "gates completed={} models={} roundtrips={} deterministic={:?} input_expected={:?} model_expected={:?} tokens_expected={:?}",
    report.gates.all_runs_completed,
    report.gates.models_consistent,
    report.gates.roundtrips_valid,
    report.gates.samples_deterministic,
    report.gates.input_matches_expected,
    report.gates.model_files_match_expected,
    report.gates.tokens_match_expected,
  );
  println!("correctness gates passed: {}", report.gates.passed);
}
