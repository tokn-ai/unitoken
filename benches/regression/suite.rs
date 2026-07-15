use std::{
  collections::{BTreeMap, BTreeSet},
  fs::File,
  path::{Component, Path, PathBuf},
};

use clap::Args as ClapArgs;
use serde::Deserialize;
use unitoken::pretokenizer::DEFAULT_EOT;

use crate::{
  codec::{self, CodecCaseConfig, ModelFormat},
  common::{
    config::{UnicodeBigramMixedBoundaryName, Unit},
    environment::resolve_threads,
    util::resolve_path_for_comparison,
  },
  pretokenizer::{
    self, BoundaryName, PretokenizerCaseConfig,
    UnicodeBigramConfig as RuntimeUnicodeBigramConfig,
  },
  trainer::{
    self,
    config::{CaseConfig, InitialAlphabetName, TieBreakName},
  },
};

const SCHEMA_VERSION: u32 = 1;
const CONFIG_DIRECTORY: &str = "benches/regression/config";

#[derive(Clone, Debug, ClapArgs)]
pub struct Args {
  /// Checked-in suite name from benches/regression/config/.
  #[arg(value_name = "SUITE", conflicts_with = "config")]
  suite: Option<String>,
  /// Path to a custom YAML suite, relative to the repository root.
  #[arg(long, value_name = "PATH")]
  config: Option<PathBuf>,
  /// Directory for reports and intermediate artifacts.
  #[arg(long)]
  output_dir: Option<PathBuf>,
  /// Parse and validate the suite without running benchmarks.
  #[arg(long)]
  check: bool,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct SuiteDefaults {
  repeats: Option<usize>,
  rayon_threads: Option<usize>,
  special_tokens: Option<Vec<String>>,
}

impl SuiteDefaults {
  fn repeats(&self) -> usize {
    self.repeats.unwrap_or(1)
  }

  fn special_tokens(&self) -> Vec<String> {
    self
      .special_tokens
      .clone()
      .unwrap_or_else(|| vec![DEFAULT_EOT.to_string()])
  }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SuiteConfig {
  schema_version: u32,
  name: String,
  #[serde(default)]
  defaults: SuiteDefaults,
  trainer: Option<TrainerSuiteConfig>,
  #[serde(default)]
  pretokenizer: Vec<PretokenizerConfig>,
  #[serde(default)]
  codec: Vec<CodecConfig>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrainerSuiteConfig {
  output: PathBuf,
  repeats: Option<usize>,
  rayon_threads: Option<usize>,
  #[serde(default = "default_bucket_size")]
  bucket_size: usize,
  #[serde(default = "default_hot_pair_window_sizes")]
  hot_pair_window_sizes: Vec<usize>,
  cases: Vec<TrainerConfig>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrainerConfig {
  name: String,
  words: PathBuf,
  unit: Unit,
  target_vocab_size: usize,
  #[serde(default = "default_initial_alphabet")]
  initial_alphabet: InitialAlphabetName,
  #[serde(default = "default_tie_break")]
  tie_break: TieBreakName,
  parallel_merge_min_occurs_in: Option<usize>,
  bigram_cutoff_freq: Option<i64>,
  #[serde(default)]
  bbpe_fallback: bool,
  #[serde(default = "default_primary_vocab_ratio")]
  primary_vocab_ratio: f64,
  special_tokens: Option<Vec<String>>,
  expected_input_sha256: Option<String>,
  expected_model_sha256: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PretokenizerConfig {
  name: String,
  text: PathBuf,
  output: PathBuf,
  #[serde(default = "default_chunk_size")]
  chunk_size: u64,
  #[serde(default = "default_boundary")]
  boundary: BoundaryName,
  unicode_bigrams: Option<UnicodeBigramConfig>,
  pat_str: Option<String>,
  special_tokens: Option<Vec<String>>,
  eot_token: Option<String>,
  repeats: Option<usize>,
  rayon_threads: Option<usize>,
  expected_input_sha256: Option<String>,
  expected_bigrams_sha256: Option<String>,
  expected_inventory_sha256: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UnicodeBigramConfig {
  top_k: usize,
  #[serde(default = "default_unicode_bigram_min_freq")]
  min_freq: i64,
  #[serde(default = "default_mixed_boundary")]
  mixed_boundary: UnicodeBigramMixedBoundaryName,
  artifact: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactReference {
  artifact: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CodecConfig {
  name: String,
  text: PathBuf,
  vocab: PathBuf,
  merges: PathBuf,
  output: PathBuf,
  unit: Unit,
  format: ModelFormat,
  #[serde(default = "default_codec_chunks")]
  chunks: usize,
  pat_str: Option<String>,
  #[serde(default = "default_split_on_vocab_bigrams")]
  split_on_vocab_bigrams: bool,
  unicode_bigrams: Option<ArtifactReference>,
  #[serde(default = "default_mixed_boundary")]
  unicode_bigram_mixed_boundary: UnicodeBigramMixedBoundaryName,
  special_tokens: Option<Vec<String>>,
  repeats: Option<usize>,
  rayon_threads: Option<usize>,
  expected_input_sha256: Option<String>,
  expected_vocab_sha256: Option<String>,
  expected_merges_sha256: Option<String>,
  expected_token_count: Option<usize>,
  expected_token_sha256: Option<String>,
}

pub fn run(args: Args) -> Result<(), String> {
  let manifest_dir = manifest_dir();
  let (config_path, config) = load_selected_config(
    args.suite.as_deref(),
    args.config.as_deref(),
    &manifest_dir,
  )?;
  let output_dir = resolve_output_directory(args.output_dir.as_deref(), &config, &manifest_dir);
  validate_config(&config, &config_path, &output_dir, &manifest_dir)?;
  if args.check {
    println!(
      "benchmark suite config is valid: {} ({})",
      config.name,
      config_path.display(),
    );
    return Ok(());
  }
  run_config(&config, &manifest_dir, &output_dir)
}

pub fn run_smoke_trainer(options: trainer::SuiteOptions) -> Result<(), String> {
  let manifest_dir = manifest_dir();
  let (_, config) = load_named_config("smoke", &manifest_dir)?;
  validate_config_header(&config)?;
  let trainer_config = config
    .trainer
    .as_ref()
    .ok_or_else(|| "smoke suite does not define trainer cases".to_string())?;
  let cases = build_trainer_cases(trainer_config, &config.defaults, &manifest_dir, &options)?;
  trainer::run_suite(&config.name, cases, options)
}

fn run_config(config: &SuiteConfig, manifest_dir: &Path, output_dir: &Path) -> Result<(), String> {
  let artifacts = declared_artifacts(config, output_dir)?;
  if let Some(trainer_config) = &config.trainer {
    let options = trainer_options(trainer_config, &config.defaults, output_dir)?;
    let cases = build_trainer_cases(trainer_config, &config.defaults, manifest_dir, &options)?;
    trainer::run_suite(&config.name, cases, options)?;
  }

  for case in &config.pretokenizer {
    let runtime = build_pretokenizer_case(case, &config.defaults, manifest_dir, &artifacts)?;
    pretokenizer::run_config(
      runtime,
      case.repeats.unwrap_or_else(|| config.defaults.repeats()),
      Some(resolve_artifact(output_dir, &case.output, "pretokenizer output")?),
    )?;
  }

  for case in &config.codec {
    let runtime = build_codec_case(case, &config.defaults, manifest_dir, &artifacts)?;
    codec::run_config(
      runtime,
      case.repeats.unwrap_or_else(|| config.defaults.repeats()),
      Some(resolve_artifact(output_dir, &case.output, "codec output")?),
    )?;
  }
  Ok(())
}

fn trainer_options(
  config: &TrainerSuiteConfig,
  defaults: &SuiteDefaults,
  output_dir: &Path,
) -> Result<trainer::SuiteOptions, String> {
  Ok(trainer::SuiteOptions {
    repeats: config.repeats.unwrap_or_else(|| defaults.repeats()),
    hot_pair_window_sizes: config.hot_pair_window_sizes.clone(),
    rayon_threads: config.rayon_threads.or(defaults.rayon_threads),
    bucket_size: config.bucket_size,
    output: Some(resolve_artifact(output_dir, &config.output, "trainer output")?),
  })
}

fn build_pretokenizer_case(
  case: &PretokenizerConfig,
  defaults: &SuiteDefaults,
  manifest_dir: &Path,
  artifacts: &BTreeMap<String, PathBuf>,
) -> Result<PretokenizerCaseConfig, String> {
  let special_tokens = case.special_tokens.clone().unwrap_or_else(|| defaults.special_tokens());
  let eot_token = case
    .eot_token
    .clone()
    .or_else(|| special_tokens.first().cloned())
    .unwrap_or_else(|| DEFAULT_EOT.to_string());
  let unicode_bigrams = case.unicode_bigrams.as_ref().map(|bigrams| RuntimeUnicodeBigramConfig {
    top_k: bigrams.top_k,
    min_freq: bigrams.min_freq,
    mixed_boundary: bigrams.mixed_boundary,
  });
  let unicode_bigrams_output = case
    .unicode_bigrams
    .as_ref()
    .map(|bigrams| artifact_from_map(artifacts, &bigrams.artifact))
    .transpose()?;
  Ok(PretokenizerCaseConfig {
    name: case.name.clone(),
    text_path: resolve_input(manifest_dir, &case.text),
    chunk_size: case.chunk_size,
    boundary: case.boundary,
    unicode_bigrams,
    unicode_bigrams_output,
    pat_str: case.pat_str.clone(),
    special_tokens,
    eot_token,
    rayon_threads: resolve_threads(case.rayon_threads.or(defaults.rayon_threads))?,
    expected_input_sha256: case.expected_input_sha256.clone(),
    expected_bigrams_sha256: case.expected_bigrams_sha256.clone(),
    expected_inventory_sha256: case.expected_inventory_sha256.clone(),
  })
}

fn build_codec_case(
  case: &CodecConfig,
  defaults: &SuiteDefaults,
  manifest_dir: &Path,
  artifacts: &BTreeMap<String, PathBuf>,
) -> Result<CodecCaseConfig, String> {
  let unicode_bigrams_path = case
    .unicode_bigrams
    .as_ref()
    .map(|reference| artifact_from_map(artifacts, &reference.artifact))
    .transpose()?;
  Ok(CodecCaseConfig {
    name: case.name.clone(),
    text_path: resolve_input(manifest_dir, &case.text),
    vocab_path: resolve_input(manifest_dir, &case.vocab),
    merges_path: resolve_input(manifest_dir, &case.merges),
    unit: case.unit,
    format: case.format,
    requested_chunks: case.chunks,
    pat_str: case.pat_str.clone(),
    split_on_vocab_bigrams: case.split_on_vocab_bigrams,
    unicode_bigrams_path,
    unicode_bigram_mixed_boundary: case.unicode_bigram_mixed_boundary,
    special_tokens: case.special_tokens.clone().unwrap_or_else(|| defaults.special_tokens()),
    rayon_threads: resolve_threads(case.rayon_threads.or(defaults.rayon_threads))?,
    expected_input_sha256: case.expected_input_sha256.clone(),
    expected_vocab_sha256: case.expected_vocab_sha256.clone(),
    expected_merges_sha256: case.expected_merges_sha256.clone(),
    expected_token_count: case.expected_token_count,
    expected_token_sha256: case.expected_token_sha256.clone(),
  })
}

fn build_trainer_cases(
  config: &TrainerSuiteConfig,
  defaults: &SuiteDefaults,
  manifest_dir: &Path,
  options: &trainer::SuiteOptions,
) -> Result<Vec<CaseConfig>, String> {
  let rayon_threads = resolve_threads(options.rayon_threads)?;
  Ok(config
    .cases
    .iter()
    .map(|case| CaseConfig {
      name: case.name.clone(),
      words_path: resolve_input(manifest_dir, &case.words),
      unit: case.unit,
      initial_alphabet: case.initial_alphabet,
      tie_break: case.tie_break,
      parallel_merge_min_occurs_in: case.parallel_merge_min_occurs_in,
      target_vocab_size: case.target_vocab_size,
      special_tokens: case.special_tokens.clone().unwrap_or_else(|| defaults.special_tokens()),
      bucket_size: options.bucket_size,
      bigram_cutoff_freq: case.bigram_cutoff_freq,
      bbpe_fallback: case.bbpe_fallback,
      primary_vocab_ratio: case.primary_vocab_ratio,
      expected_input_sha256: case.expected_input_sha256.clone(),
      expected_model_sha256: case.expected_model_sha256.clone(),
      rayon_threads,
    })
    .collect())
}

fn load_selected_config(
  name: Option<&str>,
  custom_path: Option<&Path>,
  manifest_dir: &Path,
) -> Result<(PathBuf, SuiteConfig), String> {
  match custom_path {
    Some(path) => load_config_path(&resolve_input(manifest_dir, path)),
    None => load_named_config(name.unwrap_or("smoke"), manifest_dir),
  }
}

fn load_named_config(name: &str, manifest_dir: &Path) -> Result<(PathBuf, SuiteConfig), String> {
  validate_slug(name, "suite selector")?;
  load_config_path(
    &manifest_dir
      .join(CONFIG_DIRECTORY)
      .join(format!("{name}.yml")),
  )
}

fn load_config_path(config_path: &Path) -> Result<(PathBuf, SuiteConfig), String> {
  let file = File::open(config_path)
    .map_err(|error| format!("failed to open suite config {}: {error}", config_path.display()))?;
  let config: SuiteConfig = serde_yaml_ng::from_reader(file)
    .map_err(|error| format!("failed to parse suite config {}: {error}", config_path.display()))?;
  Ok((config_path.to_path_buf(), config))
}

fn resolve_output_directory(output_dir: Option<&Path>, config: &SuiteConfig, manifest_dir: &Path) -> PathBuf {
  match output_dir {
    Some(path) => resolve_input(manifest_dir, path),
    None => manifest_dir.join("out/benchmarks/regression").join(&config.name),
  }
}

fn resolve_input(manifest_dir: &Path, path: &Path) -> PathBuf {
  if path.is_absolute() {
    path.to_path_buf()
  } else {
    manifest_dir.join(path)
  }
}

fn resolve_artifact(output_dir: &Path, artifact: &Path, label: &str) -> Result<PathBuf, String> {
  if artifact.as_os_str().is_empty()
    || artifact.is_absolute()
    || artifact
      .components()
      .any(|component| !matches!(component, Component::Normal(_)))
  {
    return Err(format!("{label} must be a non-empty relative path without '..'"));
  }
  Ok(output_dir.join(artifact))
}

fn artifact_path(output_dir: &Path, artifact: &str) -> Result<PathBuf, String> {
  validate_slug(artifact, "artifact name")?;
  Ok(output_dir.join(format!("{artifact}.json")))
}

fn declared_artifacts(
  config: &SuiteConfig,
  output_dir: &Path,
) -> Result<BTreeMap<String, PathBuf>, String> {
  let mut artifacts = BTreeMap::new();
  for case in &config.pretokenizer {
    let Some(bigrams) = &case.unicode_bigrams else {
      continue;
    };
    let path = artifact_path(output_dir, &bigrams.artifact)?;
    if artifacts.insert(bigrams.artifact.clone(), path).is_some() {
      return Err(format!("duplicate artifact producer {}", bigrams.artifact));
    }
  }
  Ok(artifacts)
}

fn artifact_from_map(
  artifacts: &BTreeMap<String, PathBuf>,
  artifact: &str,
) -> Result<PathBuf, String> {
  validate_slug(artifact, "artifact name")?;
  artifacts
    .get(artifact)
    .cloned()
    .ok_or_else(|| format!("artifact {artifact} has no pretokenizer producer in this suite"))
}

fn validate_config(
  config: &SuiteConfig,
  config_path: &Path,
  output_dir: &Path,
  manifest_dir: &Path,
) -> Result<(), String> {
  validate_config_header(config)?;

  let artifacts = declared_artifacts(config, output_dir)?;
  let mut outputs = BTreeSet::new();
  for artifact in artifacts.values() {
    insert_output(&mut outputs, artifact.clone())?;
  }
  let mut inputs = BTreeSet::from([resolve_path_for_comparison(config_path)?]);
  let mut case_names = BTreeSet::new();
  if let Some(trainer) = &config.trainer {
    if trainer.cases.is_empty() {
      return Err("trainer cases cannot be empty".to_string());
    }
    insert_output(&mut outputs, resolve_artifact(output_dir, &trainer.output, "trainer output")?)?;
    let mut options = trainer_options(trainer, &config.defaults, output_dir)?;
    trainer::validate_suite_options(&mut options)?;
    let runtime_cases = build_trainer_cases(trainer, &config.defaults, manifest_dir, &options)?;
    for (case, runtime) in trainer.cases.iter().zip(runtime_cases) {
      insert_case_name(&mut case_names, "trainer", &case.name)?;
      runtime.validate()?;
      inputs.insert(resolve_path_for_comparison(&runtime.words_path)?);
    }
  }
  for case in &config.pretokenizer {
    insert_case_name(&mut case_names, "pretokenizer", &case.name)?;
    if case.repeats == Some(0) || case.rayon_threads == Some(0) || case.chunk_size == 0 {
      return Err(format!("pretokenizer case {} has a non-positive run setting", case.name));
    }
    insert_output(
      &mut outputs,
      resolve_artifact(output_dir, &case.output, "pretokenizer output")?,
    )?;
    let runtime = build_pretokenizer_case(case, &config.defaults, manifest_dir, &artifacts)?;
    runtime.validate()?;
    inputs.insert(resolve_path_for_comparison(&runtime.text_path)?);
  }
  for case in &config.codec {
    insert_case_name(&mut case_names, "codec", &case.name)?;
    if case.repeats == Some(0) || case.rayon_threads == Some(0) || case.chunks == 0 {
      return Err(format!("codec case {} has a non-positive run setting", case.name));
    }
    insert_output(&mut outputs, resolve_artifact(output_dir, &case.output, "codec output")?)?;
    let runtime = build_codec_case(case, &config.defaults, manifest_dir, &artifacts)?;
    runtime.validate()?;
    inputs.insert(resolve_path_for_comparison(&runtime.text_path)?);
    inputs.insert(resolve_path_for_comparison(&runtime.vocab_path)?);
    inputs.insert(resolve_path_for_comparison(&runtime.merges_path)?);
  }
  if case_names.is_empty() {
    return Err("suite must define at least one benchmark case".to_string());
  }
  for output in outputs {
    if inputs.contains(&output) {
      return Err(format!("suite output would overwrite input {}", output.display()));
    }
  }
  Ok(())
}

fn validate_config_header(config: &SuiteConfig) -> Result<(), String> {
  if config.schema_version != SCHEMA_VERSION {
    return Err(format!(
      "unsupported suite schema version {}; expected {SCHEMA_VERSION}",
      config.schema_version,
    ));
  }
  validate_slug(&config.name, "suite name")?;
  if config.defaults.repeats() == 0 {
    return Err("default repeats must be positive".to_string());
  }
  if config.defaults.rayon_threads == Some(0) {
    return Err("default rayon_threads must be positive".to_string());
  }
  validate_special_tokens(&config.defaults.special_tokens())?;
  Ok(())
}

fn validate_slug(value: &str, label: &str) -> Result<(), String> {
  if value.is_empty()
    || !value.bytes().all(|byte| {
      byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')
    })
  {
    return Err(format!(
      "{label} must contain only ASCII letters, numbers, '-' or '_'",
    ));
  }
  Ok(())
}

fn insert_case_name(names: &mut BTreeSet<String>, kind: &str, name: &str) -> Result<(), String> {
  if name.trim().is_empty() {
    return Err(format!("{kind} case name cannot be empty"));
  }
  if !names.insert(name.to_string()) {
    return Err(format!("duplicate benchmark case name {name}"));
  }
  Ok(())
}

fn insert_output(outputs: &mut BTreeSet<PathBuf>, output: PathBuf) -> Result<(), String> {
  let comparable = resolve_path_for_comparison(&output)?;
  if !outputs.insert(comparable) {
    return Err(format!("duplicate suite output path {}", output.display()));
  }
  Ok(())
}

fn validate_special_tokens(tokens: &[String]) -> Result<(), String> {
  if tokens.is_empty() || tokens.iter().any(String::is_empty) {
    return Err("special_tokens must contain non-empty values".to_string());
  }
  if tokens.iter().collect::<BTreeSet<_>>().len() != tokens.len() {
    return Err("special_tokens cannot contain duplicates".to_string());
  }
  Ok(())
}

fn manifest_dir() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn default_bucket_size() -> usize {
  500
}

fn default_hot_pair_window_sizes() -> Vec<usize> {
  vec![4096]
}

fn default_initial_alphabet() -> InitialAlphabetName {
  InitialAlphabetName::RawBytes
}

fn default_tie_break() -> TieBreakName {
  TieBreakName::SmallestPairId
}

fn default_primary_vocab_ratio() -> f64 {
  0.9
}

fn default_chunk_size() -> u64 {
  pretokenizer::DEFAULT_CHUNK_SIZE
}

fn default_boundary() -> BoundaryName {
  BoundaryName::Auto
}

fn default_unicode_bigram_min_freq() -> i64 {
  pretokenizer::DEFAULT_UNICODE_BIGRAM_MIN_FREQ
}

fn default_mixed_boundary() -> UnicodeBigramMixedBoundaryName {
  UnicodeBigramMixedBoundaryName::Keep
}

fn default_codec_chunks() -> usize {
  codec::DEFAULT_CHUNKS
}

fn default_split_on_vocab_bigrams() -> bool {
  true
}

#[cfg(test)]
#[allow(
  unused_imports,
  reason = "harness-free cargo bench builds cfg(test) without the test harness",
)]
mod tests {
  use std::path::{Path, PathBuf};

  use unitoken::pretokenizer::DEFAULT_EOT;

  use crate::common::config::Unit;
  use crate::trainer::{
    self,
    config::{InitialAlphabetName, TieBreakName},
  };

  use super::{
    ArtifactReference, SuiteConfig, artifact_path, build_trainer_cases, load_named_config,
    load_selected_config, manifest_dir, resolve_artifact, trainer_options, validate_config,
  };

  #[test]
  fn checked_in_suite_configs_are_valid() {
    let manifest_dir = manifest_dir();
    for name in ["smoke", "64mib", "1gib"] {
      let (config_path, config) = load_named_config(name, &manifest_dir).unwrap();
      let output_dir = manifest_dir.join("out/benchmarks/regression/config-test");
      validate_config(&config, &config_path, &output_dir, &manifest_dir).unwrap();
    }
  }

  #[test]
  fn artifacts_cannot_escape_the_output_directory() {
    let output_dir = Path::new("out");
    assert!(resolve_artifact(output_dir, Path::new("trainer.json"), "report").is_ok());
    assert!(resolve_artifact(output_dir, Path::new("nested/trainer.json"), "report").is_ok());
    assert!(resolve_artifact(output_dir, Path::new("../trainer.json"), "report").is_err());
    assert!(resolve_artifact(output_dir, Path::new("/tmp/trainer.json"), "report").is_err());
    assert!(artifact_path(output_dir, "unicode_bigrams").is_ok());
    assert!(artifact_path(output_dir, "../unicode_bigrams").is_err());
  }

  #[test]
  fn unknown_yaml_fields_are_rejected() {
    let error = serde_yaml_ng::from_str::<SuiteConfig>(
      "schema_version: 1\nname: test\nunknown: true\n",
    )
    .unwrap_err();
    assert!(error.to_string().contains("unknown field"));
  }

  #[test]
  fn native_component_validation_is_applied() {
    let manifest_dir = manifest_dir();
    let (config_path, mut config) = load_named_config("smoke", &manifest_dir).unwrap();
    config.trainer.as_mut().unwrap().cases[0].target_vocab_size = 1;

    let error = validate_config(&config, &config_path, Path::new("out"), &manifest_dir).unwrap_err();
    assert!(error.contains("below the initial vocabulary"));

    let (_, mut config) = load_named_config("smoke", &manifest_dir).unwrap();
    config
      .trainer
      .as_mut()
      .unwrap()
      .hot_pair_window_sizes
      .clear();

    let error = validate_config(&config, &config_path, Path::new("out"), &manifest_dir).unwrap_err();
    assert!(error.contains("at least one hot-pair window size is required"));

    let (_, mut config) = load_named_config("smoke", &manifest_dir).unwrap();
    config.trainer.as_mut().unwrap().cases[0].special_tokens = Some(vec![String::new()]);

    let error = validate_config(&config, &config_path, Path::new("out"), &manifest_dir).unwrap_err();
    assert!(error.contains("contains an empty special token"));

    let (_, mut config) = load_named_config("smoke", &manifest_dir).unwrap();
    config.trainer.as_mut().unwrap().cases[0].bbpe_fallback = true;

    let error = validate_config(&config, &config_path, Path::new("out"), &manifest_dir).unwrap_err();
    assert!(error.contains("non-Unicode unit"));

    let (_, mut config) = load_named_config("smoke", &manifest_dir).unwrap();
    config.trainer.as_mut().unwrap().cases[2].primary_vocab_ratio = f64::NAN;

    let error = validate_config(&config, &config_path, Path::new("out"), &manifest_dir).unwrap_err();
    assert!(error.contains("finite range [0, 1]"));

    let (_, mut config) = load_named_config("smoke", &manifest_dir).unwrap();
    config.pretokenizer[0]
      .unicode_bigrams
      .as_mut()
      .unwrap()
      .top_k = 0;

    let error = validate_config(&config, &config_path, Path::new("out"), &manifest_dir).unwrap_err();
    assert!(error.contains("unicode_bigram_top_k must be positive"));

    let (_, mut config) = load_named_config("smoke", &manifest_dir).unwrap();
    config.codec[0].unit = Unit::Unicode;

    let error = validate_config(&config, &config_path, Path::new("out"), &manifest_dir).unwrap_err();
    assert!(error.contains("gpt2 format is not compatible with unicode units"));

    let (_, mut config) = load_named_config("smoke", &manifest_dir).unwrap();
    config.pretokenizer[0].pat_str = Some("(".to_string());

    let error = validate_config(&config, &config_path, Path::new("out"), &manifest_dir).unwrap_err();
    assert!(error.contains("invalid pretokenizer configuration"));
  }

  #[test]
  fn artifact_consumers_require_a_suite_producer() {
    let manifest_dir = manifest_dir();
    let (config_path, mut config) = load_named_config("smoke", &manifest_dir).unwrap();
    config.codec[1].unicode_bigrams = Some(ArtifactReference {
      artifact: "missing".to_string(),
    });

    let error = validate_config(&config, &config_path, Path::new("out"), &manifest_dir).unwrap_err();
    assert!(error.contains("artifact missing has no pretokenizer producer"));
  }

  #[test]
  fn artifact_producers_and_outputs_are_unique() {
    let manifest_dir = manifest_dir();
    let (config_path, mut config) = load_named_config("smoke", &manifest_dir).unwrap();
    config.pretokenizer.push(config.pretokenizer[0].clone());

    let error = validate_config(&config, &config_path, Path::new("out"), &manifest_dir).unwrap_err();
    assert!(error.contains("duplicate artifact producer unicode_bigrams"));

    let (_, mut config) = load_named_config("smoke", &manifest_dir).unwrap();
    config.codec[0].output = PathBuf::from("unicode_bigrams.json");

    let error = validate_config(&config, &config_path, Path::new("out"), &manifest_dir).unwrap_err();
    assert!(error.contains("duplicate suite output path"));
  }

  #[test]
  fn suite_outputs_cannot_overwrite_any_input() {
    let manifest_dir = manifest_dir();
    let (config_path, mut config) = load_named_config("smoke", &manifest_dir).unwrap();
    config.codec[0].output = PathBuf::from("fixtures/tinystories_sample_5M.txt");

    let error = validate_config(&config, &config_path, &manifest_dir, &manifest_dir).unwrap_err();
    assert!(error.contains("suite output would overwrite input"));

    let (_, mut config) = load_named_config("smoke", &manifest_dir).unwrap();
    config.trainer.as_mut().unwrap().output = PathBuf::from("smoke.yml");

    let output_dir = config_path.parent().unwrap();
    let error = validate_config(&config, &config_path, output_dir, &manifest_dir).unwrap_err();
    assert!(error.contains("suite output would overwrite input"));
  }

  #[test]
  fn custom_config_paths_are_repository_relative() {
    let manifest_dir = manifest_dir();
    let relative = Path::new("benches/regression/config/smoke.yml");
    let (path, config) = load_selected_config(None, Some(relative), &manifest_dir).unwrap();

    assert_eq!(path, manifest_dir.join(relative));
    assert_eq!(config.name, "smoke");

    let trainer = config.trainer.as_ref().unwrap();
    let output_dir = manifest_dir.join("out/benchmarks/regression/config-test");
    let options = trainer_options(trainer, &config.defaults, &output_dir).unwrap();
    let cases = build_trainer_cases(trainer, &config.defaults, &manifest_dir, &options).unwrap();
    assert_eq!(
      cases[0].words_path,
      manifest_dir.join("fixtures/_words.tinystories_sample_5M.json"),
    );
  }

  #[test]
  fn smoke_report_contract_remains_stable() {
    let manifest_dir = manifest_dir();
    let (_, config) = load_named_config("smoke", &manifest_dir).unwrap();
    let trainer = config.trainer.as_ref().unwrap();

    assert_eq!(trainer.output, Path::new("trainer.json"));
    assert_eq!(config.pretokenizer[0].output, Path::new("pretokenizer.json"));
    assert_eq!(config.codec.len(), 3);
    assert_eq!(config.codec[0].output, Path::new("codec-byte.json"));
    assert_eq!(config.codec[1].output, Path::new("codec-unicode.json"));
    assert!(config.codec[0].split_on_vocab_bigrams);
    assert!(config.codec[1].split_on_vocab_bigrams);
    assert!(config.codec[1].unicode_bigrams.is_none());

    let bbpe_codec = &config.codec[2];
    assert_eq!(bbpe_codec.name, "ci_zh_bbpe_codec");
    assert_eq!(bbpe_codec.output, Path::new("codec-unicode-bbpe.json"));
    assert!(bbpe_codec.split_on_vocab_bigrams);
    assert!(bbpe_codec.unicode_bigrams.is_none());
    assert_eq!(
      bbpe_codec.vocab,
      Path::new("fixtures/vocab.TinyStories_all_data_zh_1M-sample.bbpe-r90-v1000.uni.json"),
    );
    assert_eq!(
      bbpe_codec.merges,
      Path::new("fixtures/merges.TinyStories_all_data_zh_1M-sample.bbpe-r90-v1000.uni.txt"),
    );
    assert_eq!(
      bbpe_codec.expected_input_sha256.as_deref(),
      Some("c298b1680c4378091ad9e39126ac0858d78e547f3744d1a30442c12adac8e9f3"),
    );
    assert_eq!(
      bbpe_codec.expected_vocab_sha256.as_deref(),
      Some("b6a08163475d8164460309cef310a81fe344ceff1cc534b6ab9953d2a086024f"),
    );
    assert_eq!(
      bbpe_codec.expected_merges_sha256.as_deref(),
      Some("34ce25301c12d85beb0fca43d4381c1ddfa7e465a883fab090e9d42b671ffdce"),
    );
    assert_eq!(bbpe_codec.expected_token_count, Some(1_643_864));
    assert_eq!(
      bbpe_codec.expected_token_sha256.as_deref(),
      Some("6fdc974538362ae0f26af15fa10a0a94eaf75751e9ad0f9f2f6e7b8edcb1874a"),
    );
    assert_eq!(
      trainer
        .cases
        .iter()
        .map(|case| case.name.as_str())
        .collect::<Vec<_>>(),
      [
        "smoke_en_byte_v300",
        "smoke_en_byte_v1000",
        "smoke_zh_unicode_v300",
        "smoke_zh_unicode_v1000",
        "smoke_zh_unicode_bbpe_r90_v1000",
      ],
    );

    let options = trainer::SuiteOptions {
      repeats: 2,
      hot_pair_window_sizes: vec![4096],
      rayon_threads: Some(3),
      bucket_size: 701,
      output: None,
    };
    let cases = build_trainer_cases(trainer, &config.defaults, &manifest_dir, &options).unwrap();
    let expected = [
      (
        "smoke_en_byte_v300",
        Unit::Byte,
        300,
        "20b257111ca6e5ce81ee0d0e78924b9987db13029d7d006e4eb981cca151c9f4",
        "fa65e898d4cec1be5b78732ec4738b20213856a2de73bba5ca34366d347e91c0",
        false,
      ),
      (
        "smoke_en_byte_v1000",
        Unit::Byte,
        1000,
        "20b257111ca6e5ce81ee0d0e78924b9987db13029d7d006e4eb981cca151c9f4",
        "197a9f7d6ec3630370b1a30e0392b0f2fbcd2de1d36ee4d05884f01f2a877be9",
        false,
      ),
      (
        "smoke_zh_unicode_v300",
        Unit::Unicode,
        300,
        "ffb74990eb0b04ca0986a24ead7acf63e5483df7afb68c65ad2c397497a67c6a",
        "b3f2e74a4b169244774d71cd289d246847d4a56e585411436c1e4c44219e7b3a",
        false,
      ),
      (
        "smoke_zh_unicode_v1000",
        Unit::Unicode,
        1000,
        "ffb74990eb0b04ca0986a24ead7acf63e5483df7afb68c65ad2c397497a67c6a",
        "34dcb3aeb65c2220f50158d594defb73f1d5649b296c0020220266ba70f1d9e1",
        false,
      ),
      (
        "smoke_zh_unicode_bbpe_r90_v1000",
        Unit::Unicode,
        1000,
        "ffb74990eb0b04ca0986a24ead7acf63e5483df7afb68c65ad2c397497a67c6a",
        "7f6216c40793da7bf8e98dab3ffa90a15c8a233af990674b5ac7b3c861417639",
        true,
      ),
    ];
    for (case, expected) in cases.iter().zip(expected) {
      assert_eq!(case.name, expected.0);
      assert_eq!(case.unit, expected.1);
      assert_eq!(case.target_vocab_size, expected.2);
      assert_eq!(case.expected_input_sha256.as_deref(), Some(expected.3));
      assert_eq!(case.expected_model_sha256.as_deref(), Some(expected.4));
      assert_eq!(case.initial_alphabet, InitialAlphabetName::RawBytes);
      assert_eq!(case.tie_break, TieBreakName::SmallestPairId);
      assert_eq!(case.parallel_merge_min_occurs_in, None);
      assert_eq!(case.special_tokens, [DEFAULT_EOT]);
      assert_eq!(case.bucket_size, 701);
      assert_eq!(case.bigram_cutoff_freq, None);
      assert_eq!(case.bbpe_fallback, expected.5);
      assert_eq!(case.primary_vocab_ratio, 0.9);
      assert_eq!(case.rayon_threads, 3);
    }
  }
}
