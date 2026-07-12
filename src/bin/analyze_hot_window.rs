use std::{
  error::Error,
  fs::{self, File},
  io::{BufReader, BufWriter, Write},
  path::{Path, PathBuf},
  process::Command,
};

use clap::{Parser, ValueEnum};
use ordermap::OrderMap;
use serde::Serialize;
use unitoken::bpe::{
  trainer::analysis::{
    analyze_byte_words, analyze_unicode_words, HotWindowAnalysisReport,
    HotWindowPolicy,
  },
  BpeTrainerConfig, Freq, InitialAlphabet, TieBreak,
};

const TOOL_NAME: &str = "analyze_hot_window";

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Unit {
  Byte,
  Unicode,
}

impl Unit {
  fn as_str(self) -> &'static str {
    match self {
      Self::Byte => "byte",
      Self::Unicode => "unicode",
    }
  }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TieBreakArg {
  SmallestPairId,
  LargestContent,
}

impl From<TieBreakArg> for TieBreak {
  fn from(value: TieBreakArg) -> Self {
    match value {
      TieBreakArg::SmallestPairId => Self::SmallestPairId,
      TieBreakArg::LargestContent => Self::LargestContent,
    }
  }
}

impl TieBreakArg {
  fn as_str(self) -> &'static str {
    match self {
      Self::SmallestPairId => "smallest_pair_id",
      Self::LargestContent => "largest_content",
    }
  }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PolicyArg {
  ReplaceTopK,
  ThresholdNoEvict,
}

impl From<PolicyArg> for HotWindowPolicy {
  fn from(value: PolicyArg) -> Self {
    match value {
      PolicyArg::ReplaceTopK => Self::ReplaceTopK,
      PolicyArg::ThresholdNoEvict => Self::ThresholdNoEvict,
    }
  }
}

impl PolicyArg {
  fn as_str(self) -> &'static str {
    HotWindowPolicy::from(self).as_str()
  }
}

#[derive(Debug, Parser)]
#[command(about = "Simulate exact BPE occurrence hot windows over a saved word inventory")]
struct Args {
  /// JSON object mapping pre-token strings to signed frequencies.
  #[arg(long)]
  words: PathBuf,

  #[arg(long, value_enum, default_value_t = Unit::Byte)]
  unit: Unit,

  #[arg(long, default_value_t = 10_000)]
  vocab_size: usize,

  /// Comma-separated top-K refill sizes; threshold-no-evict may grow beyond K.
  #[arg(long, value_delimiter = ',', default_value = "256,1024,4096,16384")]
  window_sizes: Vec<usize>,

  #[arg(long, value_enum, default_value_t = TieBreakArg::SmallestPairId)]
  tie_break: TieBreakArg,

  #[arg(long, value_enum, default_value_t = PolicyArg::ThresholdNoEvict)]
  policy: PolicyArg,

  /// Repeat to reserve more than one special token.
  #[arg(long = "special-token", default_value = "<|endoftext|>")]
  special_tokens: Vec<String>,

  #[arg(long)]
  dataset_name: Option<String>,

  #[arg(long)]
  config_name: Option<String>,

  #[arg(long)]
  experiment_name: Option<String>,

  /// Root for conventionally named reports; ignored when --json is set.
  #[arg(long, default_value = "out/benchmarks")]
  json_dir: PathBuf,

  /// Explicit report path.
  #[arg(long, alias = "output")]
  json: Option<PathBuf>,

  #[arg(long)]
  quiet: bool,
}

#[derive(Serialize)]
struct Metadata<'a> {
  benchmark_contract: &'a str,
  script_name: &'a str,
  unitoken_version: &'a str,
  dataset_name: &'a str,
  config_name: &'a str,
  experiment_name: &'a str,
  generated_at: String,
  git_sha: Option<String>,
  git_dirty: Option<bool>,
  build_profile: &'a str,
  rayon_threads: usize,
  notes: Vec<String>,
}

#[derive(Serialize)]
struct Source<'a> {
  input_kind: &'a str,
  input_order: &'a str,
  words: &'a Path,
  file_bytes: u64,
  raw_unique_words: usize,
  raw_weighted_occurrences: i128,
  effective_unique_words: usize,
  effective_weighted_occurrences: i128,
  unit: &'a str,
  initial_alphabet: &'a str,
  tie_break: &'a str,
  special_tokens: &'a [String],
}

#[derive(Serialize)]
struct Output<'a> {
  metadata: Metadata<'a>,
  source: Source<'a>,
  analysis: HotWindowAnalysisReport,
}

fn inferred_dataset_name(path: &Path) -> String {
  path
    .parent()
    .and_then(Path::file_name)
    .or_else(|| path.file_stem())
    .and_then(|name| name.to_str())
    .unwrap_or("words")
    .to_string()
}

fn default_output_path(
  json_dir: &Path,
  dataset_name: &str,
  config_name: &str,
  experiment_name: &str,
  vocab_size: usize,
) -> PathBuf {
  json_dir
    .join(TOOL_NAME)
    .join(format!(
      "{dataset_name}.{config_name}.{experiment_name}.vocab{vocab_size}.json",
    ))
}

fn git_sha() -> Option<String> {
  let output = Command::new("git")
    .args(["rev-parse", "HEAD"])
    .current_dir(env!("CARGO_MANIFEST_DIR"))
    .output()
    .ok()?;
  output.status.success().then(|| {
    String::from_utf8_lossy(&output.stdout).trim().to_string()
  })
}

fn git_dirty() -> Option<bool> {
  let output = Command::new("git")
    .args(["status", "--porcelain", "--untracked-files=normal"])
    .current_dir(env!("CARGO_MANIFEST_DIR"))
    .output()
    .ok()?;
  output.status.success().then_some(!output.stdout.is_empty())
}

fn main() -> Result<(), Box<dyn Error>> {
  let args = Args::parse();
  if args.vocab_size == 0 {
    return Err("--vocab-size must be positive".into());
  }
  if args.window_sizes.is_empty() || args.window_sizes.contains(&0) {
    return Err("--window-sizes must contain positive integers".into());
  }
  let build_profile = if cfg!(debug_assertions) { "debug" } else { "release" };
  if cfg!(debug_assertions) {
    eprintln!("warning: debug build; use `cargo run --release --features analysis --bin analyze_hot_window -- ...` for timing data");
  }

  let file = File::open(&args.words)?;
  let words = serde_json::from_reader::<_, OrderMap<String, Freq>>(BufReader::new(file))?;
  let raw_unique_words = words.len();
  let raw_weighted_occurrences = words.values().map(|freq| *freq as i128).sum::<i128>();
  let effective_unique_words = words
    .keys()
    .filter(|word| !args.special_tokens.iter().any(|special| special == *word))
    .count();
  let effective_weighted_occurrences = words
    .iter()
    .filter(|(word, _)| !args.special_tokens.iter().any(|special| special == *word))
    .map(|(_, freq)| *freq as i128)
    .sum::<i128>();
  let file_bytes = fs::metadata(&args.words)?.len();
  let config = BpeTrainerConfig {
    initial_alphabet: match args.unit {
      Unit::Byte => InitialAlphabet::ByteLevel,
      Unit::Unicode => InitialAlphabet::RawBytes,
    },
    tie_break: args.tie_break.into(),
    parallel_merge_min_occurs_in: None,
  };
  let analysis = match args.unit {
    Unit::Byte => analyze_byte_words(
      words,
      &args.special_tokens,
      config,
      args.vocab_size,
      &args.window_sizes,
      args.policy.into(),
    ),
    Unit::Unicode => analyze_unicode_words(
      words,
      &args.special_tokens,
      config,
      args.vocab_size,
      &args.window_sizes,
      args.policy.into(),
    ),
  };

  let dataset_name = args
    .dataset_name
    .unwrap_or_else(|| inferred_dataset_name(&args.words));
  let config_name = args
    .config_name
    .unwrap_or_else(|| format!("{}-{}", args.unit.as_str(), args.tie_break.as_str()));
  let experiment_name = args
    .experiment_name
    .unwrap_or_else(|| args.policy.as_str().to_string());
  let output_path = args.json.unwrap_or_else(|| {
    default_output_path(
      &args.json_dir,
      &dataset_name,
      &config_name,
      &experiment_name,
      args.vocab_size,
    )
  });
  let output = Output {
    metadata: Metadata {
      benchmark_contract: "fixed_words_exact_hot_window_simulation_v1",
      script_name: TOOL_NAME,
      unitoken_version: env!("CARGO_PKG_VERSION"),
      dataset_name: &dataset_name,
      config_name: &config_name,
      experiment_name: &experiment_name,
      generated_at: chrono::Utc::now().to_rfc3339(),
      git_sha: git_sha(),
      git_dirty: git_dirty(),
      build_profile,
      rayon_threads: rayon::current_num_threads(),
      notes: vec![
        "Input is a compressed word-frequency inventory; corpus pretokenization is excluded.".to_string(),
        "The exact trainer remains the oracle, so this tool measures a policy without changing model output.".to_string(),
      ],
    },
    source: Source {
      input_kind: "words_json",
      input_order: "json_object_order",
      words: &args.words,
      file_bytes,
      raw_unique_words,
      raw_weighted_occurrences,
      effective_unique_words,
      effective_weighted_occurrences,
      unit: args.unit.as_str(),
      initial_alphabet: match args.unit {
        Unit::Byte => "byte_level",
        Unit::Unicode => "raw_bytes",
      },
      tie_break: args.tie_break.as_str(),
      special_tokens: &args.special_tokens,
    },
    analysis,
  };

  let rendered = serde_json::to_string_pretty(&output)?;
  if !args.quiet {
    println!("{rendered}");
  }
  if let Some(parent) = output_path.parent() {
    fs::create_dir_all(parent)?;
  }
  let mut writer = BufWriter::new(File::create(&output_path)?);
  writer.write_all(rendered.as_bytes())?;
  writer.write_all(b"\n")?;
  writer.flush()?;
  eprintln!("wrote {}", output_path.display());
  Ok(())
}
