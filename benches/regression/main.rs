mod config;
mod codec_bench;
mod fingerprint;
mod process;
mod pretokenizer_bench;
mod report;
mod rss;
mod runner;
mod util;

use std::{
  collections::BTreeSet,
  path::{Path, PathBuf},
  process::Command,
};

use clap::{Args, Parser, Subcommand};
use unitoken::pretokenizer::DEFAULT_EOT;

use crate::{
  config::{CaseConfig, CaseRequest, InitialAlphabetName, OccurrenceVariant, TieBreakName, Unit},
  fingerprint::sha256_hex,
  process::{run_child, run_isolated_cases, write_json_atomic},
  report::{EnvironmentReport, RunStatus, SuiteReport},
  util::{format_bytes, now_seconds, resolve_path_for_comparison},
};

#[derive(Debug, Parser)]
#[command(about = "Rust-only tokenizer regression benchmark")]
struct Cli {
  /// Accepted because `cargo bench` supplies this to harness-free targets.
  #[arg(long, global = true, hide = true)]
  bench: bool,
  #[command(subcommand)]
  command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
enum Commands {
  /// Run the checked-in byte and Unicode smoke inventories.
  Smoke(SuiteOptions),
  /// Run exact and bounded training against one pinned word inventory.
  Trainer(TrainerArgs),
  /// Benchmark raw-corpus Unicode-bigram selection and word counting.
  Pretokenizer(pretokenizer_bench::PretokenizerArgs),
  /// Benchmark cold file encoding and independent decoding with a pinned model.
  Codec(codec_bench::CodecArgs),
  /// Execute one isolated child case. This is an internal protocol.
  #[command(hide = true)]
  Case(ChildArgs),
  /// Execute one isolated pretokenizer case. This is an internal protocol.
  #[command(name = "pretokenizer-case", hide = true)]
  PretokenizerCase(ChildArgs),
  /// Execute one isolated codec phase. This is an internal protocol.
  #[command(name = "codec-case", hide = true)]
  CodecCase(ChildArgs),
}

#[derive(Clone, Debug, Args)]
struct SuiteOptions {
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

#[derive(Clone, Debug, Args)]
struct TrainerArgs {
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

#[derive(Debug, Args)]
struct ChildArgs {
  #[arg(long)]
  request: PathBuf,
  #[arg(long)]
  result: PathBuf,
}

fn main() {
  let cli = Cli::parse();
  let result = match cli.command {
    Some(Commands::Case(args)) => match run_child(&args.request, &args.result) {
      Ok(true) => Ok(()),
      Ok(false) => Err("isolated benchmark case failed".to_string()),
      Err(error) => Err(error),
    },
    Some(Commands::Trainer(args)) => run_trainer_suite(args),
    Some(Commands::Pretokenizer(args)) => pretokenizer_bench::run(args, environment_report()),
    Some(Commands::Codec(args)) => codec_bench::run(args, environment_report()),
    Some(Commands::Smoke(options)) => run_smoke_suite(options),
    Some(Commands::PretokenizerCase(args)) => match pretokenizer_bench::run_child(&args.request, &args.result) {
      Ok(true) => Ok(()),
      Ok(false) => Err("isolated pretokenizer benchmark case failed".to_string()),
      Err(error) => Err(error),
    },
    Some(Commands::CodecCase(args)) => match codec_bench::run_child(&args.request, &args.result) {
      Ok(true) => Ok(()),
      Ok(false) => Err("isolated codec benchmark phase failed".to_string()),
      Err(error) => Err(error),
    },
    None => run_smoke_suite(SuiteOptions::default()),
  };
  if let Err(error) = result {
    eprintln!("error: {error}");
    std::process::exit(1);
  }
}

fn run_smoke_suite(options: SuiteOptions) -> Result<(), String> {
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

fn run_trainer_suite(args: TrainerArgs) -> Result<(), String> {
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

pub(crate) fn resolve_threads(requested: Option<usize>) -> Result<usize, String> {
  let threads = requested.unwrap_or_else(|| std::thread::available_parallelism().map(usize::from).unwrap_or(1));
  if threads == 0 {
    return Err("--rayon-threads must be positive".to_string());
  }
  Ok(threads)
}

pub(crate) fn environment_report() -> EnvironmentReport {
  let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
  let hardware = hardware_info();
  let git_commit = command_output(
    Command::new("git")
      .arg("-C")
      .arg(manifest_dir)
      .args(["rev-parse", "HEAD"]),
  );
  let git_dirty = command_output(
    Command::new("git")
      .arg("-C")
      .arg(manifest_dir)
      .args(["status", "--porcelain"]),
  )
  .map(|output| !output.is_empty());
  let rustc = command_output(Command::new("rustc").arg("--version"));
  let benchmark_binary_sha256 = std::env::current_exe()
    .ok()
    .and_then(|path| std::fs::read(path).ok())
    .map(|bytes| sha256_hex(&bytes));
  EnvironmentReport {
    git_commit,
    git_dirty,
    rustc,
    os: std::env::consts::OS.to_string(),
    arch: std::env::consts::ARCH.to_string(),
    cpu_model: hardware.cpu_model,
    hardware_model: hardware.hardware_model,
    logical_cpus: std::thread::available_parallelism().map(usize::from).unwrap_or(1),
    total_memory_bytes: hardware.total_memory_bytes,
    profile: "bench".to_string(),
    debug_assertions: cfg!(debug_assertions),
    benchmark_binary_sha256,
  }
}

#[derive(Default)]
struct HardwareInfo {
  cpu_model: Option<String>,
  hardware_model: Option<String>,
  total_memory_bytes: Option<u64>,
}

fn hardware_info() -> HardwareInfo {
  #[cfg(target_os = "macos")]
  {
    let profiler = command_output(Command::new("system_profiler").args(["SPHardwareDataType", "-detailLevel", "mini"]));
    let cpu_model = command_output(Command::new("sysctl").args(["-n", "machdep.cpu.brand_string"]))
      .or_else(|| profiler.as_deref().and_then(|output| profiler_value(output, "Chip")));
    let hardware_model = command_output(Command::new("sysctl").args(["-n", "hw.model"])).or_else(|| {
      profiler
        .as_deref()
        .and_then(|output| profiler_value(output, "Model Identifier"))
    });
    let total_memory_bytes = command_output(Command::new("sysctl").args(["-n", "hw.memsize"]))
      .and_then(|value| value.parse().ok())
      .or_else(|| {
        profiler
          .as_deref()
          .and_then(|output| profiler_value(output, "Memory"))
          .and_then(|value| parse_human_memory(&value))
      });
    return HardwareInfo {
      cpu_model,
      hardware_model,
      total_memory_bytes,
    };
  }
  #[cfg(target_os = "linux")]
  {
    let mut cpu_model = None;
    if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
      for expected_key in ["model name", "Hardware", "Processor"] {
        if let Some(value) = cpuinfo.lines().find_map(|line| {
          let (key, value) = line.split_once(':')?;
          (key.trim() == expected_key)
            .then(|| value.trim().to_string())
            .filter(|value| !value.is_empty())
        }) {
          cpu_model = Some(value);
          break;
        }
      }
    }
    let hardware_model = std::fs::read_to_string("/sys/devices/virtual/dmi/id/product_name")
      .ok()
      .map(|value| value.trim().to_string())
      .filter(|value| !value.is_empty());
    let total_memory_bytes = std::fs::read_to_string("/proc/meminfo")
      .ok()
      .and_then(|meminfo| {
        meminfo.lines().find_map(|line| {
          let value = line.strip_prefix("MemTotal:")?.trim();
          value.split_whitespace().next()?.parse::<u64>().ok()
        })
      })
      .and_then(|kilobytes| kilobytes.checked_mul(1024));
    return HardwareInfo {
      cpu_model,
      hardware_model,
      total_memory_bytes,
    };
  }
  #[allow(unreachable_code)]
  HardwareInfo::default()
}

fn profiler_value(output: &str, key: &str) -> Option<String> {
  output.lines().find_map(|line| {
    let (candidate, value) = line.trim().split_once(':')?;
    (candidate == key)
      .then(|| value.trim().to_string())
      .filter(|value| !value.is_empty())
  })
}

fn parse_human_memory(value: &str) -> Option<u64> {
  let mut parts = value.split_whitespace();
  let amount = parts.next()?.parse::<u64>().ok()?;
  let multiplier = match parts.next()? {
    "KB" => 1024u64,
    "MB" => 1024u64.pow(2),
    "GB" => 1024u64.pow(3),
    "TB" => 1024u64.pow(4),
    _ => return None,
  };
  amount.checked_mul(multiplier)
}

fn command_output(command: &mut Command) -> Option<String> {
  let output = command.output().ok()?;
  output
    .status
    .success()
    .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn default_report_path(suite_name: &str, report: &SuiteReport) -> PathBuf {
  default_suite_report_path(suite_name, &report.environment)
}

pub(crate) fn default_suite_report_path(
  report_name: &str,
  environment: &EnvironmentReport,
) -> PathBuf {
  let revision = environment
    .git_commit
    .as_deref()
    .map(|commit| &commit[..commit.len().min(12)])
    .unwrap_or("unknown");
  let dirty = if environment.git_dirty == Some(true) {
    "-dirty"
  } else {
    ""
  };
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("out/benchmarks/regression")
    .join(format!("{revision}{dirty}.{}.json", sanitize_name(report_name)))
}

fn sanitize_name(value: &str) -> String {
  value
    .chars()
    .map(|ch| {
      if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
        ch
      } else {
        '_'
      }
    })
    .collect()
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
