#[macro_use]
extern crate tracing;

use clap::{Parser, Subcommand};
use indicatif::ProgressBar;
use std::{
  collections::BTreeMap, fs, io::BufReader, path::{Path, PathBuf}
};

use unitoken::{
  bpe::{BpeEncoder, BpeTrainer, CharIdx, Character, Idx, encoder::BpeBuilder}, pretokenizer::{PreTokenizer, save_words, sort_words}, spec::{Spec, gpt2::Gpt2Spec, uni::UniSpec}, traits::{CanEncode, CanTrain, Encode, Train}
};

mod _metrics;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
  #[command(subcommand)]
  command: Commands,
  #[arg(short, long, action = clap::ArgAction::Count)]
  verbose: u8,
}

#[derive(Subcommand)]
enum Commands {
  Train(TrainArgs),
  Encode(EncodeArgs),
  #[cfg(feature = "plot")]
  Plot(PlotArgs),
}

impl Commands {
  fn verbose(&self) -> u8 {
    match self {
      Commands::Train(args) => args.verbose,
      Commands::Encode(args) => args.verbose,
      #[cfg(feature = "plot")]
      Commands::Plot(args) => args.verbose,
    }
  }

  fn out_dir(&self) -> &PathBuf {
    match self {
      Commands::Train(args) => &args.out_dir,
      Commands::Encode(args) => &args.out_dir,
      #[cfg(feature = "plot")]
      Commands::Plot(args) => &args.out_dir,
    }
  }

  fn input_file(&self) -> &PathBuf {
    match self {
      Commands::Train(args) => &args.input_file,
      Commands::Encode(args) => &args.input_file,
      #[cfg(feature = "plot")]
      Commands::Plot(args) => &args.input_file,
    }
  }
}

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
pub enum SpecLevel {
  #[clap(name = "u8")]
  U8,
  #[clap(name = "char")]
  Char,
}

impl SpecLevel {
  /// Return the CLI string representation for this character level.
  pub fn as_str(&self) -> &'static str {
    match self {
      Self::U8 => "u8",
      Self::Char => "char",
    }
  }

  /// Default output spec for the given character level.
  pub fn default_spec(&self) -> SpecOutput {
    match self {
      Self::U8 => SpecOutput::Gpt2,
      Self::Char => SpecOutput::Uni,
    }
  }
}

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
pub enum SpecOutput {
  #[clap(name = "gpt2")]
  Gpt2,
  #[clap(name = "uni")]
  Uni,
}

impl SpecOutput {
  /// Return the CLI string representation for this output format.
  pub fn as_str(&self) -> &'static str {
    match self {
      Self::Gpt2 => "gpt2",
      Self::Uni => "uni",
    }
  }

  /// Construct a spec implementation for `u8` token content.
  pub fn get_u8(&self) -> Box<dyn Spec<u8, Idx>> {
    match self {
      Self::Gpt2 => Box::new(Gpt2Spec),
      Self::Uni => Box::new(UniSpec),
    }
  }

  /// Construct a spec implementation for character-level token content.
  pub fn get_char(&self) -> Box<dyn Spec<Character, CharIdx>> {
    match self {
      Self::Gpt2 => unimplemented!(),
      Self::Uni => Box::new(UniSpec),
    }
  }

  /// Construct a spec implementation for character-level content with `Idx` ids.
  pub fn get_char_idx(&self) -> Box<dyn Spec<Character, Idx>> {
    match self {
      Self::Gpt2 => unimplemented!(),
      Self::Uni => Box::new(UniSpec),
    }
  }
}

#[derive(Parser)]
struct TrainArgs {
  #[arg(short, long, action = clap::ArgAction::Count)]
  verbose: u8,
  #[arg(short, long = "out", default_value = "out")]
  out_dir: PathBuf,
  #[arg(short='n', long, default_value = "10000")]
  vocab_size: u32,
  #[arg(long = "chunks", default_value = "1024")]
  num_chunks: u32,
  #[arg(short, long, default_value = "u8")]
  char: SpecLevel,
  #[arg(long = "out-spec")]
  output_spec: Option<SpecOutput>,
  #[arg(long = "vocab")]
  vocab_name: Option<String>,
  #[arg(long = "special-tokens")]
  special_tokens_path: Option<PathBuf>,
  #[arg(value_parser = clap::value_parser!(PathBuf))]
  input_file: PathBuf,
}

#[derive(Parser)]
struct EncodeArgs {
  #[arg(short, long, action = clap::ArgAction::Count)]
  verbose: u8,
  #[arg(short, long = "out", default_value = "out")]
  out_dir: PathBuf,
  #[arg(long = "vocab")]
  vocab_name: Option<String>,
  #[arg(short='n', long)]
  vocab_size: Option<usize>,
  #[arg(long = "chunks", default_value = "1024")]
  num_chunks: u32,
  #[arg(long = "version", default_value = "2")]
  version: u8,
  #[arg(short, long, default_value = "u8")]
  char: SpecLevel,
  #[arg(long = "out-spec")]
  output_spec: Option<SpecOutput>,
  #[arg(long = "special-tokens")]
  special_tokens_path: Option<PathBuf>,
  #[arg(value_parser = clap::value_parser!(PathBuf))]
  input_file: PathBuf,
}

#[derive(Parser)]
struct PlotArgs {
  #[arg(short, long, action = clap::ArgAction::Count)]
  verbose: u8,
  #[arg(short, long = "out", default_value = "out")]
  out_dir: PathBuf,
  #[arg(value_parser = clap::value_parser!(PathBuf))]
  input_file: PathBuf,
}

fn _pretokenize<P1: AsRef<Path>, P2: AsRef<Path>>(output: P1, input: P2, num_chunks: usize, special_tokens: Vec<String>) -> BTreeMap<String, i64> {
  if output.as_ref().exists() {
    info!("pretokenize file already exists, loading from {}", output.as_ref().display());
    let buffered = BufReader::new(fs::File::open(output).expect("open _words file"));
    let result = serde_json::from_reader(buffered).expect("serde_json _words file");
    return result;
  }
  let split_special_token = special_tokens.get(0).cloned();
  let pre = PreTokenizer::new(&special_tokens, split_special_token.as_deref());

  let words = pre.get_words_from_file(&input, num_chunks).unwrap();

  debug!("Sort words");
  let sorted_words = sort_words(&words);
  debug!("Save words to {}", output.as_ref().display());
  let words_file = fs::File::create(output).unwrap();
  save_words(std::io::BufWriter::new(words_file), &sorted_words).unwrap();
  words
}


/// Train a [`BpeTrainer`] to the requested `vocab_size`.
///
/// This is a CLI helper used by the `train` subcommand.
pub fn _bpe_train<C, I>(
  words: BTreeMap<String, i64>, vocab_size: u32, special_tokens: &Vec<String>,
) -> BpeTrainer<C, I>
where
  BpeTrainer<C, I>: CanTrain<C, I>,
{
  let mut bpe = BpeTrainer::<C, I>::from_words(words, special_tokens);
  let start_vocab_idx = bpe.vocab.len();
  bpe.init_training();

  let bar = ProgressBar::new(vocab_size as u64);
  bar.set_position(start_vocab_idx as u64);
  for i in start_vocab_idx..vocab_size as usize {
    if bpe.step().is_err() {
      warn!(vocab_size=i, "No more merges can be made, stopping training early");
      break;
    }
    bar.inc(1);
  }
  bar.finish();
  bpe._metrics();
  bpe
}

/// Save a trained model as `vocab.{name}.json` and `merges.{name}.txt` in `out_dir`.
///
/// This is a CLI helper used by the `train` subcommand.
pub fn _bpe_save_train<C, I>(
  bpe: &BpeTrainer<C, I>,
  spec: &dyn Spec<C, I>,
  out_dir: &std::path::Path,
  name: &str,
) where
  BpeTrainer<C, I>: CanTrain<C, I>,
{
  let vocab_filename = format!("vocab.{name}.json");
  let merges_filename = format!("merges.{name}.txt");

  let vocab_file = std::fs::File::create(out_dir.join(vocab_filename)).unwrap();
  let merges_file = std::fs::File::create(out_dir.join(merges_filename)).unwrap();

  bpe.save_vocab_json(spec, vocab_file).unwrap();
  bpe.save_merges_txt(spec, merges_file).unwrap();
}

pub struct BpeTrainParams {
  pub input_path: PathBuf,
  pub vocab_size: u32,
  pub num_chunks: u32,
  pub special_tokens: Vec<String>,
  pub out_dir: PathBuf,
  pub char_level: SpecLevel,
  pub output_spec: SpecOutput,
  pub vocab_name: String,
}

fn bpe_train(BpeTrainParams{
  input_path,
  vocab_size,
  num_chunks,
  special_tokens,
  out_dir,
  char_level: spec,
  output_spec,
  vocab_name,
}: BpeTrainParams) {
  fs::create_dir_all(&out_dir).expect("Failed to create output directory");

  let file_stem = input_path
    .file_stem()
    .expect("Failed to get file stem")
    .to_str()
    .expect("Failed to convert file stem to str");
  // use first special_token as split_special_token

  info!("Pretokenizing input file...");
  let words = _pretokenize(
    out_dir.join(format!("_words.{file_stem}.json")),
    &input_path,
    num_chunks as _,
    special_tokens.clone(),
  );

  match spec {
    SpecLevel::U8 => {
      info!("Using GPT-2 BPE specification");

      info!("Training BPE model...");
      let bpe = _bpe_train::<u8, Idx>(words, vocab_size, &special_tokens);

      info!("Saving BPE model...");
      _bpe_save_train(&bpe, output_spec.get_u8().as_ref(), &out_dir, &vocab_name);
    }
    SpecLevel::Char => {
      info!("Using Uni BPE specification");

      info!("Training BPE model...");
      let bpe = _bpe_train::<Character, CharIdx>(words, vocab_size, &special_tokens);

      info!("Saving BPE model...");
      _bpe_save_train(&bpe, output_spec.get_char().as_ref(), &out_dir, &vocab_name);
    }
  }
}

pub struct BpeEncoderParams {
  pub input_path: PathBuf,
  pub vocab_path: PathBuf,
  pub merges_path: PathBuf,
  pub special_tokens: Option<Vec<String>>,
  pub num_chunks: u32,
  pub output_path: PathBuf,
  pub version: u8,
  pub vocab_size: Option<usize>,
}

fn bpe_encode<C>(BpeEncoderParams {
  input_path,
  vocab_path,
  merges_path,
  special_tokens,
  num_chunks,
  output_path,
  version,
  vocab_size,
}: BpeEncoderParams, spec: &dyn Spec<C, Idx>)
where
  BpeEncoder<C>: CanEncode<C, Idx>,
{
  info!("Initializing BPE encoder...");
  let bpe = BpeBuilder::new()
    .load_merges_file(merges_path, spec).unwrap()
    .load_vocab_file(vocab_path, spec).unwrap()
    .set_special_tokens(special_tokens)
    .set_vocab_size(vocab_size)
    .build(spec).unwrap();
  // (spec, vocab_path, merges_path, special_tokens, vocab_size).expect("create bpe encoder");

  info!("Encoding file: {}", input_path.display());
  let idxs = match version {
    2 => bpe.encode_file(input_path.as_ref(), num_chunks as _).expect("encode file v2"),
    _ => {
      #[allow(deprecated)]
      bpe.encode_file_with_cache(&input_path, num_chunks as _).expect("encode file")
    },
  };

  info!("Encoded idxs count: {}", idxs.len());
  info!("Saving BPE idxs... to {}", output_path.display());
  bpe.save_idxs_npy(output_path, idxs).expect("save idxs");
}



fn lines_of(s: &str) -> Vec<String> {
  s.lines().filter(|line| !line.is_empty()).map(|line| line.to_string()).collect()
}

fn run_train(args: TrainArgs) {
  let special_tokens = if let Some(special_tokens_path) = args.special_tokens_path {
    let content = fs::read_to_string(special_tokens_path).expect("Failed to read special tokens file");
    lines_of(&content)
  } else {
    lines_of(include_str!("../fixtures/default_special_tokens.txt"))
  };
  let output_spec = args.output_spec.unwrap_or(args.char.default_spec());

  let vocab_name = format!("{}[{}]", args.input_file.file_stem().unwrap().display(), args.char.as_str());
  let params = BpeTrainParams {
    input_path: args.input_file,
    vocab_size: args.vocab_size,
    num_chunks: args.num_chunks,
    special_tokens,
    out_dir: args.out_dir,
    char_level: args.char,
    output_spec,
    vocab_name,
  };

  debug!("Char Level: {:?}", params.char_level.as_str());
  debug!("Output spec: {:?}", params.output_spec.as_str());
  debug!("Special tokens: {:?}", params.special_tokens);
  debug!("Vocabulary size: {}", params.vocab_size);
  debug!("Number of chunks: {}", params.num_chunks);
  debug!("Input file: {}", params.input_path.display());
  debug!("Output directory: {}", params.out_dir.display());
  bpe_train(params);
}

fn run_encode(args: EncodeArgs) {
  let file_stem = args.input_file
    .file_stem()
    .expect("Failed to get file stem")
    .to_str()
    .expect("Failed to convert file stem to str");

  let vocab_name = args.vocab_name.unwrap_or(file_stem.to_string());

  let out_spec = args.output_spec.unwrap_or(args.char.default_spec());
  let char_level = args.char.as_str();
  let vocab_file = args.out_dir.join(format!("vocab.{vocab_name}[{char_level}].json"));
  let merges_file = args.out_dir.join(format!("merges.{vocab_name}[{char_level}].txt"));
  let out_file = args.out_dir.join(format!("idxs.{file_stem}.npy"));

  let special_tokens = if let Some(special_tokens_path) = args.special_tokens_path {
    let content = fs::read_to_string(special_tokens_path).expect("Failed to read special tokens file");
    Some(lines_of(&content))
  } else {
    None
  };

  let params = BpeEncoderParams {
    input_path: args.input_file,
    vocab_path: vocab_file,
    merges_path: merges_file,
    special_tokens,
    num_chunks: args.num_chunks,
    output_path: out_file,
    version: args.version,
    vocab_size: args.vocab_size,
  };

  debug!("Version: {}", params.version);
  debug!("Input file: {}", params.input_path.display());
  debug!("Vocabulary file: {}", params.vocab_path.display());
  debug!("Merges file: {}", params.merges_path.display());
  debug!("Output file: {}", params.output_path.display());
  debug!("Number of chunks: {}", params.num_chunks);
  debug!("Special tokens: {:?}", params.special_tokens);

  // TODO read special tokens from vocab file
  match args.char {
    SpecLevel::U8 => {
      info!("Using GPT-2 BPE specification");
      bpe_encode::<u8>(
        params,
        out_spec.get_u8().as_ref(),
      );
      return;
    }
    SpecLevel::Char => {
      info!("Using Uni BPE specification");

      bpe_encode::<Character>(
        params,
        out_spec.get_char_idx().as_ref(),
      );
    }
  }
}

#[hotpath::main(percentiles = [99])]
fn main() {
  let cli = Cli::parse();
  let verbose = cli.verbose + cli.command.verbose();
  match verbose {
    0 => tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).init(),
    1 => tracing_subscriber::fmt().with_max_level(tracing::Level::DEBUG).init(),
    _ => tracing_subscriber::fmt().with_max_level(tracing::Level::TRACE).init(),
  }
  let metrics_dir =  cli.command.out_dir().join(".metrics");
  let name = cli.command.input_file().file_name()
    .and_then(|n| n.to_str())
    .unwrap_or("noname")
    .to_string();

  std::thread::spawn({
    let metrics_snapshot_file = metrics_dir.join(format!("metrics_snapshot-[{}]-tmp.json", name));
    move || loop {
      std::thread::sleep(std::time::Duration::from_secs(30));
      let snapshot = _metrics::capture_metrics_snapshot(false);
      let Ok(file) = std::fs::File::create(&metrics_snapshot_file) else {
        warn!("Failed to create metrics snapshot file: {}", metrics_snapshot_file.display());
        continue;
      };
      serde_json::to_writer_pretty(
        std::io::BufWriter::new(file),
        &snapshot,
      ).ok();
    }
  });
  debug!("Verbosity level: {}", verbose);
  match cli.command {
    Commands::Train(train_args) => {
      _metrics::init_metrics().expect("Failed to initialize metrics recorder");
      run_train(train_args);
    }
    Commands::Encode(encode_args) => {
      _metrics::init_metrics().expect("Failed to initialize metrics recorder");
      run_encode(encode_args);
    }

    #[cfg(feature = "plot")]
    Commands::Plot(plot_args) => {
      debug!("Plotting metrics...");
      let input_file = plot_args.input_file;
      if !input_file.exists() {
        error!("Input file does not exist: {}", input_file.display());
        return;
      }
      let metrics_snapshot: _metrics::MetricsSnapshot = serde_json::from_reader(fs::File::open(input_file).expect("Failed to open input file")).expect("Failed to parse metrics snapshot");
      plot_metrics(&metrics_snapshot);
      return;
    }
  }
  info!("Done!");
  debug!("Capturing metrics snapshot...");
  let snapshot = _metrics::capture_metrics_snapshot(true);
  fs::create_dir_all(&metrics_dir).expect("Failed to create metrics directory");
  let metrics_snapshot_file = metrics_dir.join(format!("metrics_snapshot-[{}]-{}.json", name, chrono::Utc::now().timestamp_millis()));
  serde_json::to_writer_pretty(
    std::fs::File::create(&metrics_snapshot_file).expect("Failed to create metrics snapshot file"),
    &snapshot,
  ).ok();
  debug!("Metrics snapshot saved to {}", metrics_snapshot_file.display());
  #[cfg(feature = "plot")]
  plot_metrics(&snapshot);
}

#[cfg(feature = "plot")]
fn plot_metrics(metrics: &_metrics::MetricsSnapshot) {
  use textplots::*;
  use rgb::Rgb;
  for (name, block) in &metrics.gauges {
    let data = block.timestamps.iter().zip(&block.values).map(|(i, v)| (*i as f32, *v as f32)).collect::<Vec<_>>();
    if data.is_empty() {
      continue;
    }
    let x_max = data.last().unwrap().0 + 0.01;
    let x_min = data.first().unwrap().0 - 0.01;
    println!("{} [{}] {:?}", name, data.len(), data.first());
    let rgb = Rgb::new(255, 255, 0);
    Chart::new(120, 30, x_min, x_max)
      .linecolorplot(&Shape::Lines(&data), rgb)
      .display();
  }
  for (name, block) in &metrics.counters {
    let data = block.timestamps.iter().zip(&block.values).map(|(i, v)| (*i as f32, *v as f32)).collect::<Vec<_>>();
    if data.is_empty() {
      continue;
    }
    let x_max = data.last().unwrap().0 + 0.01;
    let x_min = data.first().unwrap().0 - 0.01;
    println!("{} [{}] {:?}", name, data.len(), data.first());
    let rgb = Rgb::new(255, 255, 0);
    Chart::new(120, 30, x_min, x_max)
      .linecolorplot(&Shape::Lines(&data), rgb)
      .display();
  }
  for (name, block) in &metrics.histograms {
    let data = block.timestamps.iter().zip(&block.values).map(|(i, v)| (*i as f32, *v as f32)).collect::<Vec<_>>();
    if data.is_empty() {
      continue;
    }
    let y_mean = data.iter().map(|(_, v)| *v).sum::<f32>() / (data.len() as f32);
    let y_min_true = data.iter().map(|(_, v)| *v).fold(f32::INFINITY, f32::min);
    let y_max_true = data.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max) + 1e-6;
    let y_delta = f32::min(y_mean - y_min_true, y_max_true - y_mean);
    let y_min = f32::max(y_min_true, y_mean - y_delta);
    let y_max = f32::min(y_max_true, y_mean + y_delta);
    let mut bin_y = vec![0.0; 50];
    let bin_x = bin_y.iter().enumerate().map(|(i, _)| {
      let bin_center = y_min + (i as f32 + 0.5) / (bin_y.len() as f32) * (y_max - y_min);
      bin_center
    }).collect::<Vec<_>>();
    data.iter().for_each(|&(_, i)| {
      let bin_idx = ((i - y_min) / (y_max - y_min) * (bin_y.len() as f32)) as usize;
      if bin_idx < bin_y.len() {
        bin_y[bin_idx] += 1.0;
      }
    });
    println!("{} [{}] {:?}", name, data.len(), (y_min_true, y_mean, y_max_true));
    Chart::new(120, 30, y_min, y_max)
      .lineplot(&Shape::Bars(&bin_x.into_iter().zip(bin_y).collect::<Vec<_>>()))
      .display();
  }
}
