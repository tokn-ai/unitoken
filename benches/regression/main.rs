mod codec;
mod common;
mod pretokenizer;
mod trainer;

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

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
  Smoke(trainer::SuiteOptions),
  /// Run exact and bounded training against one pinned word inventory.
  Trainer(trainer::Args),
  /// Benchmark raw-corpus Unicode-bigram selection and word counting.
  Pretokenizer(pretokenizer::Args),
  /// Benchmark cold file encoding and independent decoding with a pinned model.
  Codec(codec::Args),
  /// Execute one isolated trainer case. This is an internal protocol.
  #[command(hide = true)]
  Case(ChildArgs),
  /// Execute one isolated pretokenizer case. This is an internal protocol.
  #[command(name = "pretokenizer-case", hide = true)]
  PretokenizerCase(ChildArgs),
  /// Execute one isolated codec phase. This is an internal protocol.
  #[command(name = "codec-case", hide = true)]
  CodecCase(ChildArgs),
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
    Some(Commands::Case(args)) => child_result(
      trainer::run_child(&args.request, &args.result),
      "isolated trainer benchmark case failed",
    ),
    Some(Commands::Trainer(args)) => trainer::run(args),
    Some(Commands::Pretokenizer(args)) => pretokenizer::run(args),
    Some(Commands::Codec(args)) => codec::run(args),
    Some(Commands::Smoke(options)) => trainer::run_smoke(options),
    Some(Commands::PretokenizerCase(args)) => child_result(
      pretokenizer::run_child(&args.request, &args.result),
      "isolated pretokenizer benchmark case failed",
    ),
    Some(Commands::CodecCase(args)) => child_result(
      codec::run_child(&args.request, &args.result),
      "isolated codec benchmark phase failed",
    ),
    None => trainer::run_smoke(trainer::SuiteOptions::default()),
  };
  if let Err(error) = result {
    eprintln!("error: {error}");
    std::process::exit(1);
  }
}

fn child_result(result: Result<bool, String>, failed_message: &str) -> Result<(), String> {
  match result {
    Ok(true) => Ok(()),
    Ok(false) => Err(failed_message.to_string()),
    Err(error) => Err(error),
  }
}
