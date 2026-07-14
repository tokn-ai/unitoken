#[macro_use]
extern crate tracing;

pub mod bpe;
pub mod bigram;
pub mod counter;
pub mod spec;
pub mod pretokenizer;
pub mod traits;
#[cfg(feature = "py")]
pub mod py;

pub use bigram::Bigram;

#[derive(thiserror::Error, Debug)]
pub enum MyError {
  #[error("IO error: {0}")]
  Io(#[from] std::io::Error),
  #[error("Regex error: {0}")]
  Regex(#[from] fancy_regex::Error),
  #[error("Json error: {0}")]
  Json(#[from] serde_json::Error),
  #[error("Merge txt error: {0} at line {1}")]
  MergeTxt(&'static str, usize),
  #[error("UTF-8 error: {0}")]
  Utf8(#[from] std::str::Utf8Error),
  #[error("Character not in printable set: {0}")]
  InvalidPrintableChar(char),
  #[error("Character not in printable set: {0}")]
  InvalidPrintableEscape(String),
  #[error("Out of vocabulary: {0}")]
  Oov(String),
  #[error("Out of vocabulary idx: {0}")]
  OovIdx(u64),
  #[error("Out of vocabulary bytes: {0}")]
  OovBytes(String),
  #[error("No more step could be performed")]
  TrainStep,
  #[error("Specification error: {0}")]
  SpecError(String),
  #[error("Bpe builder: {0}")]
  BpeBuilder(String),
  #[error("Invalid BPE model: {0}")]
  InvalidBpeModel(String),
  #[error("Frequency overflow")]
  FrequencyOverflow,
  #[error("Source batch error: {0}")]
  SourceBatch(&'static str),
}

pub type MyResult<T> = Result<T, MyError>;
