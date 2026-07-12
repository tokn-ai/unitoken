use std::{collections::BTreeSet, path::PathBuf};

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
#[value(rename_all = "snake_case")]
pub enum Unit {
  Byte,
  Unicode,
}

impl Unit {
  pub fn as_str(self) -> &'static str {
    match self {
      Self::Byte => "byte",
      Self::Unicode => "unicode",
    }
  }
}

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
