use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use unitoken::pretokenizer::UnicodeBigramMixedBoundary;

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
pub enum UnicodeBigramMixedBoundaryName {
  Keep,
  Split,
}

impl UnicodeBigramMixedBoundaryName {
  pub fn core(self) -> UnicodeBigramMixedBoundary {
    match self {
      Self::Keep => UnicodeBigramMixedBoundary::Keep,
      Self::Split => UnicodeBigramMixedBoundary::Split,
    }
  }

  pub fn as_str(self) -> &'static str {
    match self {
      Self::Keep => "keep",
      Self::Split => "split",
    }
  }
}
