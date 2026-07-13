use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
  Completed,
  Failed,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RunFailure {
  pub phase: String,
  pub message: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct EnvironmentReport {
  pub git_commit: Option<String>,
  pub git_dirty: Option<bool>,
  pub rustc: Option<String>,
  pub os: String,
  pub arch: String,
  pub cpu_model: Option<String>,
  pub hardware_model: Option<String>,
  pub logical_cpus: usize,
  pub total_memory_bytes: Option<u64>,
  pub profile: String,
  pub debug_assertions: bool,
  pub benchmark_binary_sha256: Option<String>,
}
