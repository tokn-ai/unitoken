use std::{
  fs,
  io::Write,
  path::{Path, PathBuf},
  process::{Command, ExitStatus},
  time::{SystemTime, UNIX_EPOCH},
};

use crate::{
  config::CaseRequest,
  report::{CaseOutcome, RunStatus},
  runner::execute_case,
};

pub fn run_isolated_protocol<Request, Outcome>(
  child_command: &str,
  requests: &[Request],
  request_id: impl Fn(&Request) -> String,
  validate: impl Fn(&Request, &ExitStatus, Outcome) -> Result<Outcome, String>,
  failed: impl Fn(Request, String) -> Outcome,
) -> Result<Vec<Outcome>, String>
where
  Request: Clone + serde::Serialize,
  Outcome: serde::de::DeserializeOwned,
{
  let executable = std::env::current_exe().map_err(|error| format!("cannot resolve benchmark executable: {error}"))?;
  let temporary_dir = TemporaryDirectory::create("unitoken-regression")?;
  let mut outcomes = Vec::with_capacity(requests.len());

  for (index, request) in requests.iter().enumerate() {
    let request_path = temporary_dir.path().join(format!("request-{index}.json"));
    let result_path = temporary_dir.path().join(format!("result-{index}.json"));
    if let Err(error) = write_json(&request_path, request) {
      outcomes.push(failed(request.clone(), error));
      continue;
    }
    let child = match Command::new(&executable)
      .arg(child_command)
      .arg("--request")
      .arg(&request_path)
      .arg("--result")
      .arg(&result_path)
      .output()
    {
      Ok(child) => child,
      Err(error) => {
        outcomes.push(failed(
          request.clone(),
          format!("cannot launch child {}: {error}", request_id(request)),
        ));
        continue;
      }
    };

    let outcome = match read_json::<Outcome>(&result_path)
      .and_then(|outcome| validate(request, &child.status, outcome))
    {
      Ok(outcome) => outcome,
      Err(read_error) => failed(
        request.clone(),
        child_failure_message(&child.status, &child.stderr, &child.stdout, &read_error),
      ),
    };
    outcomes.push(outcome);
  }

  Ok(outcomes)
}

pub fn run_protocol_child<Request, Outcome>(
  request_path: &Path,
  result_path: &Path,
  execute: impl FnOnce(Request) -> Outcome,
  passed: impl FnOnce(&Outcome) -> bool,
) -> Result<bool, String>
where
  Request: serde::de::DeserializeOwned,
  Outcome: serde::Serialize,
{
  let request = read_json::<Request>(request_path)?;
  let outcome = execute(request);
  let passed = passed(&outcome);
  write_json_atomic(result_path, &outcome)?;
  Ok(passed)
}

pub fn run_isolated_cases(requests: &[CaseRequest]) -> Result<Vec<CaseOutcome>, String> {
  run_isolated_protocol(
    "case",
    requests,
    CaseRequest::id,
    validate_child_outcome,
    |request, message| CaseOutcome::failed(request, "child_process", message),
  )
}

pub fn run_child(request_path: &Path, result_path: &Path) -> Result<bool, String> {
  run_protocol_child(request_path, result_path, execute_case, |outcome| outcome.measurement.is_some())
}

pub(crate) struct TemporaryDirectory {
  path: PathBuf,
}

impl TemporaryDirectory {
  pub(crate) fn create(prefix: &str) -> Result<Self, String> {
    let timestamp = SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .unwrap_or_default()
      .as_nanos();
    let base = std::env::temp_dir();
    for suffix in 0..100u32 {
      let path = base.join(format!(
        "{prefix}-{}-{timestamp}-{suffix}",
        std::process::id(),
      ));
      match fs::create_dir(&path) {
        Ok(()) => return Ok(Self { path }),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
        Err(error) => {
          return Err(format!("cannot create temporary directory {}: {error}", path.display()));
        }
      }
    }
    Err("cannot allocate a unique temporary benchmark directory".to_string())
  }

  pub(crate) fn path(&self) -> &Path {
    &self.path
  }
}

impl Drop for TemporaryDirectory {
  fn drop(&mut self) {
    let _ = fs::remove_dir_all(&self.path);
  }
}

fn write_json(path: &Path, value: &impl serde::Serialize) -> Result<(), String> {
  let file = fs::File::create(path).map_err(|error| format!("cannot create {}: {error}", path.display()))?;
  serde_json::to_writer(file, value).map_err(|error| format!("cannot serialize {}: {error}", path.display()))
}

pub fn write_json_atomic(path: &Path, value: &impl serde::Serialize) -> Result<(), String> {
  if let Some(parent) = path.parent() {
    fs::create_dir_all(parent).map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
  }
  let temporary_path = path.with_extension(format!("tmp-{}", std::process::id()));
  let mut file = fs::File::create(&temporary_path)
    .map_err(|error| format!("cannot create {}: {error}", temporary_path.display()))?;
  serde_json::to_writer_pretty(&mut file, value)
    .map_err(|error| format!("cannot serialize {}: {error}", temporary_path.display()))?;
  file
    .write_all(b"\n")
    .map_err(|error| format!("cannot finish {}: {error}", temporary_path.display()))?;
  file
    .sync_all()
    .map_err(|error| format!("cannot sync {}: {error}", temporary_path.display()))?;
  fs::rename(&temporary_path, path).map_err(|error| format!("cannot replace {}: {error}", path.display()))
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, String> {
  let file = fs::File::open(path).map_err(|error| format!("cannot open {}: {error}", path.display()))?;
  serde_json::from_reader(file).map_err(|error| format!("cannot parse {}: {error}", path.display()))
}

pub fn validate_outcome_shape(
  status: &ExitStatus,
  completed: bool,
  has_measurement: bool,
  has_error: bool,
) -> Result<(), String> {
  let shape_is_valid = if completed {
    has_measurement && !has_error
  } else {
    !has_measurement && has_error
  };
  if !shape_is_valid {
    return Err("child result has inconsistent status, measurement, and error fields".to_string());
  }
  if status.success() != completed {
    return Err(format!("child exit status {status} disagrees with result status"));
  }
  Ok(())
}

fn validate_child_outcome(
  request: &CaseRequest,
  status: &ExitStatus,
  outcome: CaseOutcome,
) -> Result<CaseOutcome, String> {
  if outcome.request != *request || outcome.case_id != request.id() {
    return Err("child result does not match its request".to_string());
  }
  validate_outcome_shape(
    status,
    outcome.status == RunStatus::Completed,
    outcome.measurement.is_some(),
    outcome.error.is_some(),
  )?;
  Ok(outcome)
}

fn child_failure_message(status: &ExitStatus, stderr: &[u8], stdout: &[u8], read_error: &str) -> String {
  let stderr = String::from_utf8_lossy(stderr).trim().to_string();
  let stdout = String::from_utf8_lossy(stdout).trim().to_string();
  let mut parts = vec![format!("child exited with {status}; {read_error}")];
  if !stderr.is_empty() {
    parts.push(format!("stderr: {stderr}"));
  }
  if !stdout.is_empty() {
    parts.push(format!("stdout: {stdout}"));
  }
  parts.join("; ")
}
