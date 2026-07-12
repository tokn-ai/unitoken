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

pub fn run_isolated_cases(requests: &[CaseRequest]) -> Result<Vec<CaseOutcome>, String> {
  let executable = std::env::current_exe().map_err(|error| format!("cannot resolve benchmark executable: {error}"))?;
  let temporary_dir = create_temporary_dir()?;
  let mut outcomes = Vec::with_capacity(requests.len());

  for (index, request) in requests.iter().enumerate() {
    let request_path = temporary_dir.join(format!("request-{index}.json"));
    let result_path = temporary_dir.join(format!("result-{index}.json"));
    write_json(&request_path, request)?;
    let child = Command::new(&executable)
      .arg("case")
      .arg("--request")
      .arg(&request_path)
      .arg("--result")
      .arg(&result_path)
      .output()
      .map_err(|error| format!("cannot launch child {}: {error}", request.id()))?;

    let outcome = match read_json::<CaseOutcome>(&result_path)
      .and_then(|outcome| validate_child_outcome(request, &child.status, outcome))
    {
      Ok(outcome) => outcome,
      Err(read_error) => CaseOutcome::failed(
        request.clone(),
        "child_process",
        child_failure_message(&child.status, &child.stderr, &child.stdout, &read_error),
      ),
    };
    outcomes.push(outcome);
    let _ = fs::remove_file(request_path);
    let _ = fs::remove_file(result_path);
  }

  let _ = fs::remove_dir(temporary_dir);
  Ok(outcomes)
}

pub fn run_child(request_path: &Path, result_path: &Path) -> Result<bool, String> {
  let request = read_json::<CaseRequest>(request_path)?;
  let outcome = execute_case(request);
  let passed = outcome.measurement.is_some();
  write_json_atomic(result_path, &outcome)?;
  Ok(passed)
}

fn create_temporary_dir() -> Result<PathBuf, String> {
  let timestamp = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .unwrap_or_default()
    .as_nanos();
  let base = std::env::temp_dir();
  for suffix in 0..100u32 {
    let path = base.join(format!(
      "unitoken-regression-{}-{timestamp}-{suffix}",
      std::process::id(),
    ));
    match fs::create_dir(&path) {
      Ok(()) => return Ok(path),
      Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
      Err(error) => {
        return Err(format!("cannot create temporary directory {}: {error}", path.display()));
      }
    }
  }
  Err("cannot allocate a unique temporary benchmark directory".to_string())
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

fn validate_child_outcome(
  request: &CaseRequest,
  status: &ExitStatus,
  outcome: CaseOutcome,
) -> Result<CaseOutcome, String> {
  if outcome.request != *request || outcome.case_id != request.id() {
    return Err("child result does not match its request".to_string());
  }
  let shape_is_valid = match outcome.status {
    RunStatus::Completed => outcome.measurement.is_some() && outcome.error.is_none(),
    RunStatus::Failed => outcome.measurement.is_none() && outcome.error.is_some(),
  };
  if !shape_is_valid {
    return Err("child result has inconsistent status, measurement, and error fields".to_string());
  }
  if status.success() != (outcome.status == RunStatus::Completed) {
    return Err(format!(
      "child exit status {status} disagrees with result status {:?}",
      outcome.status,
    ));
  }
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
