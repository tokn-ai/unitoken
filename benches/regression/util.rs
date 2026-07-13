use std::{
  fs,
  path::{Path, PathBuf},
  time::{Duration, SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FileIdentity {
  len: u64,
  modified: Option<SystemTime>,
  #[cfg(unix)]
  device: u64,
  #[cfg(unix)]
  inode: u64,
  #[cfg(unix)]
  change_time_seconds: i64,
  #[cfg(unix)]
  change_time_nanoseconds: i64,
}

impl FileIdentity {
  pub fn capture(path: &Path) -> Result<Self, String> {
    let metadata = fs::metadata(path)
      .map_err(|error| format!("cannot inspect {}: {error}", path.display()))?;
    if !metadata.is_file() {
      return Err(format!("{} is not a file", path.display()));
    }
    #[cfg(unix)]
    {
      use std::os::unix::fs::MetadataExt;
      Ok(Self {
        len: metadata.len(),
        modified: metadata.modified().ok(),
        device: metadata.dev(),
        inode: metadata.ino(),
        change_time_seconds: metadata.ctime(),
        change_time_nanoseconds: metadata.ctime_nsec(),
      })
    }
    #[cfg(not(unix))]
    {
      Ok(Self {
        len: metadata.len(),
        modified: metadata.modified().ok(),
      })
    }
  }

  pub fn len(&self) -> u64 {
    self.len
  }

  pub fn ensure_unchanged(&self, path: &Path) -> Result<(), String> {
    let after = Self::capture(path)?;
    if after != *self {
      return Err(format!("{} changed while the benchmark was running", path.display()));
    }
    Ok(())
  }
}

pub fn validate_sha256(field: &str, value: Option<&str>) -> Result<(), String> {
  let Some(value) = value else { return Ok(()) };
  if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
    return Err(format!("{field} must contain 64 hexadecimal characters"));
  }
  Ok(())
}

pub fn file_stem(path: &Path, fallback: &str) -> String {
  path
    .file_stem()
    .and_then(|value| value.to_str())
    .unwrap_or(fallback)
    .to_string()
}

pub fn now_seconds() -> u64 {
  SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .unwrap_or_default()
    .as_secs()
}

pub fn duration_ns(duration: Duration) -> u64 {
  duration.as_nanos().min(u64::MAX as u128) as u64
}

pub fn duration_ms(duration_ns: u64) -> f64 {
  duration_ns as f64 / 1_000_000.0
}

pub fn throughput_mib(bytes: u64, elapsed_ns: u64) -> f64 {
  if elapsed_ns == 0 {
    0.0
  } else {
    bytes as f64 / 1024.0 / 1024.0 / (elapsed_ns as f64 / 1e9)
  }
}

pub fn format_bytes(bytes: Option<u64>) -> String {
  bytes
    .map(|bytes| format!("{:.1} MiB", bytes as f64 / 1024.0 / 1024.0))
    .unwrap_or_else(|| "unsupported".to_string())
}

pub fn short_hash(hash: &str) -> &str {
  &hash[..hash.len().min(12)]
}

pub fn resolve_path_for_comparison(path: &Path) -> Result<PathBuf, String> {
  if path.exists() {
    return fs::canonicalize(path)
      .map_err(|error| format!("cannot resolve {}: {error}", path.display()));
  }
  let parent = path
    .parent()
    .filter(|parent| !parent.as_os_str().is_empty())
    .unwrap_or_else(|| Path::new("."));
  let filename = path
    .file_name()
    .ok_or_else(|| format!("{} has no filename", path.display()))?;
  if let Ok(parent) = fs::canonicalize(parent) {
    return Ok(parent.join(filename));
  }
  let absolute = if path.is_absolute() {
    path.to_path_buf()
  } else {
    std::env::current_dir()
      .map_err(|error| format!("cannot resolve current directory: {error}"))?
      .join(path)
  };
  let mut normalized = PathBuf::new();
  for component in absolute.components() {
    match component {
      std::path::Component::CurDir => {}
      std::path::Component::ParentDir => {
        normalized.pop();
      }
      component => normalized.push(component.as_os_str()),
    }
  }
  Ok(normalized)
}
