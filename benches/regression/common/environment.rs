use std::{
  path::{Path, PathBuf},
  process::Command,
};

use super::{fingerprint::sha256_hex, report::EnvironmentReport};

pub fn resolve_threads(requested: Option<usize>) -> Result<usize, String> {
  let threads = requested.unwrap_or_else(|| std::thread::available_parallelism().map(usize::from).unwrap_or(1));
  if threads == 0 {
    return Err("--rayon-threads must be positive".to_string());
  }
  Ok(threads)
}

pub fn environment_report() -> EnvironmentReport {
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

pub fn default_suite_report_path(
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
