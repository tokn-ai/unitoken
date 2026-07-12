#[cfg(target_os = "linux")]
use std::fs;
use std::{
  sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
  },
  thread,
  time::Duration,
};

pub const SAMPLE_INTERVAL: Duration = Duration::from_millis(5);

pub struct RssSampler {
  stop: Arc<AtomicBool>,
  peak: Arc<AtomicU64>,
  thread: Option<thread::JoinHandle<()>>,
}

impl RssSampler {
  pub fn start() -> Option<Self> {
    let initial = current_rss_bytes()?;
    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(initial));
    let thread_stop = Arc::clone(&stop);
    let thread_peak = Arc::clone(&peak);
    let thread = thread::spawn(move || {
      while !thread_stop.load(Ordering::Relaxed) {
        if let Some(current) = current_rss_bytes() {
          thread_peak.fetch_max(current, Ordering::Relaxed);
        }
        thread::park_timeout(SAMPLE_INTERVAL);
      }
    });
    Some(Self {
      stop,
      peak,
      thread: Some(thread),
    })
  }

  pub fn observe(&self) {
    if let Some(current) = current_rss_bytes() {
      self.peak.fetch_max(current, Ordering::Relaxed);
    }
  }

  pub fn finish(mut self) -> u64 {
    self.observe();
    self.stop_and_join();
    self.peak.load(Ordering::Relaxed)
  }

  fn stop_and_join(&mut self) {
    self.stop.store(true, Ordering::Relaxed);
    if let Some(thread) = self.thread.take() {
      thread.thread().unpark();
      let _ = thread.join();
    }
  }
}

impl Drop for RssSampler {
  fn drop(&mut self) {
    self.stop_and_join();
  }
}

pub fn process_peak_rss_bytes() -> Option<u64> {
  process_peak_rss_bytes_impl()
}

pub fn current_rss_bytes() -> Option<u64> {
  current_rss_bytes_impl()
}

pub fn peak_rss_source() -> Option<&'static str> {
  cfg!(unix).then_some("getrusage_ru_maxrss")
}

pub fn current_rss_source() -> Option<&'static str> {
  if cfg!(target_os = "linux") {
    Some("proc_self_statm")
  } else if cfg!(target_os = "macos") {
    Some("mach_task_basic_info")
  } else {
    None
  }
}

#[cfg(unix)]
fn process_peak_rss_bytes_impl() -> Option<u64> {
  let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
  // SAFETY: getrusage initializes the supplied rusage value on success.
  let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
  if status != 0 {
    return None;
  }
  // SAFETY: the successful getrusage call initialized usage.
  let peak = unsafe { usage.assume_init() }.ru_maxrss;
  if peak < 0 {
    return None;
  }

  #[cfg(target_os = "macos")]
  {
    Some(peak as u64)
  }
  #[cfg(not(target_os = "macos"))]
  {
    (peak as u64).checked_mul(1024)
  }
}

#[cfg(not(unix))]
fn process_peak_rss_bytes_impl() -> Option<u64> {
  None
}

#[cfg(target_os = "linux")]
fn current_rss_bytes_impl() -> Option<u64> {
  let resident_pages = fs::read_to_string("/proc/self/statm")
    .ok()?
    .split_whitespace()
    .nth(1)?
    .parse::<u64>()
    .ok()?;
  // SAFETY: sysconf has no memory-safety preconditions.
  let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
  if page_size <= 0 {
    return None;
  }
  resident_pages.checked_mul(page_size as u64)
}

#[cfg(target_os = "macos")]
fn current_rss_bytes_impl() -> Option<u64> {
  const MACH_TASK_BASIC_INFO: i32 = 20;

  #[repr(C)]
  struct TimeValue {
    seconds: i32,
    microseconds: i32,
  }

  #[repr(C)]
  struct MachTaskBasicInfo {
    virtual_size: u64,
    resident_size: u64,
    resident_size_max: u64,
    user_time: TimeValue,
    system_time: TimeValue,
    policy: i32,
    suspend_count: i32,
  }

  unsafe extern "C" {
    fn mach_task_self() -> u32;
    fn task_info(target_task: u32, flavor: i32, task_info_out: *mut i32, task_info_count: *mut u32) -> i32;
  }

  let mut info = std::mem::MaybeUninit::<MachTaskBasicInfo>::zeroed();
  let mut count = (std::mem::size_of::<MachTaskBasicInfo>() / std::mem::size_of::<i32>()) as u32;
  // SAFETY: info points to writable storage of the size described by count.
  let status = unsafe {
    task_info(
      mach_task_self(),
      MACH_TASK_BASIC_INFO,
      info.as_mut_ptr().cast::<i32>(),
      &mut count,
    )
  };
  if status != 0 {
    return None;
  }
  // SAFETY: task_info initialized info on success.
  Some(unsafe { info.assume_init() }.resident_size)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn current_rss_bytes_impl() -> Option<u64> {
  None
}
