use std::sync::{Arc, Mutex, OnceLock};

use metrics::{Counter, CounterFn, Gauge, GaugeFn, Histogram, HistogramFn, Recorder, SetRecorderError};
use ordermap::OrderMap;

pub struct Frame<T> {
  value: T,
  timestamp: std::time::Instant,
}

impl<T> Frame<T> {
  /// Create a new frame recording `value` at the current timestamp.
  pub fn new(value: T) -> Self {
    Self {
      value,
      timestamp: std::time::Instant::now(),
    }
  }
}

#[derive(Default)]
struct Block<T> {
  frames: Vec<Frame<T>>,
  current: T,
}

struct GlobalStore {
  started_at: std::time::Instant,
  counters: Mutex<OrderMap<String, Block<u64>>>,
  gauges: Mutex<OrderMap<String, Block<f64>>>,
  histograms: Mutex<OrderMap<String, Block<f64>>>,
}

static GLOBAL_STORE: OnceLock<Arc<GlobalStore>> = OnceLock::new();

fn global_store() -> &'static Arc<GlobalStore> {
  GLOBAL_STORE.get_or_init(|| {
    Arc::new(GlobalStore {
      started_at: std::time::Instant::now(),
      counters: Mutex::new(OrderMap::new()),
      gauges: Mutex::new(OrderMap::new()),
      histograms: Mutex::new(OrderMap::new()),
    })
  })
}

struct CounterHandle {
  key: String,
  store: Arc<GlobalStore>,
}

struct GaugeHandle {
  key: String,
  store: Arc<GlobalStore>,
}

struct HistogramHandle {
  key: String,
  store: Arc<GlobalStore>,
}


impl CounterFn for CounterHandle {
  fn increment(&self, value: u64) {
    let mut map = self.store.counters.lock().unwrap();
    let entry = map.entry(self.key.clone()).or_default();
    entry.current += value;
    entry.frames.push(Frame::new(entry.current));
  }

  fn absolute(&self, value: u64) {
    let mut map = self.store.counters.lock().unwrap();
    let entry = map.entry(self.key.clone()).or_default();
    entry.current = value;
    entry.frames.push(Frame::new(entry.current));
  }
}

impl GaugeFn for GaugeHandle {
  fn increment(&self, value: f64) {
    let mut map = self.store.gauges.lock().unwrap();
    let entry = map.entry(self.key.clone()).or_default();
    entry.current += value;
    entry.frames.push(Frame::new(entry.current));
  }

  fn decrement(&self, value: f64) {
    let mut map = self.store.gauges.lock().unwrap();
    let entry = map.entry(self.key.clone()).or_default();
    entry.current -= value;
    entry.frames.push(Frame::new(entry.current));
  }

  fn set(&self, value: f64) {
    let mut map = self.store.gauges.lock().unwrap();
    let entry = map.entry(self.key.clone()).or_default();
    entry.current = value;
    entry.frames.push(Frame::new(entry.current));
  }
}

impl HistogramFn for HistogramHandle {
  fn record(&self, value: f64) {
    let mut map = self.store.histograms.lock().unwrap();
    let entry = map.entry(self.key.clone()).or_default();
    entry.current = value;
    entry.frames.push(Frame::new(entry.current));
  }
}

pub struct MetricsRecorder;
impl Recorder for MetricsRecorder {
  fn describe_counter(&self, _key: metrics::KeyName, _unit: Option<metrics::Unit>, _description: metrics::SharedString) { }

  fn describe_gauge(&self, _key: metrics::KeyName, _unit: Option<metrics::Unit>, _description: metrics::SharedString) { }

  fn describe_histogram(&self, _key: metrics::KeyName, _unit: Option<metrics::Unit>, _description: metrics::SharedString) { }

  fn register_counter(&self, key: &metrics::Key, _metadata: &metrics::Metadata<'_>) -> Counter {
    let store = global_store().clone();
    let handle = CounterHandle {
      key: key.name().to_string(),
      store,
    };
    Counter::from_arc(Arc::new(handle))
  }

  fn register_gauge(&self, key: &metrics::Key, _metadata: &metrics::Metadata<'_>) -> Gauge {
    let store = global_store().clone();
    let handle = GaugeHandle {
      key: key.name().to_string(),
      store,
    };
    Gauge::from_arc(Arc::new(handle))
  }

  fn register_histogram(&self, key: &metrics::Key, _metadata: &metrics::Metadata<'_>) -> Histogram {
    let store = global_store().clone();
    let handle = HistogramHandle {
      key: key.name().to_string(),
      store,
    };
    Histogram::from_arc(Arc::new(handle))
  }
}

/// Install the in-process metrics recorder used by this crate.
///
/// This must be called at most once per process; subsequent calls will error.
pub fn init_metrics() -> Result<(), SetRecorderError<MetricsRecorder>> {
  let recorder = MetricsRecorder;
  metrics::set_global_recorder(recorder)?;
  Ok(())
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct BlockSnapshot<T> {
  pub values: Vec<T>,
  pub timestamps: Vec<f64>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct MetricsSnapshot {
  pub counters: OrderMap<String, BlockSnapshot<u64>>,
  pub gauges: OrderMap<String, BlockSnapshot<f64>>,
  pub histograms: OrderMap<String, BlockSnapshot<f64>>,
}

#[must_use]
/// Capture a snapshot of recorded metrics.
///
/// If `clear` is true, recorded frames are cleared after snapshotting.
pub fn capture_metrics_snapshot(clear: bool) -> MetricsSnapshot {
  let store = global_store();
  fn block_snapshot<T: Copy>(block: &mut Block<T>, started_at: std::time::Instant, clear: bool) -> BlockSnapshot<T> {
    let mut values = Vec::new();
    let mut timestamps = Vec::new();
    for frame in &block.frames {
      values.push(frame.value);
      let elapsed = frame.timestamp.duration_since(started_at).as_secs_f64();
      timestamps.push(elapsed);
    }
    if clear {
      block.frames.clear();
    }
    BlockSnapshot {
      values,
      timestamps,
    }
  }
  let counters = store.counters.lock().unwrap()
    .iter_mut()
    .filter(|(_, block)| !block.frames.is_empty())
    .map(|(key, block)| {
      (key.clone(), block_snapshot(block, store.started_at, clear))
    })
    .collect();
  let gauges = store.gauges.lock().unwrap()
    .iter_mut()
    .filter(|(_, block)| !block.frames.is_empty())
    .map(|(key, block)| {
      (key.clone(), block_snapshot(block, store.started_at, clear))
    })
    .collect();
  let histograms = store.histograms.lock().unwrap()
    .iter_mut()
    .filter(|(_, block)| !block.frames.is_empty())
    .map(|(key, block)| {
      (key.clone(), block_snapshot(block, store.started_at, clear))
    })
    .collect();
  MetricsSnapshot {
    counters,
    gauges,
    histograms,
  }
}
