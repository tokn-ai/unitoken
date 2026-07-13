use std::{
  fmt::Display,
  fs,
  time::Instant,
};

use ordermap::OrderMap;
use unitoken::{
  bpe::{
    BpeTrainer, BpeTrainerConfig, CharIdx, CharSplit, CharToIdx, Character, Freq, HasChar, Idx, IdxLike,
    InitialAlphabet, TieBreak, Word, utils::WordDebugExt,
  },
  traits::{CanStrToWord, CanToWord, CanTrain, Train},
};

use crate::{
  config::{CaseRequest, InitialAlphabetName, TieBreakName, Unit},
  fingerprint::{CanonicalId, CanonicalUnit, fingerprint_model, sha256_hex},
  report::{
    CaseMeasurement, CaseOutcome, HotPairWindowReport, InputReport, MemoryReport, StepBucket, TimingReport,
    TrainingCounts,
  },
  rss,
  util::duration_ns,
};

struct LoadedInventory {
  words: OrderMap<String, Freq>,
  input: InputReport,
  load_ns: u64,
  current_rss_bytes: Option<u64>,
  peak_rss_bytes: Option<u64>,
}

#[derive(Debug)]
struct CaseError {
  phase: &'static str,
  message: String,
}

impl CaseError {
  fn new(phase: &'static str, message: impl Into<String>) -> Self {
    Self {
      phase,
      message: message.into(),
    }
  }

  fn from_error(phase: &'static str, error: impl Display) -> Self {
    Self::new(phase, error.to_string())
  }
}

pub fn execute_case(request: CaseRequest) -> CaseOutcome {
  if let Err(error) = request.validate() {
    return CaseOutcome::failed(request, "configuration", error);
  }

  let pool = match rayon::ThreadPoolBuilder::new()
    .num_threads(request.case.rayon_threads)
    .build()
  {
    Ok(pool) => pool,
    Err(error) => {
      return CaseOutcome::failed(request, "rayon_pool", error.to_string());
    }
  };
  match pool.install(|| execute_case_inner(&request)) {
    Ok(measurement) => CaseOutcome::completed(request, measurement),
    Err(error) => CaseOutcome::failed(request, error.phase, error.message),
  }
}

fn execute_case_inner(request: &CaseRequest) -> Result<CaseMeasurement, CaseError> {
  let inventory = load_inventory(request)?;
  match request.case.unit {
    Unit::Byte => run_training::<u8, Idx>(request, inventory),
    Unit::Unicode => run_training::<Character, CharIdx>(request, inventory),
  }
}

fn load_inventory(request: &CaseRequest) -> Result<LoadedInventory, CaseError> {
  let started = Instant::now();
  let bytes = fs::read(&request.case.words_path).map_err(|error| CaseError::from_error("inventory_load", error))?;
  let sha256 = sha256_hex(&bytes);
  if let Some(expected) = request.case.expected_input_sha256.as_ref()
    && !expected.eq_ignore_ascii_case(&sha256)
  {
    return Err(CaseError::new(
      "inventory_validate",
      format!("inventory SHA-256 {sha256} does not match expected {expected}"),
    ));
  }
  let words = serde_json::from_slice::<OrderMap<String, Freq>>(&bytes)
    .map_err(|error| CaseError::from_error("inventory_parse", error))?;
  if words.is_empty() {
    return Err(CaseError::new("inventory_validate", "word inventory is empty"));
  }

  let mut weighted_occurrences = 0u64;
  for (word, frequency) in &words {
    if *frequency <= 0 {
      return Err(CaseError::new(
        "inventory_validate",
        format!("word {word:?} has non-positive frequency {frequency}"),
      ));
    }
    weighted_occurrences = weighted_occurrences
      .checked_add(*frequency as u64)
      .ok_or_else(|| CaseError::new("inventory_validate", "weighted occurrence count overflowed"))?;
  }

  let input = InputReport {
    path: fs::canonicalize(&request.case.words_path).unwrap_or_else(|_| request.case.words_path.clone()),
    bytes: bytes.len() as u64,
    sha256,
    unique_words: words.len(),
    weighted_occurrences,
  };
  Ok(LoadedInventory {
    words,
    input,
    load_ns: duration_ns(started.elapsed()),
    current_rss_bytes: rss::current_rss_bytes(),
    peak_rss_bytes: rss::process_peak_rss_bytes(),
  })
}

fn run_training<C, I>(request: &CaseRequest, inventory: LoadedInventory) -> Result<CaseMeasurement, CaseError>
where
  C: CanStrToWord + CanToWord<u8> + CanonicalUnit + CharSplit + CharToIdx<I> + Clone + Ord + Send + Sync + 'static,
  I: CanonicalId + HasChar<C> + IdxLike,
  Word<C>: WordDebugExt,
  BpeTrainer<C, I>: CanTrain<C, I>,
{
  let LoadedInventory {
    words,
    input,
    load_ns,
    current_rss_bytes: current_after_inventory_load_bytes,
    peak_rss_bytes: peak_after_inventory_load_bytes,
  } = inventory;
  let config = BpeTrainerConfig {
    initial_alphabet: match request.case.initial_alphabet {
      InitialAlphabetName::RawBytes => InitialAlphabet::RawBytes,
      InitialAlphabetName::ByteLevel => InitialAlphabet::ByteLevel,
    },
    tie_break: match request.case.tie_break {
      TieBreakName::SmallestPairId => TieBreak::SmallestPairId,
      TieBreakName::LargestContent => TieBreak::LargestContent,
    },
    parallel_merge_min_occurs_in: request.case.parallel_merge_min_occurs_in,
    hot_pair_window_size: request.variant.hot_pair_window_size,
    bigram_cutoff_freq: request.case.bigram_cutoff_freq,
  };

  let build_rss_sampler = rss::RssSampler::start();
  let started = Instant::now();
  let mut trainer = BpeTrainer::<C, I>::from_words_with_config(words, &request.case.special_tokens, config);
  let build_trainer_ns = duration_ns(started.elapsed());
  let initial_vocab_size = trainer.vocab_size();
  let current_after_trainer_build_bytes = rss::current_rss_bytes();
  let peak_after_trainer_build_bytes = rss::process_peak_rss_bytes();
  let sampled_peak_during_trainer_build_bytes = build_rss_sampler.map(rss::RssSampler::finish);

  let training_rss_sampler = rss::RssSampler::start();
  let started = Instant::now();
  trainer.init_training();
  let init_training_ns = duration_ns(started.elapsed());
  let current_after_init_training_bytes = rss::current_rss_bytes();
  let peak_after_init_training_bytes = rss::process_peak_rss_bytes();
  if let Some(sampler) = training_rss_sampler.as_ref() {
    sampler.observe();
  }

  let mut step_count = 0usize;
  let mut training_steps_ns = 0u64;
  let mut step_buckets = Vec::new();
  while trainer.vocab_size() < request.case.target_vocab_size {
    let start_vocab_size = trainer.vocab_size();
    let started = Instant::now();
    let mut bucket_steps = 0usize;
    while bucket_steps < request.case.bucket_size && trainer.vocab_size() < request.case.target_vocab_size {
      trainer
        .step()
        .map_err(|error| CaseError::from_error("training_steps", error))?;
      bucket_steps += 1;
      step_count += 1;
    }
    let bucket_ns = duration_ns(started.elapsed());
    training_steps_ns = training_steps_ns.saturating_add(bucket_ns);
    step_buckets.push(StepBucket {
      start_vocab_size,
      end_vocab_size: trainer.vocab_size(),
      step_count: bucket_steps,
      duration_ns: bucket_ns,
      current_rss_bytes: rss::current_rss_bytes(),
      process_peak_rss_bytes: rss::process_peak_rss_bytes(),
    });
    if let Some(sampler) = training_rss_sampler.as_ref() {
      sampler.observe();
    }
  }

  let final_vocab_size = trainer.vocab_size();
  if final_vocab_size != request.case.target_vocab_size {
    return Err(CaseError::new(
      "target_vocab",
      format!(
        "training reached vocabulary {final_vocab_size}, expected exactly {}",
        request.case.target_vocab_size,
      ),
    ));
  }
  let current_after_training_bytes = rss::current_rss_bytes();
  let process_peak_rss_through_training_bytes = rss::process_peak_rss_bytes();
  let sampled_peak_during_training_bytes = training_rss_sampler.map(rss::RssSampler::finish);
  let rss_sample_interval_ms = (sampled_peak_during_trainer_build_bytes.is_some()
    || sampled_peak_during_training_bytes.is_some())
  .then_some(rss::SAMPLE_INTERVAL.as_millis() as u64);
  let last_merge_freq = trainer.last_merge_freq();
  let hot_pair_window = trainer.hot_pair_window_stats().map(|stats| HotPairWindowReport {
    hydration_scans: stats.hydration_scans,
    hydrated_word_entries: stats.hydrated_word_entries,
    batch_prunes: stats.batch_prunes,
    prune_evictions: stats.prune_evictions,
    peak_resident_pairs: stats.peak_resident_pairs,
    final_resident_pairs: trainer.hot_resident_pairs(),
    resident_occurrence_capacity_entries: trainer.hot_occurrence_capacity(),
  });
  let counts = TrainingCounts {
    initial_vocab_size,
    final_vocab_size,
    step_count,
    merge_count: trainer.merges.len(),
    last_merge_freq,
  };

  let started = Instant::now();
  let model = trainer.validate_model()
  .map_err(|error| CaseError::from_error("validate_model", error))?;
  let validate_model_ns = duration_ns(started.elapsed());

  let started = Instant::now();
  let fingerprints = fingerprint_model(&model, &trainer.words).map_err(|error| CaseError::new("fingerprint", error))?;
  let fingerprint_ns = duration_ns(started.elapsed());
  let final_merge_at_or_above_bigram_cutoff = request
    .case
    .bigram_cutoff_freq
    .map(|cutoff| last_merge_freq.is_none_or(|frequency| frequency >= cutoff));
  let core_training_ns = build_trainer_ns
    .saturating_add(init_training_ns)
    .saturating_add(training_steps_ns);

  Ok(CaseMeasurement {
    input,
    actual_rayon_threads: rayon::current_num_threads(),
    counts,
    fingerprints,
    timing: TimingReport {
      inventory_load_ns: load_ns,
      build_trainer_ns,
      init_training_ns,
      training_steps_ns,
      validate_model_ns,
      fingerprint_ns,
      core_training_ns,
    },
    memory: MemoryReport {
      current_rss_source: rss::current_rss_source().map(str::to_string),
      peak_rss_source: rss::peak_rss_source().map(str::to_string),
      current_after_inventory_load_bytes,
      current_after_trainer_build_bytes,
      current_after_init_training_bytes,
      current_after_training_bytes,
      peak_after_inventory_load_bytes,
      peak_after_trainer_build_bytes,
      peak_after_init_training_bytes,
      sampled_peak_during_trainer_build_bytes,
      sampled_peak_during_training_bytes,
      rss_sample_interval_ms,
      process_peak_rss_through_training_bytes,
    },
    step_buckets,
    model_valid: true,
    target_vocab_reached: final_vocab_size == request.case.target_vocab_size,
    final_merge_at_or_above_bigram_cutoff,
    hot_pair_window,
  })
}
