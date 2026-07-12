//! Exact-oracle simulations for evaluating bounded occurrence windows.
//!
//! This module is feature-gated because it exposes trainer internals for
//! benchmarking, not a supported training policy.

use std::{hash::Hash, mem::size_of, time::Instant};

use ahash::{AHashMap, AHashSet};
use ordermap::OrderMap;
use serde::Serialize;

use crate::{
  bpe::{
    utils::WordDebugExt, BpeTrainer, BpeTrainerConfig, CharIdx, Character, Freq,
    HasChar, Idx, IdxLike, Merge, MergeData, Word,
  },
  traits::{CanStrToWord, CanToWord},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotWindowPolicy {
  ReplaceTopK,
  ThresholdNoEvict,
}

impl HotWindowPolicy {
  pub fn as_str(self) -> &'static str {
    match self {
      Self::ReplaceTopK => "replace_top_k_on_cold_winner",
      Self::ThresholdNoEvict => "threshold_admission_without_eviction",
    }
  }
}

#[derive(Debug, Clone, Serialize)]
pub struct DistributionSummary {
  pub count: usize,
  pub min: Option<u64>,
  pub p50: Option<u64>,
  pub p90: Option<u64>,
  pub p99: Option<u64>,
  pub max: Option<u64>,
  pub mean: Option<f64>,
}

impl DistributionSummary {
  fn from_samples(mut samples: Vec<u64>) -> Self {
    if samples.is_empty() {
      return Self {
        count: 0,
        min: None,
        p50: None,
        p90: None,
        p99: None,
        max: None,
        mean: None,
      };
    }

    samples.sort_unstable();
    let sum = samples.iter().map(|value| *value as f64).sum::<f64>();
    Self {
      count: samples.len(),
      min: samples.first().copied(),
      p50: Some(percentile(&samples, 50)),
      p90: Some(percentile(&samples, 90)),
      p99: Some(percentile(&samples, 99)),
      max: samples.last().copied(),
      mean: Some(sum / samples.len() as f64),
    }
  }
}

fn percentile(sorted: &[u64], percentile: usize) -> u64 {
  let index = (sorted.len().saturating_sub(1) * percentile).div_ceil(100);
  sorted[index]
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct OraclePostingSnapshot {
  pub stored_pair_count: u64,
  pub positive_pair_count: u64,
  pub nonpositive_pair_count: u64,
  pub occurrence_len: u64,
  pub occurrence_capacity: u64,
  /// Payload only: two ids and one frequency per pair. Hash-table and token
  /// content overhead are deliberately excluded.
  pub pair_frequency_payload_bytes: u64,
  /// Payload only: allocated `u64` slots in all occurrence sets.
  pub occurrence_payload_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct HotWindowResult {
  pub window_size: usize,
  pub pair_winners: u64,
  pub hot_winners: u64,
  pub cold_winners: u64,
  pub cold_winner_rate: f64,
  pub hydration_scans: u64,
  pub hydration_seconds: f64,
  pub scanned_word_entries: u64,
  pub scanned_encoded_units: u64,
  pub scanned_adjacent_positions: u64,
  pub promotions: u64,
  pub refill_admissions: u64,
  pub re_promotions: u64,
  pub evictions: u64,
  pub fresh_target_positive_candidates: u64,
  pub dynamic_admissions: u64,
  pub threshold_rejections: u64,
  pub cutoff_tie_refills: u64,
  pub max_cutoff_tie_width: u64,
  pub merges_per_refill: DistributionSummary,
  pub peak_resident_pairs: u64,
  pub peak_hot_occurrence_len: u64,
  pub peak_hot_occurrence_capacity: u64,
  pub peak_hot_occurrence_payload_bytes: u64,
  pub peak_resident_occurrence_set_structural_bytes: u64,
  pub peak_resident_to_window_ratio: f64,
  pub final_resident_pairs: u64,
  pub final_hot_occurrence_len: u64,
  pub final_hot_occurrence_capacity: u64,
  pub final_admission_threshold: Option<Freq>,
  pub peak_oracle_stored_pair_count: u64,
  pub peak_oracle_positive_pair_count: u64,
  pub peak_oracle_occurrence_len: u64,
  pub peak_oracle_occurrence_capacity: u64,
  pub peak_oracle_occurrence_payload_bytes: u64,
  /// Ratio of the two independently observed capacity peaks. This is a
  /// directional storage estimate, not an allocator-RSS measurement.
  pub peak_hot_to_oracle_occurrence_capacity_ratio: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct HotWindowAnalysisReport {
  pub policy: String,
  pub target_vocab_size: usize,
  pub initial_vocab_size: usize,
  pub final_vocab_size: usize,
  pub completed: bool,
  pub initial_unit_steps: u64,
  pub pair_merge_steps: u64,
  pub init_training_seconds: f64,
  /// Wall time for exact merge execution plus all simulated windows.
  pub simulation_merge_seconds: f64,
  /// Shared across the K sweep: one exact-heap inspection can serve every
  /// window that encounters the same cold winner.
  pub shared_refill_context_builds: u64,
  pub shared_refill_context_seconds: f64,
  pub initial_oracle: OraclePostingSnapshot,
  /// Component-wise peaks observed after every exact merge-map update.
  pub peak_oracle: OraclePostingSnapshot,
  pub final_oracle: OraclePostingSnapshot,
  pub windows: Vec<HotWindowResult>,
  pub notes: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
struct RankedPair<I> {
  tp: (I, I),
  freq: Freq,
}

struct RefillContext<I> {
  ranked: Vec<RankedPair<I>>,
  frequency_counts: AHashMap<Freq, u64>,
}

#[derive(Debug, Clone, Copy)]
struct PairStorage {
  positive: bool,
  occurrence_len: u64,
  occurrence_capacity: u64,
}

impl PairStorage {
  fn from_merge<C, I>(merge: &Merge<C, I>) -> Option<Self> {
    if merge.target.is_some() {
      return None;
    }
    Some(Self {
      positive: merge.data.freq > 0,
      occurrence_len: merge.data.occurs_in.len() as u64,
      occurrence_capacity: merge.data.occurs_in.capacity() as u64,
    })
  }
}

struct PendingOracleUpdates<I> {
  entries: Vec<((I, I), Option<PairStorage>)>,
}

impl<I> PendingOracleUpdates<I>
where
  I: Copy + Eq + Hash + Ord,
{
  fn prepare<C>(
    trainer: &BpeTrainer<C, I>,
    selected: (I, I),
    changes: &AHashMap<(I, I), MergeData>,
  ) -> Self {
    let entries = changes
      .iter()
      .filter(|(tp, data)| **tp != selected && data.freq != 0)
      .map(|(tp, _)| {
        (
          *tp,
          trainer.pre_merges.get(tp).and_then(PairStorage::from_merge),
        )
      })
      .collect();
    Self {
      entries,
    }
  }

  fn apply<C>(self, trainer: &BpeTrainer<C, I>, accounting: &mut OracleAccounting) {
    for (tp, old) in self.entries {
      let new = trainer.pre_merges.get(&tp).and_then(PairStorage::from_merge);
      accounting.replace_pair::<I>(old, new);
    }
    accounting.observe_peak::<I>();
  }
}

struct OracleAccounting {
  current: OraclePostingSnapshot,
  peak: OraclePostingSnapshot,
}

impl OracleAccounting {
  fn new(initial: OraclePostingSnapshot) -> Self {
    Self {
      current: initial.clone(),
      peak: initial,
    }
  }

  fn remove_selected<C, I>(&mut self, selected: &Merge<C, I>) {
    self.replace_pair::<I>(PairStorage::from_merge(selected), None);
  }

  fn replace_pair<I>(&mut self, old: Option<PairStorage>, new: Option<PairStorage>) {
    if let Some(old) = old {
      self.current.stored_pair_count = self.current.stored_pair_count.saturating_sub(1);
      if old.positive {
        self.current.positive_pair_count = self.current.positive_pair_count.saturating_sub(1);
      } else {
        self.current.nonpositive_pair_count = self.current.nonpositive_pair_count.saturating_sub(1);
      }
      self.current.occurrence_len = self.current.occurrence_len.saturating_sub(old.occurrence_len);
      self.current.occurrence_capacity = self
        .current
        .occurrence_capacity
        .saturating_sub(old.occurrence_capacity);
    }
    if let Some(new) = new {
      self.current.stored_pair_count += 1;
      if new.positive {
        self.current.positive_pair_count += 1;
      } else {
        self.current.nonpositive_pair_count += 1;
      }
      self.current.occurrence_len = self.current.occurrence_len.saturating_add(new.occurrence_len);
      self.current.occurrence_capacity = self
        .current
        .occurrence_capacity
        .saturating_add(new.occurrence_capacity);
    }
    refresh_payload_bytes::<I>(&mut self.current);
  }

  fn observe_peak<I>(&mut self) {
    self.peak.stored_pair_count = self.peak.stored_pair_count.max(self.current.stored_pair_count);
    self.peak.positive_pair_count = self.peak.positive_pair_count.max(self.current.positive_pair_count);
    self.peak.nonpositive_pair_count = self
      .peak
      .nonpositive_pair_count
      .max(self.current.nonpositive_pair_count);
    self.peak.occurrence_len = self.peak.occurrence_len.max(self.current.occurrence_len);
    self.peak.occurrence_capacity = self
      .peak
      .occurrence_capacity
      .max(self.current.occurrence_capacity);
    refresh_payload_bytes::<I>(&mut self.peak);
  }
}

fn refresh_payload_bytes<I>(snapshot: &mut OraclePostingSnapshot) {
  snapshot.pair_frequency_payload_bytes = snapshot
    .stored_pair_count
    .saturating_mul(size_of::<((I, I), Freq)>() as u64);
  snapshot.occurrence_payload_bytes = snapshot
    .occurrence_capacity
    .saturating_mul(size_of::<u64>() as u64);
}

struct HotWindowState<I> {
  policy: HotWindowPolicy,
  window_size: usize,
  hot: AHashMap<(I, I), HotEntry>,
  ever_promoted: AHashSet<(I, I)>,
  pair_winners: u64,
  hot_winners: u64,
  cold_winners: u64,
  hydration_scans: u64,
  hydration_seconds: f64,
  scanned_word_entries: u64,
  scanned_encoded_units: u64,
  scanned_adjacent_positions: u64,
  promotions: u64,
  refill_admissions: u64,
  re_promotions: u64,
  evictions: u64,
  fresh_target_positive_candidates: u64,
  dynamic_admissions: u64,
  threshold_rejections: u64,
  cutoff_tie_refills: u64,
  max_cutoff_tie_width: u64,
  current_refill_uses: Option<u64>,
  refill_uses: Vec<u64>,
  current_occurrence_len: u64,
  current_occurrence_capacity: u64,
  peak_resident_pairs: u64,
  peak_hot_occurrence_len: u64,
  peak_hot_occurrence_capacity: u64,
  admission_threshold: Option<Freq>,
}

struct HotEntry {
  occurs_in: AHashSet<u64>,
}

impl<I> HotWindowState<I>
where
  I: Copy + Eq + Hash,
{
  fn new(window_size: usize, policy: HotWindowPolicy) -> Self {
    Self {
      policy,
      window_size,
      hot: AHashMap::new(),
      ever_promoted: AHashSet::new(),
      pair_winners: 0,
      hot_winners: 0,
      cold_winners: 0,
      hydration_scans: 0,
      hydration_seconds: 0.0,
      scanned_word_entries: 0,
      scanned_encoded_units: 0,
      scanned_adjacent_positions: 0,
      promotions: 0,
      refill_admissions: 0,
      re_promotions: 0,
      evictions: 0,
      fresh_target_positive_candidates: 0,
      dynamic_admissions: 0,
      threshold_rejections: 0,
      cutoff_tie_refills: 0,
      max_cutoff_tie_width: 0,
      current_refill_uses: None,
      refill_uses: Vec::new(),
      current_occurrence_len: 0,
      current_occurrence_capacity: 0,
      peak_resident_pairs: 0,
      peak_hot_occurrence_len: 0,
      peak_hot_occurrence_capacity: 0,
      admission_threshold: None,
    }
  }

  fn contains(&self, tp: (I, I)) -> bool {
    self.hot.contains_key(&tp)
  }

  fn close_refill_epoch(&mut self) {
    if let Some(uses) = self.current_refill_uses.take() {
      self.refill_uses.push(uses);
    }
  }

  fn record_promotion(&mut self, tp: (I, I), dynamic: bool) {
    self.promotions += 1;
    if dynamic {
      self.dynamic_admissions += 1;
    } else {
      self.refill_admissions += 1;
    }
    if !self.ever_promoted.insert(tp) {
      self.re_promotions += 1;
    }
  }

  fn refill<C>(
    &mut self,
    words: &[crate::bpe::PreToken<C, I>],
    desired: &[RankedPair<I>],
    context: &RefillContext<I>,
  ) {
    self.close_refill_epoch();
    self.hydration_scans += 1;

    let desired_set = desired.iter().map(|candidate| candidate.tp).collect::<AHashSet<_>>();
    let hydrate = match self.policy {
      HotWindowPolicy::ReplaceTopK => {
        self.evictions += self
          .hot
          .keys()
          .filter(|tp| !desired_set.contains(tp))
          .count() as u64;
        for candidate in desired {
          if !self.hot.contains_key(&candidate.tp) {
            self.record_promotion(candidate.tp, false);
          }
        }
        desired.to_vec()
      }
      HotWindowPolicy::ThresholdNoEvict => {
        let missing = desired
          .iter()
          .copied()
          .filter(|candidate| !self.hot.contains_key(&candidate.tp))
          .collect::<Vec<_>>();
        for candidate in &missing {
          self.record_promotion(candidate.tp, false);
        }
        missing
      }
    };

    let started = Instant::now();
    let mut hydrated = hydrate
      .into_iter()
      .map(|candidate| {
        (
          candidate.tp,
          HotEntry {
            occurs_in: AHashSet::new(),
          },
        )
      })
      .collect::<AHashMap<_, _>>();
    let mut encoded_units = 0u64;
    let mut adjacent_positions = 0u64;
    for (word_idx, word) in words.iter().enumerate() {
      encoded_units = encoded_units.saturating_add(word.idxs.len() as u64);
      adjacent_positions = adjacent_positions.saturating_add(word.idxs.len().saturating_sub(1) as u64);
      for tp in word.idxs.iter().copied().zip(word.idxs.iter().skip(1).copied()) {
        if let Some(entry) = hydrated.get_mut(&tp) {
          entry.occurs_in.insert(word_idx as u64);
        }
      }
    }
    self.hydration_seconds += started.elapsed().as_secs_f64();
    self.scanned_word_entries = self.scanned_word_entries.saturating_add(words.len() as u64);
    self.scanned_encoded_units = self.scanned_encoded_units.saturating_add(encoded_units);
    self.scanned_adjacent_positions = self.scanned_adjacent_positions.saturating_add(adjacent_positions);
    match self.policy {
      HotWindowPolicy::ReplaceTopK => self.hot = hydrated,
      HotWindowPolicy::ThresholdNoEvict => self.hot.extend(hydrated),
    }
    self.current_occurrence_len = self
      .hot
      .values()
      .map(|entry| entry.occurs_in.len() as u64)
      .sum();
    self.current_occurrence_capacity = self
      .hot
      .values()
      .map(|entry| entry.occurs_in.capacity() as u64)
      .sum();
    self.admission_threshold = desired.last().map(|candidate| candidate.freq);
    self.current_refill_uses = Some(0);

    if let Some(cutoff) = desired.last().map(|candidate| candidate.freq) {
      let total_at_cutoff = context.frequency_counts.get(&cutoff).copied().unwrap_or(0);
      let retained_at_cutoff = desired.iter().filter(|candidate| candidate.freq == cutoff).count() as u64;
      if desired.len() == self.window_size && total_at_cutoff > retained_at_cutoff {
        self.cutoff_tie_refills += 1;
      }
      self.max_cutoff_tie_width = self.max_cutoff_tie_width.max(total_at_cutoff);
    }
    self.observe_hot_peak();
  }

  fn observe_winner<C>(
    &mut self,
    words: &[crate::bpe::PreToken<C, I>],
    winner: (I, I),
    refill: Option<(&[RankedPair<I>], &RefillContext<I>)>,
  ) {
    self.pair_winners += 1;
    if self.contains(winner) {
      self.hot_winners += 1;
    } else {
      self.cold_winners += 1;
      let (desired, context) = refill.expect("a cold winner requires refill data");
      self.refill(words, desired, context);
    }

    let removed = self.hot.remove(&winner).expect("a winner must be resident after refill");
    #[cfg(test)]
    for (word_idx, word) in words.iter().enumerate() {
      let pair_exists = word
        .idxs
        .iter()
        .copied()
        .zip(word.idxs.iter().skip(1).copied())
        .any(|tp| tp == winner);
      if pair_exists {
        assert!(
          removed.occurs_in.contains(&(word_idx as u64)),
          "resident postings must cover every current occurrence",
        );
      }
    }
    self.current_occurrence_len = self
      .current_occurrence_len
      .saturating_sub(removed.occurs_in.len() as u64);
    self.current_occurrence_capacity = self
      .current_occurrence_capacity
      .saturating_sub(removed.occurs_in.capacity() as u64);
    *self.current_refill_uses.as_mut().expect("pair winners follow a refill") += 1;
  }

  fn apply_changes(&mut self, target_idx: I, changes: &AHashMap<(I, I), MergeData>) {
    for (tp, data) in changes {
      let Some(entry) = self.hot.get_mut(tp) else {
        continue;
      };
      if data.freq > 0 {
        let old_len = entry.occurs_in.len() as u64;
        let old_capacity = entry.occurs_in.capacity() as u64;
        entry.occurs_in.extend(data.occurs_in.iter().copied());
        self.current_occurrence_len = self
          .current_occurrence_len
          .saturating_sub(old_len)
          .saturating_add(entry.occurs_in.len() as u64);
        self.current_occurrence_capacity = self
          .current_occurrence_capacity
          .saturating_sub(old_capacity)
          .saturating_add(entry.occurs_in.capacity() as u64);
      }
    }
    if self.policy == HotWindowPolicy::ThresholdNoEvict {
      let threshold = self.admission_threshold.expect("a pair merge follows a populated refill");
      for (tp, data) in changes {
        if data.freq <= 0 || (tp.0 != target_idx && tp.1 != target_idx) {
          continue;
        }
        self.fresh_target_positive_candidates += 1;
        debug_assert!(
          !self.hot.contains_key(tp),
          "a pair containing the fresh target cannot already be resident",
        );
        if data.freq < threshold {
          self.threshold_rejections += 1;
          continue;
        }

        let entry = HotEntry {
          occurs_in: data.occurs_in.clone(),
        };
        self.current_occurrence_len = self
          .current_occurrence_len
          .saturating_add(entry.occurs_in.len() as u64);
        self.current_occurrence_capacity = self
          .current_occurrence_capacity
          .saturating_add(entry.occurs_in.capacity() as u64);
        self.hot.insert(*tp, entry);
        self.record_promotion(*tp, true);
      }
    }
    self.observe_hot_peak();
  }

  fn observe_hot_peak(&mut self) {
    self.peak_resident_pairs = self.peak_resident_pairs.max(self.hot.len() as u64);
    self.peak_hot_occurrence_len = self.peak_hot_occurrence_len.max(self.current_occurrence_len);
    self.peak_hot_occurrence_capacity = self
      .peak_hot_occurrence_capacity
      .max(self.current_occurrence_capacity);
  }

  fn finish(mut self, peak_oracle: &OraclePostingSnapshot) -> HotWindowResult {
    self.close_refill_epoch();
    let cold_winner_rate = if self.pair_winners == 0 {
      0.0
    } else {
      self.cold_winners as f64 / self.pair_winners as f64
    };
    let peak_capacity_ratio = if peak_oracle.occurrence_capacity == 0 {
      0.0
    } else {
      self.peak_hot_occurrence_capacity as f64 / peak_oracle.occurrence_capacity as f64
    };
    HotWindowResult {
      window_size: self.window_size,
      pair_winners: self.pair_winners,
      hot_winners: self.hot_winners,
      cold_winners: self.cold_winners,
      cold_winner_rate,
      hydration_scans: self.hydration_scans,
      hydration_seconds: self.hydration_seconds,
      scanned_word_entries: self.scanned_word_entries,
      scanned_encoded_units: self.scanned_encoded_units,
      scanned_adjacent_positions: self.scanned_adjacent_positions,
      promotions: self.promotions,
      refill_admissions: self.refill_admissions,
      re_promotions: self.re_promotions,
      evictions: self.evictions,
      fresh_target_positive_candidates: self.fresh_target_positive_candidates,
      dynamic_admissions: self.dynamic_admissions,
      threshold_rejections: self.threshold_rejections,
      cutoff_tie_refills: self.cutoff_tie_refills,
      max_cutoff_tie_width: self.max_cutoff_tie_width,
      merges_per_refill: DistributionSummary::from_samples(self.refill_uses),
      peak_resident_pairs: self.peak_resident_pairs,
      peak_hot_occurrence_len: self.peak_hot_occurrence_len,
      peak_hot_occurrence_capacity: self.peak_hot_occurrence_capacity,
      peak_hot_occurrence_payload_bytes: self
        .peak_hot_occurrence_capacity
        .saturating_mul(size_of::<u64>() as u64),
      peak_resident_occurrence_set_structural_bytes: self
        .peak_resident_pairs
        .saturating_mul(size_of::<AHashSet<u64>>() as u64),
      peak_resident_to_window_ratio: self.peak_resident_pairs as f64 / self.window_size as f64,
      final_resident_pairs: self.hot.len() as u64,
      final_hot_occurrence_len: self.current_occurrence_len,
      final_hot_occurrence_capacity: self.current_occurrence_capacity,
      final_admission_threshold: self.admission_threshold,
      peak_oracle_stored_pair_count: peak_oracle.stored_pair_count,
      peak_oracle_positive_pair_count: peak_oracle.positive_pair_count,
      peak_oracle_occurrence_len: peak_oracle.occurrence_len,
      peak_oracle_occurrence_capacity: peak_oracle.occurrence_capacity,
      peak_oracle_occurrence_payload_bytes: peak_oracle
        .occurrence_capacity
        .saturating_mul(size_of::<u64>() as u64),
      peak_hot_to_oracle_occurrence_capacity_ratio: peak_capacity_ratio,
    }
  }
}

fn oracle_snapshot<C, I>(trainer: &BpeTrainer<C, I>) -> OraclePostingSnapshot {
  let mut snapshot = OraclePostingSnapshot::default();
  let mut observe = |merge: &Merge<C, I>| {
    if merge.target.is_some() {
      return;
    }
    snapshot.stored_pair_count += 1;
    if merge.data.freq > 0 {
      snapshot.positive_pair_count += 1;
    } else {
      snapshot.nonpositive_pair_count += 1;
    }
    snapshot.occurrence_len = snapshot
      .occurrence_len
      .saturating_add(merge.data.occurs_in.len() as u64);
    snapshot.occurrence_capacity = snapshot
      .occurrence_capacity
      .saturating_add(merge.data.occurs_in.capacity() as u64);
  };
  for merge in trainer.pre_merges.values() {
    observe(merge);
  }
  refresh_payload_bytes::<I>(&mut snapshot);
  snapshot
}

fn refill_context<C, I>(
  trainer: &mut BpeTrainer<C, I>,
  selected: &Merge<C, I>,
  limit: usize,
) -> RefillContext<I>
where
  C: Ord,
  I: Copy + Eq + Hash + Ord,
{
  let mut ranked = vec![RankedPair {
    tp: selected.tp,
    freq: selected.data.freq,
  }];
  let mut frequency_counts = AHashMap::<Freq, u64>::new();
  frequency_counts.insert(selected.data.freq, 1);
  let mut seen_candidates = AHashSet::new();
  seen_candidates.insert(selected.tp);
  let mut popped = Vec::new();
  let mut cutoff = (limit == 1).then_some(selected.data.freq);

  while let Some(candidate) = trainer.merge_heap.pop() {
    if candidate.freq <= 0 {
      continue;
    }
    let Some(merge) = trainer.pre_merges.get(&candidate.tp) else {
      continue;
    };
    if merge.data.freq != candidate.freq {
      continue;
    }
    if candidate.content.as_ref().is_some_and(|content| merge.content != *content) {
      continue;
    }

    let tp = candidate.tp;
    let freq = candidate.freq;
    let is_pair = merge.target.is_none();
    if !seen_candidates.insert(tp) {
      continue;
    }
    popped.push(candidate);
    if cutoff.is_some_and(|cutoff| freq < cutoff) {
      break;
    }
    if !is_pair {
      continue;
    }

    *frequency_counts.entry(freq).or_default() += 1;
    if ranked.len() < limit {
      ranked.push(RankedPair {
        tp,
        freq,
      });
      if ranked.len() == limit {
        cutoff = Some(freq);
      }
    }
  }
  trainer.merge_heap.extend(popped);

  RefillContext {
    ranked,
    frequency_counts,
  }
}

fn run_analysis<C, I>(
  mut trainer: BpeTrainer<C, I>,
  target_vocab_size: usize,
  window_sizes: &[usize],
  policy: HotWindowPolicy,
) -> (HotWindowAnalysisReport, BpeTrainer<C, I>)
where
  Word<C>: WordDebugExt,
  C: CanStrToWord + CanToWord<u8> + Clone + Ord + Send + Sync,
  I: IdxLike + HasChar<C> + Hash,
{
  assert!(!window_sizes.is_empty(), "at least one window size is required");
  assert!(window_sizes.iter().all(|size| *size > 0), "window sizes must be positive");
  let mut window_sizes = window_sizes.to_vec();
  window_sizes.sort_unstable();
  window_sizes.dedup();
  let mut states = window_sizes
    .into_iter()
    .map(|window_size| HotWindowState::new(window_size, policy))
    .collect::<Vec<_>>();

  let initial_vocab_size = trainer.vocab.len();
  let init_started = Instant::now();
  trainer._build_pre_merges();
  let init_training_seconds = init_started.elapsed().as_secs_f64();
  let initial_oracle = oracle_snapshot(&trainer);
  let mut oracle_accounting = OracleAccounting::new(initial_oracle.clone());

  let mut initial_unit_steps = 0u64;
  let mut pair_merge_steps = 0u64;
  let mut shared_refill_context_builds = 0u64;
  let mut shared_refill_context_seconds = 0.0;
  let merge_started = Instant::now();
  while trainer.vocab.len() < target_vocab_size {
    let Some(merge) = trainer._get_largest_merge() else {
      break;
    };
    if merge.target.is_some() {
      initial_unit_steps += 1;
      trainer._step_with_observer(merge, |_, _, _| {});
      continue;
    }

    pair_merge_steps += 1;
    let refill_limit = states
      .iter()
      .filter(|state| !state.contains(merge.tp))
      .map(|state| state.window_size)
      .max();
    let context = refill_limit.map(|limit| {
      shared_refill_context_builds += 1;
      let started = Instant::now();
      let context = refill_context(&mut trainer, &merge, limit);
      shared_refill_context_seconds += started.elapsed().as_secs_f64();
      context
    });
    for state in &mut states {
      let refill = if state.contains(merge.tp) {
        None
      } else {
        let context = context.as_ref().expect("a cold state requires refill data");
        let desired_len = state.window_size.min(context.ranked.len());
        Some((&context.ranked[..desired_len], context))
      };
      state.observe_winner(&trainer.words, merge.tp, refill);
    }
    let selected = merge.tp;
    oracle_accounting.remove_selected(&merge);
    let mut pending_oracle_updates = None;
    trainer._step_with_observer(merge, |trainer, target_idx, changes| {
      if let Some(changes) = changes {
        for state in &mut states {
          state.apply_changes(target_idx, changes);
        }
        pending_oracle_updates = Some(PendingOracleUpdates::prepare(
          trainer,
          selected,
          changes,
        ));
      }
    });
    pending_oracle_updates
      .expect("pair merges always produce a change map")
      .apply(&trainer, &mut oracle_accounting);
  }
  let simulation_merge_seconds = merge_started.elapsed().as_secs_f64();
  let final_vocab_size = trainer.vocab.len();
  let final_oracle = oracle_snapshot(&trainer);
  assert_eq!(
    oracle_accounting.current,
    final_oracle,
    "incremental oracle accounting diverged from the trainer",
  );
  let peak_oracle = oracle_accounting.peak;
  let policy_note = match policy {
    HotWindowPolicy::ReplaceTopK => {
      "A cold winner replaces the window with the exact current top-K positive pairs, including that winner, then hydrates all K occurrence sets in one inventory scan."
    }
    HotWindowPolicy::ThresholdNoEvict => {
      "A cold winner unions the exact current top-K positive pairs into the resident set; newly created positive pairs are admitted without a scan when their frequency reaches the least frequency in that refill snapshot, and residents are never evicted."
    }
  };
  let mut notes = vec![
    "The current exact trainer remains the winner-selection oracle; simulated windows never change merge order or output.".to_string(),
    "Initial Unicode units are not counted as pair-window entries.".to_string(),
    policy_note.to_string(),
  ];
  if policy == HotWindowPolicy::ThresholdNoEvict {
    notes.push(
      "The refill cutoff remains fixed until the next cold-winner refill so cooled residents cannot collapse the admission gate; K is a refill snapshot size, not a hard resident limit.".to_string(),
    );
  }
  notes.extend([
    "Positive occurrence deltas extend resident postings; negative deltas retain stale word ids, matching the current trainer's lazy occurrence semantics.".to_string(),
    "Occurrence payload estimates exclude hash-table buckets, allocators, and token content; pair-frequency payload is only two ids plus one frequency per stored pair and excludes ranking overhead.".to_string(),
    "peak_resident_occurrence_set_structural_bytes counts only the in-map AHashSet handles; map buckets and separately allocated posting storage remain excluded.".to_string(),
    "simulation_merge_seconds includes exact oracle training and all requested K simulations; per-window hydration_seconds excludes shared heap inspection and bookkeeping.".to_string(),
    "Multiple K values are measured in ascending order in one process, so later hydration scans may benefit from warmed caches; run one K at a time for isolated timing.".to_string(),
  ]);
  let report = HotWindowAnalysisReport {
    policy: policy.as_str().to_string(),
    target_vocab_size,
    initial_vocab_size,
    final_vocab_size,
    completed: final_vocab_size >= target_vocab_size,
    initial_unit_steps,
    pair_merge_steps,
    init_training_seconds,
    simulation_merge_seconds,
    shared_refill_context_builds,
    shared_refill_context_seconds,
    initial_oracle,
    peak_oracle: peak_oracle.clone(),
    final_oracle,
    windows: states
      .into_iter()
      .map(|state| state.finish(&peak_oracle))
      .collect(),
    notes,
  };
  (report, trainer)
}

#[doc(hidden)]
pub fn analyze_byte_words(
  words: OrderMap<String, Freq>,
  special_tokens: &[String],
  config: BpeTrainerConfig,
  target_vocab_size: usize,
  window_sizes: &[usize],
  policy: HotWindowPolicy,
) -> HotWindowAnalysisReport {
  run_analysis(
    BpeTrainer::<u8, Idx>::from_words_with_config(words, special_tokens, config),
    target_vocab_size,
    window_sizes,
    policy,
  ).0
}

#[doc(hidden)]
pub fn analyze_unicode_words(
  words: OrderMap<String, Freq>,
  special_tokens: &[String],
  config: BpeTrainerConfig,
  target_vocab_size: usize,
  window_sizes: &[usize],
  policy: HotWindowPolicy,
) -> HotWindowAnalysisReport {
  run_analysis(
    BpeTrainer::<Character, CharIdx>::from_words_with_config(
      words,
      special_tokens,
      config,
    ),
    target_vocab_size,
    window_sizes,
    policy,
  ).0
}

#[cfg(test)]
mod tests {
  use crate::bpe::TieBreak;

  use super::*;

  fn words(entries: &[(&str, Freq)]) -> OrderMap<String, Freq> {
    entries
      .iter()
      .map(|(word, freq)| ((*word).to_string(), *freq))
      .collect()
  }

  fn merge_snapshot<C, I>(trainer: &BpeTrainer<C, I>) -> Vec<((I, I), Option<I>, Freq)>
  where
    I: Copy,
  {
    trainer
      .merges
      .iter()
      .map(|merge| (merge.tp, merge.target, merge.data.freq))
      .collect()
  }

  fn run_baseline<C, I>(
    trainer: BpeTrainer<C, I>,
    target_vocab_size: usize,
    window_sizes: &[usize],
  ) -> (HotWindowAnalysisReport, BpeTrainer<C, I>)
  where
    Word<C>: WordDebugExt,
    C: CanStrToWord + CanToWord<u8> + Clone + Ord + Send + Sync,
    I: IdxLike + HasChar<C> + Hash,
  {
    run_analysis(
      trainer,
      target_vocab_size,
      window_sizes,
      HotWindowPolicy::ReplaceTopK,
    )
  }

  fn run_threshold<C, I>(
    trainer: BpeTrainer<C, I>,
    target_vocab_size: usize,
    window_sizes: &[usize],
  ) -> (HotWindowAnalysisReport, BpeTrainer<C, I>)
  where
    Word<C>: WordDebugExt,
    C: CanStrToWord + CanToWord<u8> + Clone + Ord + Send + Sync,
    I: IdxLike + HasChar<C> + Hash,
  {
    run_analysis(
      trainer,
      target_vocab_size,
      window_sizes,
      HotWindowPolicy::ThresholdNoEvict,
    )
  }

  #[test]
  fn fixed_window_refills_when_independent_pairs_are_exhausted() {
    let trainer = BpeTrainer::<u8, Idx>::from_words(
      words(&[("ab", 10), ("cd", 9), ("ef", 8)]),
      &[],
    );
    let (report, _) = run_baseline(trainer, 259, &[1, 2]);
    assert_eq!(report.pair_merge_steps, 3);
    assert_eq!(report.windows[0].hydration_scans, 3);
    assert_eq!(report.windows[1].hydration_scans, 2);
    assert_eq!(report.windows[1].merges_per_refill.max, Some(2));
  }

  #[test]
  fn newly_created_pair_starts_cold() {
    let trainer = BpeTrainer::<u8, Idx>::from_words(words(&[("abc", 10)]), &[]);
    let (report, _) = run_baseline(trainer, 258, &[1]);
    assert_eq!(report.pair_merge_steps, 2);
    assert_eq!(report.windows[0].cold_winners, 2);
  }

  #[test]
  fn threshold_admission_keeps_a_new_winner_hot_without_another_scan() {
    let trainer = BpeTrainer::<u8, Idx>::from_words(words(&[("abc", 10)]), &[]);
    let (report, _) = run_threshold(trainer, 258, &[1]);
    let window = &report.windows[0];
    assert_eq!(window.hydration_scans, 1);
    assert_eq!(window.cold_winners, 1);
    assert_eq!(window.hot_winners, 1);
    assert_eq!(window.fresh_target_positive_candidates, 1);
    assert_eq!(window.dynamic_admissions, 1);
    assert_eq!(window.threshold_rejections, 0);
  }

  #[test]
  fn threshold_admission_keeps_complete_multiword_postings() {
    let trainer = BpeTrainer::<u8, Idx>::from_words(
      words(&[("abq", 10), ("abqr", 9), ("abqs", 8)]),
      &[],
    );
    let (report, _) = run_threshold(trainer, 258, &[1]);
    assert_eq!(report.windows[0].hydration_scans, 1);
    assert_eq!(report.windows[0].hot_winners, 1);
  }

  #[test]
  fn threshold_admission_rejects_new_pairs_below_the_hot_minimum() {
    let trainer = BpeTrainer::<u8, Idx>::from_words(
      words(&[("ab", 10), ("zabq", 5)]),
      &[],
    );
    let (report, _) = run_threshold(trainer, 257, &[1]);
    let window = &report.windows[0];
    assert_eq!(window.fresh_target_positive_candidates, 2);
    assert_eq!(window.dynamic_admissions, 0);
    assert_eq!(window.threshold_rejections, 2);
    assert_eq!(window.final_admission_threshold, Some(15));
  }

  #[test]
  fn threshold_admission_can_grow_beyond_k_without_eviction() {
    let trainer = BpeTrainer::<u8, Idx>::from_words(words(&[("zabq", 5)]), &[]);
    let (report, _) = run_threshold(trainer, 257, &[1]);
    let window = &report.windows[0];
    assert_eq!(window.dynamic_admissions, 2);
    assert_eq!(window.evictions, 0);
    assert_eq!(window.peak_resident_pairs, 2);
    assert_eq!(window.final_resident_pairs, 2);
    assert_eq!(window.peak_resident_to_window_ratio, 2.0);
  }

  #[test]
  fn threshold_refill_unions_residents_instead_of_evicting_them() {
    let mut state = HotWindowState::<Idx>::new(1, HotWindowPolicy::ThresholdNoEvict);
    state.hot.insert(
      (1, 2),
      HotEntry {
        occurs_in: [0].into_iter().collect(),
      },
    );
    state.current_occurrence_len = 1;
    state.current_occurrence_capacity = state.hot[&(1, 2)].occurs_in.capacity() as u64;
    state.admission_threshold = Some(5);
    let words = vec![crate::bpe::PreToken {
      src: Vec::<u8>::new().into(),
      idxs: vec![3, 4],
      freq: 10,
    }];
    let desired = [RankedPair {
      tp: (3, 4),
      freq: 10,
    }];
    let context = RefillContext {
      ranked: desired.to_vec(),
      frequency_counts: [(10, 1)].into_iter().collect(),
    };

    state.refill(&words, &desired, &context);
    assert!(state.hot.contains_key(&(1, 2)));
    assert!(state.hot.contains_key(&(3, 4)));
    assert_eq!(state.evictions, 0);
  }

  #[test]
  fn threshold_admission_only_considers_fresh_target_pairs() {
    let mut state = HotWindowState::<Idx>::new(1, HotWindowPolicy::ThresholdNoEvict);
    state.hot.insert(
      (1, 2),
      HotEntry {
        occurs_in: [0].into_iter().collect(),
      },
    );
    state.current_occurrence_len = 1;
    state.current_occurrence_capacity = state.hot[&(1, 2)].occurs_in.capacity() as u64;
    state.admission_threshold = Some(10);
    let target = 100;
    let changes = [
      ((1, 2), MergeData::new(-9)),
      ((target, 5), MergeData::new(10).add_occurs_in([1, 2])),
      ((6, target), MergeData::new(9).add_occurs_in([3])),
      ((3, 4), MergeData::new(20).add_occurs_in([4])),
    ]
    .into_iter()
    .collect();

    state.apply_changes(target, &changes);
    assert!(state.hot.contains_key(&(target, 5)));
    assert!(!state.hot.contains_key(&(6, target)));
    assert!(!state.hot.contains_key(&(3, 4)));
    assert_eq!(state.fresh_target_positive_candidates, 2);
    assert_eq!(state.dynamic_admissions, 1);
    assert_eq!(state.threshold_rejections, 1);
    assert_eq!(state.admission_threshold, Some(10));
  }

  #[test]
  fn reports_ties_that_cross_the_window_cutoff() {
    let trainer = BpeTrainer::<u8, Idx>::from_words(
      words(&[("ab", 10), ("cd", 10), ("ef", 10)]),
      &[],
    );
    let (report, _) = run_baseline(trainer, 257, &[2]);
    assert_eq!(report.windows[0].cutoff_tie_refills, 1);
    assert_eq!(report.windows[0].max_cutoff_tie_width, 3);
  }

  #[test]
  fn unicode_initial_units_do_not_enter_the_pair_window() {
    let trainer = BpeTrainer::<Character, CharIdx>::from_words(words(&[("你a", 10)]), &[]);
    let (report, _) = run_baseline(trainer, 258, &[1]);
    assert_eq!(report.initial_unit_steps, 1);
    assert_eq!(report.pair_merge_steps, 1);
    assert_eq!(report.windows[0].hydration_scans, 1);
  }

  #[test]
  fn simulation_preserves_exact_training_for_both_tie_breaks() {
    for tie_break in [TieBreak::SmallestPairId, TieBreak::LargestContent] {
      let config = BpeTrainerConfig {
        tie_break,
        ..BpeTrainerConfig::default()
      };
      let inventory = words(&[("abca", 10), ("bcab", 10), ("cabc", 10)]);
      let mut exact = BpeTrainer::<u8, Idx>::from_words_with_config(
        inventory.clone(),
        &[],
        config,
      );
      exact.train_until(261).unwrap();

      let analyzed = BpeTrainer::<u8, Idx>::from_words_with_config(inventory, &[], config);
      let (_, analyzed) = run_baseline(analyzed, 261, &[1, 3]);
      assert_eq!(analyzed.vocab, exact.vocab);
      assert_eq!(merge_snapshot(&analyzed), merge_snapshot(&exact));
    }
  }

  #[test]
  fn byte_level_analysis_uses_the_configured_byte_vocab_ids() {
    let config = BpeTrainerConfig::hf_byte_level();
    let trainer = BpeTrainer::<u8, Idx>::from_words_with_config(
      words(&[("ÿa", 10), ("¡b", 9)]),
      &[],
      config,
    );
    for word in &trainer.words {
      for (byte, idx) in word.src.iter().zip(&word.idxs) {
        assert_eq!(Some(idx), trainer.byte_vocab.get(byte));
      }
    }
  }

  #[test]
  fn oracle_storage_includes_nonpositive_pairs_and_stale_postings() {
    let trainer = BpeTrainer::<u8, Idx>::from_words(words(&[("abc", 10)]), &[]);
    let (report, _) = run_baseline(trainer, 257, &[1]);
    assert!(report.final_oracle.nonpositive_pair_count > 0);
    assert!(report.final_oracle.occurrence_capacity > 0);
    assert_eq!(
      report.final_oracle.stored_pair_count,
      report.final_oracle.positive_pair_count + report.final_oracle.nonpositive_pair_count,
    );
  }

  #[test]
  fn heap_refill_matches_a_full_exact_ranking() {
    use super::super::MergeCandidate;

    for tie_break in [TieBreak::SmallestPairId, TieBreak::LargestContent] {
      let config = BpeTrainerConfig {
        tie_break,
        ..BpeTrainerConfig::default()
      };
      let mut trainer = BpeTrainer::<u8, Idx>::from_words_with_config(
        words(&[("ababa", 10), ("babab", 10), ("abcab", 8), ("bcabc", 8)]),
        &[],
        config,
      );
      trainer._build_pre_merges();
      let first = trainer._get_largest_merge().unwrap();
      trainer._step(first);
      let selected = trainer._get_largest_merge().unwrap();

      let mut expected = std::iter::once(&selected)
        .chain(
          trainer
            .pre_merges
            .values()
            .filter(|merge| merge.target.is_none() && merge.data.freq > 0),
        )
        .map(|merge| MergeCandidate::from_merge(merge, tie_break))
        .collect::<Vec<_>>();
      expected.sort_unstable_by(|left, right| right.cmp(left));
      expected.truncate(3);
      let expected_tps = expected.iter().map(|candidate| candidate.tp).collect::<Vec<_>>();
      let cutoff = expected.last().unwrap().freq;
      let expected_at_cutoff = std::iter::once(&selected)
        .chain(trainer.pre_merges.values())
        .filter(|merge| {
          merge.target.is_none() && merge.data.freq > 0 && merge.data.freq == cutoff
        })
        .count() as u64;

      let context = refill_context(&mut trainer, &selected, 3);
      assert_eq!(
        context.ranked.iter().map(|candidate| candidate.tp).collect::<Vec<_>>(),
        expected_tps,
      );
      assert_eq!(context.frequency_counts.get(&cutoff), Some(&expected_at_cutoff));
    }
  }

  #[test]
  fn a_huge_window_is_bounded_by_the_available_candidates() {
    let trainer = BpeTrainer::<u8, Idx>::from_words(words(&[("abc", 10)]), &[]);
    let (report, _) = run_baseline(trainer, 257, &[usize::MAX]);
    assert_eq!(report.windows[0].hydration_scans, 1);
    assert!(report.windows[0].peak_resident_pairs <= 2);
  }

  #[test]
  fn empty_inventory_reports_incomplete_training() {
    let trainer = BpeTrainer::<u8, Idx>::from_words(words(&[]), &[]);
    let (report, _) = run_baseline(trainer, 257, &[1]);
    assert!(!report.completed);
    assert_eq!(report.final_vocab_size, 256);
    assert_eq!(report.pair_merge_steps, 0);
    assert_eq!(report.windows[0].hydration_scans, 0);
  }

  #[test]
  fn unicode_simulation_preserves_exact_training() {
    let inventory = words(&[("你好你", 10), ("好你啊", 9)]);
    let mut exact = BpeTrainer::<Character, CharIdx>::from_words(inventory.clone(), &[]);
    exact.train_until(262).unwrap();

    let analyzed = BpeTrainer::<Character, CharIdx>::from_words(inventory, &[]);
    let (_, analyzed) = run_baseline(analyzed, 262, &[1, 4]);
    assert_eq!(analyzed.vocab, exact.vocab);
    assert_eq!(merge_snapshot(&analyzed), merge_snapshot(&exact));
  }

  #[test]
  #[should_panic(expected = "at least one window size is required")]
  fn empty_window_sweep_is_rejected() {
    let trainer = BpeTrainer::<u8, Idx>::from_words(words(&[("ab", 1)]), &[]);
    let _ = run_baseline(trainer, 257, &[]);
  }
}
