use std::hash::Hash;

use ahash::{AHashMap, AHashSet};

use super::{Freq, MergeData, PreToken};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HotPairWindowStats {
  pub hydration_scans: u64,
  pub hydrated_word_entries: u64,
  pub batch_prunes: u64,
  pub prune_evictions: u64,
  pub peak_resident_pairs: usize,
}

#[derive(Debug)]
pub(crate) struct HotPairWindow<I> {
  window_size: Option<usize>,
  occurrences: AHashMap<(I, I), AHashSet<u64>>,
  admission_threshold: Option<Freq>,
  stats: HotPairWindowStats,
}

impl<I> Default for HotPairWindow<I> {
  fn default() -> Self {
    Self {
      window_size: None,
      occurrences: AHashMap::new(),
      admission_threshold: None,
      stats: HotPairWindowStats::default(),
    }
  }
}

impl<I> HotPairWindow<I> {
  pub(crate) fn new(window_size: Option<usize>) -> Self {
    assert!(window_size != Some(0), "hot_pair_window_size must be positive");
    Self {
      window_size,
      ..Self::default()
    }
  }

  pub(crate) fn reset(&mut self, window_size: Option<usize>) {
    *self = Self::new(window_size);
  }

  pub(crate) fn is_enabled(&self) -> bool {
    self.window_size.is_some()
  }

  pub(crate) fn window_size(&self) -> usize {
    self.window_size.expect("hot pair window is disabled")
  }

  pub(crate) fn stats(&self) -> Option<&HotPairWindowStats> {
    self.is_enabled().then_some(&self.stats)
  }

  pub(crate) fn len(&self) -> usize {
    self.occurrences.len()
  }

  pub(crate) fn occurrence_capacity(&self) -> usize {
    self.occurrences.values().map(|occurrences| occurrences.capacity()).sum()
  }
}

impl<I> HotPairWindow<I>
where
  I: Copy + Eq + Hash,
{
  pub(crate) fn contains(&self, tp: &(I, I)) -> bool {
    self.occurrences.contains_key(tp)
  }

  pub(crate) fn resident_pairs(&self) -> impl Iterator<Item = &(I, I)> {
    self.occurrences.keys()
  }

  #[cfg(test)]
  pub(crate) fn occurrences(&self, tp: &(I, I)) -> Option<&AHashSet<u64>> {
    self.occurrences.get(tp)
  }

  pub(crate) fn take(&mut self, tp: &(I, I)) -> Option<AHashSet<u64>> {
    self.occurrences.remove(tp)
  }

  pub(crate) fn hydrate<C>(
    &mut self,
    ranked: &[((I, I), Freq)],
    words: &[PreToken<C, I>],
    required: Option<(I, I)>,
  ) {
    let desired = ranked.iter().map(|(tp, _)| *tp).collect::<AHashSet<_>>();
    if let Some(required) = required {
      assert!(
        desired.contains(&required),
        "a cold pair winner must be present in the exact top-K refill",
      );
    }

    let mut hydrated = desired
      .iter()
      .copied()
      .filter(|tp| !self.occurrences.contains_key(tp))
      .map(|tp| (tp, AHashSet::new()))
      .collect::<AHashMap<_, _>>();
    for (word_idx, word) in words.iter().enumerate() {
      for tp in word.idxs.iter().copied().zip(word.idxs.iter().skip(1).copied()) {
        if let Some(occurrences) = hydrated.get_mut(&tp) {
          occurrences.insert(word_idx as u64);
        }
      }
    }

    self.occurrences.extend(hydrated);
    self.admission_threshold = ranked.last().map(|(_, freq)| *freq);
    self.stats.hydration_scans += 1;
    self.stats.hydrated_word_entries = self
      .stats
      .hydrated_word_entries
      .saturating_add(words.len() as u64);
    self.observe_peak();
  }

  pub(crate) fn apply_changes(
    &mut self,
    target_idx: I,
    changes: &AHashMap<(I, I), MergeData>,
  ) {
    for (tp, data) in changes {
      if data.freq > 0 {
        if let Some(occurrences) = self.occurrences.get_mut(tp) {
          occurrences.extend(data.occurs_in.iter().copied());
        }
      }
    }

    let threshold = self
      .admission_threshold
      .expect("a pair merge follows an initialized hot window");
    for (tp, data) in changes {
      if data.freq <= 0
        || (tp.0 != target_idx && tp.1 != target_idx)
        || data.freq < threshold
      {
        continue;
      }
      debug_assert!(!self.occurrences.contains_key(tp));
      self.occurrences.insert(*tp, data.occurs_in.clone());
    }
    self.observe_peak();
  }

  pub(crate) fn remove_pairs<'a>(&mut self, pairs: impl IntoIterator<Item = &'a (I, I)>)
  where
    I: 'a,
  {
    for tp in pairs {
      self.occurrences.remove(tp);
    }
  }

  pub(crate) fn needs_prune(&self) -> bool {
    self.len() > self.window_size().saturating_mul(2)
  }

  pub(crate) fn prune(&mut self, ranked: &[((I, I), Freq)]) {
    let retained = ranked.iter().map(|(tp, _)| *tp).collect::<AHashSet<_>>();
    let before = self.occurrences.len();
    self.occurrences.retain(|tp, _| retained.contains(tp));
    self.admission_threshold = ranked.last().map(|(_, freq)| *freq);
    self.stats.batch_prunes += 1;
    self.stats.prune_evictions = self
      .stats
      .prune_evictions
      .saturating_add(before.saturating_sub(self.occurrences.len()) as u64);
  }

  fn observe_peak(&mut self) {
    self.stats.peak_resident_pairs = self.stats.peak_resident_pairs.max(self.len());
  }

  #[cfg(test)]
  pub(crate) fn replace_residents(
    &mut self,
    occurrences: AHashMap<(I, I), AHashSet<u64>>,
  ) {
    self.occurrences = occurrences;
  }
}
