use std::{collections::HashMap, hash::Hash, num::NonZeroUsize};

use ahash::{AHashMap, AHashSet};
use slab::Slab;

use super::{Freq, Merge, MergeData, PreToken, Word};

type Pair<I> = (I, I);

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HotPairWindowStats {
  pub hydration_scans: u64,
  pub hydrated_word_entries: u64,
  pub batch_prunes: u64,
  pub prune_evictions: u64,
  pub peak_resident_pairs: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OccursInId(NonZeroUsize);

impl OccursInId {
  fn from_key(key: usize) -> Self {
    Self(NonZeroUsize::new(key.checked_add(1).expect("occurs_in slab exhausted")).unwrap())
  }

  fn key(self) -> usize {
    self.0.get() - 1
  }
}

#[derive(Debug, Default)]
struct OccursInStore {
  sets: Slab<AHashSet<u64>>,
}

impl OccursInStore {
  fn insert(&mut self, occurs_in: AHashSet<u64>) -> OccursInId {
    OccursInId::from_key(self.sets.insert(occurs_in))
  }

  #[cfg(test)]
  fn get(&self, id: OccursInId) -> &AHashSet<u64> {
    self.sets.get(id.key()).expect("stale occurs_in id")
  }

  fn get_mut(&mut self, id: OccursInId) -> &mut AHashSet<u64> {
    self.sets.get_mut(id.key()).expect("stale occurs_in id")
  }

  fn remove(&mut self, id: OccursInId) -> AHashSet<u64> {
    self.sets.try_remove(id.key()).expect("stale occurs_in id")
  }

  fn capacity(&self) -> usize {
    self.sets.iter().map(|(_, occurs_in)| occurs_in.capacity()).sum()
  }
}

/// Trainer-internal state for a discovered pair or Unicode initial unit.
///
/// Persistent occurrence sets live in [`PairStore::occurs_in`]; this state
/// carries only a store-local handle and is intentionally not cloneable.
#[derive(Debug)]
pub(crate) struct PairState<C, I> {
  pub(crate) content: (Word<C>, Word<C>),
  pub(crate) target: Option<I>,
  pub(crate) freq: Freq,
  occurs_in: Option<OccursInId>,
}

impl<C, I> PairState<C, I> {
  pub(crate) fn new(content: (Word<C>, Word<C>)) -> Self {
    Self {
      content,
      target: None,
      freq: 0,
      occurs_in: None,
    }
  }

  pub(crate) fn with_target(mut self, target: I) -> Self {
    self.target = Some(target);
    self
  }
}

/// Authoritative trainer candidate table and persistent `occurs_in` owner.
///
/// Exact mode gives every discovered pair an occurrence-set handle. Bounded
/// mode gives handles only to resident pairs and retains exact frequencies for
/// cold pairs in `pairs`.
#[derive(Debug)]
pub(crate) struct PairStore<C, I> {
  pairs: HashMap<Pair<I>, PairState<C, I>>,
  occurs_in: OccursInStore,
  window_size: Option<usize>,
  admission_threshold: Option<Freq>,
  residents: AHashSet<Pair<I>>,
  stats: HotPairWindowStats,
}

impl<C, I> Default for PairStore<C, I> {
  fn default() -> Self {
    Self {
      pairs: HashMap::new(),
      occurs_in: OccursInStore::default(),
      window_size: None,
      admission_threshold: None,
      residents: AHashSet::new(),
      stats: HotPairWindowStats::default(),
    }
  }
}

impl<C, I> PairStore<C, I> {
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

  pub(crate) fn is_bounded(&self) -> bool {
    self.window_size.is_some()
  }

  pub(crate) fn window_size(&self) -> usize {
    self.window_size.expect("pair occurrence window is disabled")
  }

  pub(crate) fn stats(&self) -> Option<&HotPairWindowStats> {
    self.is_bounded().then_some(&self.stats)
  }

  pub(crate) fn len(&self) -> usize {
    self.pairs.len()
  }

  pub(crate) fn resident_len(&self) -> usize {
    if self.is_bounded() {
      self.residents.len()
    } else {
      0
    }
  }

  pub(crate) fn resident_occurrence_capacity(&self) -> usize {
    if self.is_bounded() {
      self.occurs_in.capacity()
    } else {
      0
    }
  }
}

impl<C, I> PairStore<C, I>
where
  I: Copy + Eq + Hash,
{

  #[cfg(test)]
  pub(crate) fn contains_key(&self, tp: &Pair<I>) -> bool {
    self.pairs.contains_key(tp)
  }

  pub(crate) fn get(&self, tp: &Pair<I>) -> Option<&PairState<C, I>> {
    self.pairs.get(tp)
  }

  pub(crate) fn iter(&self) -> impl Iterator<Item = (&Pair<I>, &PairState<C, I>)> {
    self.pairs.iter()
  }

  pub(crate) fn insert_initial(&mut self, tp: Pair<I>, state: PairState<C, I>) {
    debug_assert!(state.target.is_some());
    debug_assert!(state.occurs_in.is_none());
    self.pairs.insert(tp, state);
  }

  pub(crate) fn add_initial_freq(&mut self, tp: &Pair<I>, freq: Freq) -> bool {
    let Some(state) = self.pairs.get_mut(tp) else {
      return false;
    };
    state.freq += freq;
    true
  }

  pub(crate) fn add_pair(
    &mut self,
    tp: Pair<I>,
    content: impl FnOnce() -> (Word<C>, Word<C>),
    freq: Freq,
    occurs_in: impl IntoIterator<Item = u64>,
  ) {
    let (pairs, store) = (&mut self.pairs, &mut self.occurs_in);
    let state = pairs.entry(tp).or_insert_with(|| PairState::new(content()));
    state.freq += freq;
    if self.window_size.is_none() {
      let id = *state
        .occurs_in
        .get_or_insert_with(|| store.insert(AHashSet::new()));
      store.get_mut(id).extend(occurs_in);
    }
  }

  #[cfg(test)]
  pub(crate) fn occurrence_vec(&self, tp: &Pair<I>) -> Vec<u64> {
    let Some(id) = self.pairs.get(tp).and_then(|state| state.occurs_in) else {
      return Vec::new();
    };
    let mut result = self.occurs_in.get(id).iter().copied().collect::<Vec<_>>();
    result.sort_unstable();
    result
  }

  #[cfg(test)]
  pub(crate) fn occurrence_set_count(&self) -> usize {
    self.occurs_in.sets.len()
  }

  pub(crate) fn resident_pairs(&self) -> impl Iterator<Item = &Pair<I>> {
    self.residents.iter()
  }

  pub(crate) fn is_resident(&self, tp: &Pair<I>) -> bool {
    self.pairs.get(tp).is_some_and(|state| state.occurs_in.is_some())
  }

  pub(crate) fn hydrate(
    &mut self,
    ranked: &[(Pair<I>, Freq)],
    words: &[PreToken<C, I>],
    required: Option<Pair<I>>,
  ) {
    assert!(self.is_bounded(), "cannot hydrate exact pair occurrences");
    let desired = ranked.iter().map(|(tp, _)| *tp).collect::<AHashSet<_>>();
    if let Some(required) = required {
      assert!(
        desired.contains(&required),
        "a cold pair winner must be present in the exact top-K refill",
      );
    }

    let (pairs, store, residents) = (
      &mut self.pairs,
      &mut self.occurs_in,
      &mut self.residents,
    );
    let mut hydrated = AHashMap::new();
    for tp in desired {
      let state = pairs.get_mut(&tp).expect("ranked pair is missing from pair store");
      if state.occurs_in.is_some() {
        continue;
      }
      let id = store.insert(AHashSet::new());
      state.occurs_in = Some(id);
      residents.insert(tp);
      hydrated.insert(tp, id);
    }

    for (word_idx, word) in words.iter().enumerate() {
      for tp in word.idxs.iter().copied().zip(word.idxs.iter().skip(1).copied()) {
        if let Some(id) = hydrated.get(&tp) {
          store.get_mut(*id).insert(word_idx as u64);
        }
      }
    }

    self.admission_threshold = ranked.last().map(|(_, freq)| *freq);
    self.stats.hydration_scans += 1;
    self.stats.hydrated_word_entries = self
      .stats
      .hydrated_word_entries
      .saturating_add(words.len() as u64);
    self.observe_peak();
    self.debug_assert_residency();
  }

  pub(crate) fn apply_changes<F>(
    &mut self,
    selected: Pair<I>,
    target_idx: I,
    changes: AHashMap<Pair<I>, MergeData>,
    mut resolve_content: F,
  ) -> Vec<Pair<I>>
  where
    F: FnMut(Pair<I>) -> (Word<C>, Word<C>),
  {
    let bounded = self.is_bounded();
    let threshold = self.admission_threshold;
    let (pairs, store, residents) = (
      &mut self.pairs,
      &mut self.occurs_in,
      &mut self.residents,
    );
    let mut updated = Vec::new();

    for (tp, data) in changes {
      if tp == selected || data.freq == 0 {
        continue;
      }
      let state = pairs
        .entry(tp)
        .or_insert_with(|| PairState::new(resolve_content(tp)));
      state.freq += data.freq;

      if !bounded {
        if data.freq > 0 {
          let id = *state
            .occurs_in
            .get_or_insert_with(|| store.insert(AHashSet::new()));
          store.get_mut(id).extend(data.occurs_in);
        }
      } else if let Some(id) = state.occurs_in {
        if data.freq > 0 {
          store.get_mut(id).extend(data.occurs_in);
        }
        if state.freq <= 0 {
          state.occurs_in = None;
          residents.remove(&tp);
          drop(store.remove(id));
        }
      } else if data.freq > 0
        && (tp.0 == target_idx || tp.1 == target_idx)
        && data.freq >= threshold.expect("bounded pair store is not initialized")
      {
        state.occurs_in = Some(store.insert(data.occurs_in));
        residents.insert(tp);
      }
      updated.push(tp);
    }

    self.observe_peak();
    self.debug_assert_residency();
    updated
  }

  pub(crate) fn needs_prune(&self) -> bool {
    self.is_bounded() && self.residents.len() > self.window_size().saturating_mul(2)
  }

  pub(crate) fn prune(&mut self, ranked: &[(Pair<I>, Freq)]) {
    let retained = ranked.iter().map(|(tp, _)| *tp).collect::<AHashSet<_>>();
    let evicted = self
      .residents
      .iter()
      .filter(|tp| !retained.contains(tp))
      .copied()
      .collect::<Vec<_>>();
    let before = self.residents.len();
    for tp in evicted {
      self.release(&tp);
    }
    self.admission_threshold = ranked.last().map(|(_, freq)| *freq);
    self.stats.batch_prunes += 1;
    self.stats.prune_evictions = self
      .stats
      .prune_evictions
      .saturating_add(before.saturating_sub(self.residents.len()) as u64);
    self.debug_assert_residency();
  }

  pub(crate) fn take_merge(&mut self, tp: &Pair<I>) -> Option<Merge<C, I>> {
    let state = self.pairs.remove(tp)?;
    let occurs_in = state
      .occurs_in
      .map(|id| self.occurs_in.remove(id))
      .unwrap_or_default();
    self.residents.remove(tp);
    let mut merge = Merge::new(*tp, state.content);
    merge.target = state.target;
    merge.data = MergeData {
      occurs_in,
      freq: state.freq,
    };
    self.debug_assert_residency();
    Some(merge)
  }

  /// Put back a selected pair when an automatic stopping policy declines it.
  pub(crate) fn restore_pair(&mut self, merge: Merge<C, I>) {
    debug_assert!(merge.target.is_none());
    let tp = merge.tp;
    let occurs_in = self.occurs_in.insert(merge.data.occurs_in);
    let previous = self.pairs.insert(tp, PairState {
      content: merge.content,
      target: None,
      freq: merge.data.freq,
      occurs_in: Some(occurs_in),
    });
    debug_assert!(previous.is_none());
    if self.is_bounded() {
      self.residents.insert(tp);
      self.observe_peak();
    }
    self.debug_assert_residency();
  }

  #[cfg(test)]
  pub(crate) fn clone_merge(&self, tp: &Pair<I>) -> Option<Merge<C, I>>
  where
    C: Clone,
  {
    let state = self.pairs.get(tp)?;
    let occurs_in = state
      .occurs_in
      .map(|id| self.occurs_in.get(id).clone())
      .unwrap_or_default();
    let mut merge = Merge::new(*tp, state.content.clone());
    merge.target = state.target;
    merge.data = MergeData {
      occurs_in,
      freq: state.freq,
    };
    Some(merge)
  }

  fn release(&mut self, tp: &Pair<I>) {
    let state = self.pairs.get_mut(tp).expect("resident pair is missing");
    let id = state.occurs_in.take().expect("resident pair has no occurs_in set");
    self.residents.remove(tp);
    drop(self.occurs_in.remove(id));
  }

  fn observe_peak(&mut self) {
    self.stats.peak_resident_pairs = self.stats.peak_resident_pairs.max(self.residents.len());
  }

  fn debug_assert_residency(&self) {
    #[cfg(debug_assertions)]
    if self.is_bounded() {
      debug_assert!(self.residents.iter().all(|tp| {
        self.pairs.get(tp).is_some_and(|state| state.occurs_in.is_some())
      }));
      debug_assert!(self.pairs.iter().all(|(tp, state)| {
        state.occurs_in.is_some() == self.residents.contains(tp)
      }));
    }
  }

  #[cfg(test)]
  pub(crate) fn force_all_resident_empty(&mut self) {
    assert!(self.is_bounded());
    let pairs = self
      .pairs
      .iter()
      .filter(|(_, state)| state.target.is_none())
      .map(|(tp, _)| *tp)
      .collect::<Vec<_>>();
    for tp in pairs {
      if !self.is_resident(&tp) {
        let id = self.occurs_in.insert(AHashSet::new());
        self.pairs.get_mut(&tp).unwrap().occurs_in = Some(id);
        self.residents.insert(tp);
      }
    }
    self.debug_assert_residency();
  }
}

#[cfg(test)]
mod tests {
  use std::mem::size_of;

  use super::*;

  #[test]
  fn occurs_in_id_is_compact_and_slab_keys_are_reused() {
    assert_eq!(size_of::<Option<OccursInId>>(), size_of::<usize>());

    let mut store = OccursInStore::default();
    let first = store.insert([1].into_iter().collect());
    assert_eq!(store.remove(first).into_iter().collect::<Vec<_>>(), [1]);

    let reused = store.insert([2].into_iter().collect());
    assert_eq!(reused, first);
    assert_eq!(store.get(reused).iter().copied().collect::<Vec<_>>(), [2]);
  }

  #[test]
  fn pair_store_clears_handles_before_reusing_slab_slots() {
    let mut pairs = PairStore::<u8, u32>::new(None);
    pairs.add_pair((1, 2), || (vec![1].into(), vec![2].into()), 3, [7]);
    let first_id = pairs.get(&(1, 2)).unwrap().occurs_in.unwrap();

    let selected = pairs.take_merge(&(1, 2)).unwrap();
    assert_eq!(selected.data.occurs_in_vec(), [7]);
    assert!(!pairs.contains_key(&(1, 2)));

    pairs.add_pair((3, 4), || (vec![3].into(), vec![4].into()), 2, [9]);
    let reused_id = pairs.get(&(3, 4)).unwrap().occurs_in.unwrap();
    assert_eq!(reused_id, first_id);
    assert_eq!(pairs.occurrence_vec(&(3, 4)), [9]);
  }
}
