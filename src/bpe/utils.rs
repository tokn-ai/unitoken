use std::{collections::{BTreeMap, HashMap, hash_map}, hash::Hash};

use ahash::AHashMap;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator as _};

use crate::{MyError, MyResult, spec::WordDisplay, traits::CanStrToWord};

use super::*;

pub trait ToWord<C> {
  fn to_word(self) -> Word<C>;
}

impl<C> ToWord<C> for Vec<C> {
  fn to_word(self) -> Word<C> {
    Arc::from(self.into_boxed_slice())
  }
}

impl<C: Clone> ToWord<C> for &[C] {
  fn to_word(self) -> Word<C> {
    Arc::from(self.to_owned().into_boxed_slice())
  }
}

impl ToWord<u8> for &str {
  fn to_word(self) -> Word<u8> {
    Arc::from(self.as_bytes().to_owned().into_boxed_slice())
  }
}

impl ToWord<Character> for &str {
  fn to_word(self) -> Word<Character> {
    let chars = self.chars().map(|ch| Character::Unicode(ch)).collect::<Vec<_>>();
    Arc::from(chars.into_boxed_slice())
  }
}

impl ToWord<u8> for u8 {
  fn to_word(self) -> Word<u8> {
    Arc::from(vec![self].into_boxed_slice())
  }
}

impl ToWord<Character> for u8 {
  fn to_word(self) -> Word<Character> {
    Arc::from(vec![Character::Byte(self)].into_boxed_slice())
  }
}

impl ToWord<Character> for char {
  fn to_word(self) -> Word<Character> {
    Arc::from(vec![Character::Unicode(self)].into_boxed_slice())
  }
}

pub trait WordDebugExt {
  fn debug_display(&self) -> String;
  fn to_string_lossy(&self) -> String;
}

impl WordDebugExt for Word<u8> {
  fn debug_display(&self) -> String {
    crate::spec::unitoken::UnitokenSpec.word_display(self)
  }
  fn to_string_lossy(&self) -> String {
    String::from_utf8_lossy(self).to_string()
  }
}

impl WordDebugExt for Word<Character> {
  fn debug_display(&self) -> String {
    self
      .iter()
      .map(|c| match c {
        Character::Unicode(ch) => ch.to_string(),
        Character::Byte(b) => format!("\\x{:02x}", *b),
      })
      .collect()
  }
  fn to_string_lossy(&self) -> String {
    let mut buffer = Vec::new();
    self.iter().for_each(|c| match c {
      Character::Unicode(ch) => buffer.extend_from_slice(ch.to_string().as_bytes()),
      Character::Byte(b) => buffer.extend_from_slice(&[*b]),
    });
    String::from_utf8_lossy(&buffer).to_string()
  }
}

struct LocalMergeData {
  freq: Freq,
}

impl LocalMergeData {
  fn new(freq: Freq) -> Self {
    Self {
      freq,
    }
  }
}

struct MergeWordUpdate<I> {
  word_idx: usize,
  idxs: Vec<I>,
  changes: Vec<((I, I), LocalMergeData)>,
}

struct MergeBatchUpdate<I> {
  word_updates: Vec<(usize, Vec<I>)>,
  changes: Vec<(usize, (I, I), Freq)>,
}

const LARGE_WORD_DICT_THRESHOLD: usize = 1_000_000;
const LARGE_DICT_PARALLEL_MERGE_OCCURS_IN_THRESHOLD: usize = 1024;
const SMALL_DICT_PARALLEL_MERGE_OCCURS_IN_THRESHOLD: usize = 64 * 1024;

pub(crate) fn should_parallel_merge(
  words_len: usize,
  occurs_in_len: usize,
  parallel_merge_min_occurs_in: Option<usize>,
) -> bool {
  if let Some(threshold) = parallel_merge_min_occurs_in {
    return occurs_in_len >= threshold;
  }
  let threshold = if words_len >= LARGE_WORD_DICT_THRESHOLD {
    LARGE_DICT_PARALLEL_MERGE_OCCURS_IN_THRESHOLD
  } else {
    SMALL_DICT_PARALLEL_MERGE_OCCURS_IN_THRESHOLD
  };
  occurs_in_len >= threshold
}

fn add_local_change<I: Eq + Copy>(
  changes: &mut Vec<((I, I), LocalMergeData)>,
  tp: (I, I),
  freq_delta: Freq,
) {
  if let Some((_, data)) = changes.iter_mut().find(|(existing, _)| *existing == tp) {
    data.freq += freq_delta;
  } else {
    changes.push((tp, LocalMergeData::new(freq_delta)));
  }
}

fn merge_word<I>(
  word_idx: usize,
  w_idx: &[I],
  w_freq: Freq,
  merge_tp: (I, I),
  target_idx: I,
) -> Option<MergeWordUpdate<I>>
where
  I: Ord + Copy,
{
  let mut changes = Vec::<((I, I), LocalMergeData)>::with_capacity(8);
  let mut new_idxs = Vec::with_capacity(w_idx.len());
  let mut i = 0;
  let mut last_tp: Option<(I, I)> = None;
  let mut did_merge = false;
  while i + 1 < w_idx.len() {
    let tp = (w_idx[i], w_idx[i + 1]);
    if tp == merge_tp {
      did_merge = true;
      new_idxs.push(target_idx);
      i += 2;
      add_local_change(&mut changes, tp, -w_freq);
      // deal with left neighbor,
      // e.g. in "abcd", when merging "b" and "c",
      // old_tp = ("a", "b"), new_tp = ("a", "bc")
      if let Some(old_tp) = last_tp {
        let new_tp = (old_tp.0, target_idx);
        add_local_change(&mut changes, old_tp, -w_freq);
        add_local_change(&mut changes, new_tp, w_freq);
        // if i >= w_idx.len(), loop is end, and last_tp never reads
        // last_tp = Some(new_tp);
      }
      // deal with right neighbor, notice i+=2 above
      // e.g. in "abcd", when merging "b" and "c",
      // old_tp = ("c", "d"), new_tp = ("bc", "d")
      if i < w_idx.len() {
        let old_tp = (tp.1, w_idx[i]);
        let new_tp = (target_idx, old_tp.1);
        add_local_change(&mut changes, old_tp, -w_freq);
        add_local_change(&mut changes, new_tp, w_freq);
        last_tp = Some(new_tp);
      }
    } else {
      new_idxs.push(w_idx[i]);
      last_tp = Some(tp);
      i += 1;
    }
  }
  if i < w_idx.len() {
    new_idxs.push(w_idx[i]);
  }

  // `occurs_in` is maintained lazily, so it may contain stale word ids.
  // Stale affected words should not rewrite or emit zero deltas.
  if !did_merge {
    return None;
  }

  Some(MergeWordUpdate {
    word_idx,
    idxs: new_idxs,
    changes,
  })
}

fn merge_flat_changes<I>(changes: Vec<(usize, (I, I), Freq)>) -> AHashMap<(I, I), MergeData>
where
  I: Eq + Hash,
{
  let mut merged = AHashMap::<(I, I), MergeData>::new();
  for (word_idx, tp, freq) in changes {
    let data = merged.entry(tp).or_default();
    data.freq += freq;
    if freq > 0 {
      data.occurs_in.insert(word_idx as _);
    }
  }
  merged
}

fn merge_words_sequential<C, I>(
  words: &mut Vec<PreToken<C, I>>,
  affected_words: impl IntoIterator<Item = usize>,
  merge_tp: (I, I),
  target_idx: I,
) -> AHashMap<(I, I), MergeData>
where
  I: Ord + Copy + Hash,
{
  let mut changes = Vec::new();
  for word_idx in affected_words {
    let word = &mut words[word_idx];
    if let Some(update) = merge_word(word_idx, &word.idxs, word.freq, merge_tp, target_idx) {
      word.idxs = update.idxs;
      changes.extend(update.changes.into_iter().map(|(tp, data)| {
        (update.word_idx, tp, data.freq)
      }));
    }
  }
  merge_flat_changes(changes)
}

pub(crate) fn _merge<C, I>(
  words: &mut Vec<PreToken<C, I>>,
  merge: &Merge<C, I>,
  target_idx: I,
  parallel_merge_min_occurs_in: Option<usize>,
) -> AHashMap<(I, I), MergeData>
where
  C: Send + Sync,
  I: Ord + Copy + Hash + Send + Sync,
{
  // Frequencies stay exact, but affected-word sets are lazy: positive deltas
  // add possible word ids, negative deltas do not remove stale ids. Stale ids
  // are filtered by `merge_word` when the selected pair is no longer present.
  if !should_parallel_merge(words.len(), merge.data.occurs_in.len(), parallel_merge_min_occurs_in) {
    let affected_words = merge.data.occurs_in.iter().copied().map(|i| i as usize);
    return merge_words_sequential(words, affected_words, merge.tp, target_idx);
  }

  let update = merge
    .data
    .occurs_in
    .par_iter()
    .fold(
      || MergeBatchUpdate {
        word_updates: Vec::new(),
        changes: Vec::new(),
      },
      |mut batch, &word_idx| {
        let word_idx = word_idx as usize;
        let word = &words[word_idx];
        if let Some(update) = merge_word(word_idx, &word.idxs, word.freq, merge.tp, target_idx) {
          batch.word_updates.push((update.word_idx, update.idxs));
          batch.changes.extend(update.changes.into_iter().map(|(tp, data)| {
            (update.word_idx, tp, data.freq)
          }));
        }
        batch
      },
    )
    .reduce(
      || MergeBatchUpdate {
        word_updates: Vec::new(),
        changes: Vec::new(),
      },
      |mut left, mut right| {
        left.word_updates.append(&mut right.word_updates);
        left.changes.append(&mut right.changes);
        left
      },
    );

  for (word_idx, idxs) in update.word_updates {
    words[word_idx].idxs = idxs;
  }
  merge_flat_changes(update.changes)
}

pub(crate) fn _vocab_get<C, I>(vocab: &BTreeMap<I, Word<C>>, idx: I) -> MyResult<Word<C>>
where
  C: CanStrToWord,
  I: IdxLike + HasChar<C>,
{
  vocab.get(&idx).cloned().or_else(|| idx.idx_to_word()).ok_or_else(|| MyError::OovIdx(idx.to_u64()))
}

pub(crate) fn _update_merge_map<C, I>(
  merge_map: &mut HashMap<(I, I), Merge<C, I>>, merge: &Merge<C, I>,
  changes: AHashMap<(I, I), MergeData>, vocab: Option<&BTreeMap<I, Word<C>>>,
  update_occurrences: bool,
) -> Vec<(I, I)>
where
  I: IdxLike + HasChar<C> + Hash,
  C: CanStrToWord,
  Word<C>: WordDebugExt,
{
  let mut updated_tps = Vec::new();
  for (tp, data) in changes {
    if tp == merge.tp {
      continue;
    }
    if data.freq == 0 {
      continue;
    }
    let entry = merge_map.entry(tp);
    let entry = match entry {
      hash_map::Entry::Occupied(e) => e.into_mut(),
      hash_map::Entry::Vacant(e) => {
        if let Some(vocab) = vocab {
          let content = (
            _vocab_get(vocab, tp.0).unwrap(),
            _vocab_get(vocab, tp.1).unwrap(),
          );
          e.insert(Merge::new(tp, content))
        } else {
          continue;
        }
      }
    };
    // println!("  Change {:?} {:?} {:?}: freq {} -> {}", tp, entry.content.0.display(), entry.content.1.display(), entry.data.freq, entry.data.freq + data.freq);
    entry.data.freq += data.freq;
    // Keep affected-word sets lazy: positive deltas add possible positions,
    // negative deltas only repair frequency and leave stale positions behind.
    if update_occurrences && data.freq > 0 {
      entry.data.occurs_in.extend(data.occurs_in);
    }
    updated_tps.push(tp);
  }
  updated_tps
}
