use std::{cmp::Ordering, collections::{BinaryHeap, BTreeMap, BTreeSet, HashMap, hash_map::Entry}, hash::Hash, ops::Range, sync::atomic::AtomicU64};

use ahash::{AHashMap, AHashSet};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{MyError, MyResult, traits::{CanStrToWord, CanToWord, CanTrain, Train}};

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitialAlphabet {
  /// Insert bytes in raw byte order, preserving GPT-2/tiktoken-compatible ids.
  RawBytes,
  /// Insert bytes in Hugging Face ByteLevel alphabet order.
  ByteLevel,
}

impl Default for InitialAlphabet {
  fn default() -> Self {
    Self::RawBytes
  }
}

impl InitialAlphabet {
  fn bytes(self) -> Vec<u8> {
    match self {
      Self::RawBytes => (0u8..=255).collect(),
      Self::ByteLevel => byte_level_alphabet_bytes(),
    }
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TieBreak {
  /// Resolve equal frequencies by the smallest pair ids, matching Hugging Face BPE.
  /// Initial units are prioritized before this pair ordering is applied.
  SmallestPairId,
  /// Resolve equal frequencies by lexicographically largest token content.
  /// Initial units are prioritized before this content ordering is applied.
  LargestContent,
}

impl Default for TieBreak {
  fn default() -> Self {
    Self::SmallestPairId
  }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BpeTrainerConfig {
  pub initial_alphabet: InitialAlphabet,
  pub tie_break: TieBreak,
  /// Override the `occurs_in` cutoff for Rayon merge rewrites.
  ///
  /// `None` keeps the built-in heuristic: high cutoff for small word
  /// dictionaries and lower cutoff for very large dictionaries.
  pub parallel_merge_min_occurs_in: Option<usize>,
}

impl BpeTrainerConfig {
  pub fn hf_byte_level() -> Self {
    Self {
      initial_alphabet: InitialAlphabet::ByteLevel,
      tie_break: TieBreak::SmallestPairId,
      parallel_merge_min_occurs_in: None,
    }
  }
}

fn byte_level_alphabet_bytes() -> Vec<u8> {
  let mut pairs = (0u8..=255)
    .map(|byte| (byte, byte_to_unicode(byte)))
    .collect::<Vec<_>>();
  pairs.sort_by_key(|(_, ch)| *ch);
  pairs.into_iter().map(|(byte, _)| byte).collect()
}

fn byte_to_unicode(byte: u8) -> char {
  if (b'!'..=b'~').contains(&byte)
    || (0xA1..=0xAC).contains(&byte)
    || (0xAE..=0xFF).contains(&byte)
  {
    return byte as char;
  }
  let mut n = 0u32;
  for b in 0u8..=255 {
    if (b'!'..=b'~').contains(&b)
      || (0xA1..=0xAC).contains(&b)
      || (0xAE..=0xFF).contains(&b)
    {
      continue;
    }
    if b == byte {
      return char::from_u32(256 + n).unwrap();
    }
    n += 1;
  }
  unreachable!("all bytes are covered")
}

#[derive(Debug, Default)]
pub struct BpeTrainer<C, I> {
  pub start_vocab_idx: AtomicU64,
  pub _byte_vocab_start_idx: Option<u64>,
  pub byte_vocab: HashMap<u8, I>,
  pub config: BpeTrainerConfig,
  pub special_tokens: Vec<String>,
  pub vocab: BTreeMap<I, Word<C>>,
  pub merges: Vec<Merge<C, I>>,
  pub pre_merges: HashMap<(I, I), Merge<C, I>>,
  merge_heap: BinaryHeap<MergeCandidate<C, I>>,
  pub words: Vec<PreToken<C, I>>,
}

#[derive(Debug, Clone)]
struct MergeCandidate<C, I> {
  freq: Freq,
  tp: (I, I),
  content: Option<(Word<C>, Word<C>)>,
  kind: MergeCandidateKind,
  tie_break: TieBreak,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum MergeCandidateKind {
  Pair,
  InitialUnit,
}

impl<C, I> MergeCandidate<C, I> {
  fn from_merge(merge: &Merge<C, I>, tie_break: TieBreak) -> Self
  where
    I: Copy,
  {
    Self {
      freq: merge.data.freq,
      tp: merge.tp,
      content: match tie_break {
        TieBreak::SmallestPairId => None,
        TieBreak::LargestContent => Some(merge.content.clone()),
      },
      kind: if merge.target.is_some() {
        MergeCandidateKind::InitialUnit
      } else {
        MergeCandidateKind::Pair
      },
      tie_break,
    }
  }
}

impl<C: Ord, I: Ord> Ord for MergeCandidate<C, I> {
  fn cmp(&self, other: &Self) -> Ordering {
    self
      .freq
      .cmp(&other.freq)
      // Initial units must be available before pair merges. Prefer them on an
      // equal-frequency boundary, then apply the configured tie-break.
      .then_with(|| self.kind.cmp(&other.kind))
      .then_with(|| match self.tie_break {
        TieBreak::SmallestPairId => other.tp.cmp(&self.tp),
        TieBreak::LargestContent => self.content.as_ref().unwrap().cmp(other.content.as_ref().unwrap()),
      })
  }
}

impl<C: Ord, I: Ord> PartialOrd for MergeCandidate<C, I> {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

impl<C: Ord, I: Ord> PartialEq for MergeCandidate<C, I> {
  fn eq(&self, other: &Self) -> bool {
    self.cmp(other) == Ordering::Equal
  }
}

impl<C: Ord, I: Ord> Eq for MergeCandidate<C, I> {}

// Small inventories do not amortize the range scan and local-map reduction.
const PARALLEL_INIT_MIN_WORDS: usize = 100_000;
// Balance work by encoded units rather than words because pretoken lengths vary widely.
const PARALLEL_INIT_CHUNK_UNITS: usize = 256 * 1024;
// Bound the temporary local maps retained before each serial reduction.
const PARALLEL_INIT_MAX_BATCHES: usize = 16;

#[derive(Default)]
struct PairBuildData {
  freq: Freq,
  occurs_in: Vec<u64>,
}

struct PreMergeBuildBatch<I> {
  initial_freqs: AHashMap<I, Freq>,
  pairs: AHashMap<(I, I), PairBuildData>,
}

struct PreMergeBuildRange {
  words: Range<usize>,
  units: usize,
}

impl<I> Default for PreMergeBuildBatch<I> {
  fn default() -> Self {
    Self {
      initial_freqs: AHashMap::new(),
      pairs: AHashMap::new(),
    }
  }
}

fn collect_pre_merge_build_batch<C, I>(
  words: &[PreToken<C, I>], word_offset: usize, vocab: &BTreeMap<I, Word<C>>,
) -> PreMergeBuildBatch<I>
where
  I: Copy + Eq + Hash + Ord,
{
  let mut batch = PreMergeBuildBatch::default();
  for (local_word_idx, word) in words.iter().enumerate() {
    let word_idx = (word_offset + local_word_idx) as u64;
    for unit in word.idxs.iter().copied() {
      if !vocab.contains_key(&unit) {
        *batch.initial_freqs.entry(unit).or_default() += word.freq;
      }
    }
    for tp in word.idxs.iter().copied().zip(word.idxs.iter().skip(1).copied()) {
      let entry = batch.pairs.entry(tp).or_default();
      entry.freq += word.freq;
      // A pair records each word once, even when it occurs repeatedly in that word.
      if entry.occurs_in.last().copied() != Some(word_idx) {
        entry.occurs_in.push(word_idx);
      }
    }
  }
  batch
}

fn pre_merge_build_ranges<C, I>(
  words: &[PreToken<C, I>], max_units: usize,
) -> Vec<PreMergeBuildRange> {
  let max_units = max_units.max(1);
  let mut ranges = Vec::new();
  let mut start = 0;
  let mut units = 0usize;
  for (word_idx, word) in words.iter().enumerate() {
    let word_units = word.idxs.len().max(1);
    if word_idx > start && units.saturating_add(word_units) > max_units {
      ranges.push(PreMergeBuildRange {
        words: start..word_idx,
        units,
      });
      start = word_idx;
      units = 0;
    }
    units = units.saturating_add(word_units);
  }
  if start < words.len() {
    ranges.push(PreMergeBuildRange {
      words: start..words.len(),
      units,
    });
  }
  ranges
}

impl<C, I: IdxLike> BpeTrainer<C, I>
where
  Word<C>: WordDebugExt,
  C: CanStrToWord + CanToWord<u8>,
{
  /// Build a trainer from `(word, frequency)` pairs.
  ///
  /// - `special_tokens` are reserved at the start of the vocabulary.
  /// - Words equal to any special token are skipped.
  pub fn from_words<Iter: IntoIterator<Item = (S, Freq)>, S: AsRef<str>>(words: Iter, special_tokens: &[String]) -> Self
  where
    C: CharToIdx<I>,
    I: HasChar<C>,
  {
    Self::from_words_with_config(words, special_tokens, BpeTrainerConfig::default())
  }

  pub fn from_words_with_config<Iter: IntoIterator<Item = (S, Freq)>, S: AsRef<str>>(
    words: Iter, special_tokens: &[String], config: BpeTrainerConfig,
  ) -> Self
  where
    C: CharToIdx<I>,
    I: HasChar<C>,
  {
    let vocab_start_idx = special_tokens.len() as u64;
    let sp_set = special_tokens.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let tokens = Self::_words_to_tokens(words, vocab_start_idx, &sp_set, None);
    Self::new_with_config(tokens, special_tokens.to_vec(), config)
  }

  /// Create a trainer from already pre-tokenized words.
  ///
  /// This initializes vocab with `special_tokens` and a 256-entry byte vocabulary.
  pub fn new(words: Vec<PreToken<C, I>>, special_tokens: Vec<String>) -> Self {
    Self::new_with_config(words, special_tokens, BpeTrainerConfig::default())
  }

  pub fn new_with_config(words: Vec<PreToken<C, I>>, special_tokens: Vec<String>, config: BpeTrainerConfig) -> Self {
    let mut bpe = Self::empty();
    bpe.config = config;
    bpe._vocab_insert_special_tokens(special_tokens);
    bpe._vocab_insert_all_single_byte();
    bpe.words = words;
    bpe
  }

  /// Insert the full single-byte vocabulary (0..=255) into `self.vocab`.
  ///
  /// Returns the next available vocab index.
  pub fn _vocab_insert_all_single_byte(&mut self) -> I {
    let start_idx = self.start_vocab_idx.fetch_add(256, std::sync::atomic::Ordering::AcqRel);
    let vocab = &mut self.vocab;
    self.byte_vocab.clear();
    for (offset, byte) in self.config.initial_alphabet.bytes().into_iter().enumerate() {
      let idx = I::from_u64(offset as u64 + start_idx);
      if byte < 128 {
        vocab.insert(idx, (byte as char).to_string().to_word());
      } else {
        vocab.insert(idx, byte.to_word());
      }
      self.byte_vocab.insert(byte, idx);
    }
    self._byte_vocab_start_idx = Some(start_idx);
    I::from_u64(start_idx + 256)
  }

  /// Convert `(word, frequency)` input into [`PreToken`]s.
  ///
  /// Words that match `special_tokens` are skipped.
  pub fn _words_to_tokens<Iter: IntoIterator<Item = (S, Freq)>, S: AsRef<str>>(
    words: Iter, vocab_start_idx: u64, special_tokens: &BTreeSet<&str>, byte_vocab: Option<&HashMap<u8, I>>,
  ) -> Vec<PreToken<C, I>>
  where
    C: CharToIdx<I>,
  {
    let mut tokens = Vec::new();
    for (w, freq) in words.into_iter() {
      let w = w.as_ref();
      if special_tokens.contains(w) {
        continue;
      }
      let src = w.to_word();
      let idxs = src.iter().map(|b| b.char_to_idx(vocab_start_idx, byte_vocab)).collect::<Vec<_>>();
      let pre_token = PreToken {
        src: src.clone(),
        idxs,
        freq: freq as Freq,
      };
      tokens.push(pre_token);
    }
    tokens
  }
}

impl<C: CanStrToWord, I: IdxLike> BpeTrainer<C, I>
where
  Word<C>: WordDebugExt,
{
  /// Insert special tokens at the start of the vocabulary.
  ///
  /// Returns the next available vocab index.
  pub fn _vocab_insert_special_tokens(&mut self, special_tokens: Vec<String>) -> I {
    let length = special_tokens.len();
    let start_idx = self.start_vocab_idx.fetch_add(length as u64, std::sync::atomic::Ordering::AcqRel);
    let vocab = &mut self.vocab;
    for (i, token) in special_tokens.iter().enumerate() {
      vocab.insert(I::from_u64(i as u64 + start_idx), token.as_str().to_word());
    }
    self.special_tokens.extend(special_tokens);
    I::from_u64(start_idx + length as u64)
  }

  /// Validate the current state and return an immutable model snapshot.
  ///
  /// Merge targets are removed from the initial vocabulary, then introduced by
  /// replaying merges in rank order. This catches missing operands and merges
  /// emitted before their dependencies.
  pub fn validate_model(&self) -> MyResult<BpeModel<C, I>>
  where
    C: CharSplit + Clone + Ord,
    I: HasChar<C>,
  {
    let mut vocab_contents = BTreeSet::new();
    for token in self.vocab.values() {
      let normalized = C::from_vec_u8(&C::to_vec_u8(token));
      if !vocab_contents.insert(normalized) {
        return Err(MyError::InvalidBpeModel(format!(
          "duplicate vocabulary token {}",
          token.debug_display(),
        )));
      }
    }
    drop(vocab_contents);

    let mut target_ids = BTreeSet::new();
    let mut outputs = Vec::with_capacity(self.merges.len());
    for (rank, merge) in self.merges.iter().enumerate() {
      for (side, idx, expected) in [
        ("left", merge.tp.0, &merge.content.0),
        ("right", merge.tp.1, &merge.content.1),
      ] {
        let Some(actual) = self.vocab.get(&idx).cloned().or_else(|| idx.idx_to_word()) else {
          return Err(MyError::InvalidBpeModel(format!(
            "merge {rank} {side} operand id is missing from the vocabulary",
          )));
        };
        if &actual != expected {
          return Err(MyError::InvalidBpeModel(format!(
            "merge {rank} {side} operand id resolves to {}, expected {}",
            actual.debug_display(),
            expected.debug_display(),
          )));
        }
      }

      let Some(target) = merge.target else {
        return Err(MyError::InvalidBpeModel(format!(
          "merge {rank} has no target",
        )));
      };
      if !target_ids.insert(target) {
        return Err(MyError::InvalidBpeModel(format!(
          "merge {rank} reuses an earlier target",
        )));
      }

      let output = merge.merged_content();
      let Some(target_content) = self.vocab.get(&target) else {
        return Err(MyError::InvalidBpeModel(format!(
          "merge {rank} target is missing from the vocabulary",
        )));
      };
      if target_content != &output {
        return Err(MyError::InvalidBpeModel(format!(
          "merge {rank} target is {}, expected {}",
          target_content.debug_display(),
          output.debug_display(),
        )));
      }
      outputs.push(output);
    }

    let mut available = self
      .vocab
      .iter()
      .filter(|(idx, _)| !target_ids.contains(idx))
      .map(|(_, token)| token.clone())
      .collect::<BTreeSet<_>>();

    for (rank, (merge, output)) in self.merges.iter().zip(outputs).enumerate() {
      if !available.contains(&merge.content.0) {
        return Err(MyError::InvalidBpeModel(format!(
          "merge {rank} left operand {} is not an initial token or an earlier merge",
          merge.content.0.debug_display(),
        )));
      }
      if !available.contains(&merge.content.1) {
        return Err(MyError::InvalidBpeModel(format!(
          "merge {rank} right operand {} is not an initial token or an earlier merge",
          merge.content.1.debug_display(),
        )));
      }
      if !available.insert(output.clone()) {
        return Err(MyError::InvalidBpeModel(format!(
          "merge {rank} does not introduce a new token {}",
          output.debug_display(),
        )));
      }
    }

    let vocab_by_content = self
      .vocab
      .iter()
      .map(|(idx, token)| (token.clone(), *idx))
      .collect::<BTreeMap<_, _>>();
    let merges = self
      .merges
      .iter()
      .map(|merge| {
        let tp = (
          *vocab_by_content.get(&merge.content.0).unwrap(),
          *vocab_by_content.get(&merge.content.1).unwrap(),
        );
        let mut model_merge = Merge::new(tp, merge.content.clone()).with_target(merge.target.unwrap());
        model_merge.data.freq = merge.data.freq;
        model_merge
      })
      .collect();
    Ok(BpeModel::new(
      self.special_tokens.clone(),
      self.vocab.clone(),
      merges,
    ))
  }
}

impl<C, I> BpeTrainer<C, I> {
  /// Construct an empty trainer with no vocab, merges, or words.
  pub fn empty() -> Self {
    Self {
      start_vocab_idx: AtomicU64::new(0),
      _byte_vocab_start_idx: None,
      byte_vocab: HashMap::new(),
      config: BpeTrainerConfig::default(),
      vocab: BTreeMap::new(),
      merges: Vec::new(),
      pre_merges: HashMap::new(),
      merge_heap: BinaryHeap::new(),
      special_tokens: Vec::new(),
      words: Vec::new(),
    }
  }
}

impl<C, I: IdxLike> BpeTrainer<C, I>
where
  Word<C>: WordDebugExt,
  I: HasChar<C>,
  C: CanStrToWord + Ord + Send + Sync,
{
  /// Initialize the merge candidate map from `self.words`.
  ///
  /// This computes merge frequencies and document-occurrence sets used by [`Train::step`].
  pub fn _build_pre_merges(&mut self) {
    let parallel = self.words.len() >= PARALLEL_INIT_MIN_WORDS && rayon::current_num_threads() > 1;
    self._build_pre_merges_with_options(parallel, PARALLEL_INIT_CHUNK_UNITS);
  }

  fn _build_pre_merges_with_options(&mut self, parallel: bool, chunk_units: usize) {
    debug!("Initializing BPE training with {} words", self.words.len());
    self.pre_merges.clear();
    self.merge_heap.clear();
    if parallel {
      self._build_pre_merges_parallel(chunk_units);
    } else {
      self._build_pre_merges_sequential();
    }
    self.rebuild_merge_heap();
  }

  fn _build_pre_merges_sequential(&mut self) {
    let mut vocab_contents = None;
    let mut materialized_initial_units = AHashSet::new();
    let vocab_get = |i: I| {
      self.vocab.get(&i).cloned().or_else(|| i.idx_to_word()).ok_or_else(|| MyError::OovIdx(i.to_u64()))
    };
    let i_none = I::from_u64(u64::MAX);
    let empty_word = Vec::<C>::new().to_word();
    for (word_idx, word) in self.words.iter().enumerate() {
      for unit in word.idxs.iter().copied() {
        if self.vocab.contains_key(&unit) || materialized_initial_units.contains(&unit) {
          continue;
        }
        let tp = (i_none, unit);
        if let Some(merge) = self.pre_merges.get_mut(&tp) {
          merge.data.freq += word.freq;
          continue;
        }

        let content = vocab_get(unit).unwrap();
        // Unicode units remain CharIdx::Char in the word inventory after they
        // receive a numeric vocab id, so compare content when training resumes.
        if vocab_contents
          .get_or_insert_with(|| self.vocab.values().cloned().collect::<BTreeSet<_>>())
          .contains(&content)
        {
          materialized_initial_units.insert(unit);
          continue;
        }
        let mut merge = Merge::new(tp, (empty_word.clone(), content)).with_target(unit);
        merge.data.freq = word.freq;
        self.pre_merges.insert(tp, merge);
      }
      for tp in word.idxs.iter().copied().zip(word.idxs.iter().skip(1).copied()) {
        let merge = self.pre_merges.entry(tp).or_insert_with(|| {
          Merge::new(tp, (vocab_get(tp.0).unwrap(), vocab_get(tp.1).unwrap()))
        });
        merge.add(word_idx as u64, word.freq);
      }
    }
  }

  fn _build_pre_merges_parallel(&mut self, chunk_units: usize) {
    let chunk_units = chunk_units.max(1);
    let ranges = pre_merge_build_ranges(&self.words, chunk_units);
    let ranges_per_wave = rayon::current_num_threads().clamp(1, PARALLEL_INIT_MAX_BATCHES);
    let i_none = I::from_u64(u64::MAX);
    let empty_word = Vec::<C>::new().to_word();
    let vocab = &self.vocab;
    let vocab_get = |i: I| {
      vocab.get(&i).cloned().or_else(|| i.idx_to_word()).ok_or_else(|| MyError::OovIdx(i.to_u64()))
    };
    let mut vocab_contents = None;
    let mut materialized_initial_units = AHashSet::new();

    let mut range_idx = 0;
    while range_idx < ranges.len() {
      // Process bounded waves so temporary maps do not scale with the whole corpus.
      // A single oversized word cannot be split without changing its occurrence id.
      let wave_len = if ranges[range_idx].units > chunk_units {
        1
      } else {
        ranges[range_idx..]
          .iter()
          .take(ranges_per_wave)
          .take_while(|range| range.units <= chunk_units)
          .count()
          .max(1)
      };
      let range_wave = &ranges[range_idx..range_idx + wave_len];
      let batches = range_wave
        .par_iter()
        .map(|range| {
          collect_pre_merge_build_batch(
            &self.words[range.words.clone()],
            range.words.start,
            vocab,
          )
        })
        .collect::<Vec<_>>();

      for batch in batches {
        for (unit, freq) in batch.initial_freqs {
          if materialized_initial_units.contains(&unit) {
            continue;
          }
          let tp = (i_none, unit);
          match self.pre_merges.entry(tp) {
            Entry::Occupied(mut entry) => entry.get_mut().data.freq += freq,
            Entry::Vacant(entry) => {
              let content = vocab_get(unit).unwrap();
              if vocab_contents
                .get_or_insert_with(|| vocab.values().cloned().collect::<BTreeSet<_>>())
                .contains(&content)
              {
                materialized_initial_units.insert(unit);
                continue;
              }
              let mut merge = Merge::new(tp, (empty_word.clone(), content)).with_target(unit);
              merge.data.freq = freq;
              entry.insert(merge);
            }
          }
        }

        for (tp, data) in batch.pairs {
          match self.pre_merges.entry(tp) {
            Entry::Occupied(mut entry) => {
              let merge = entry.get_mut();
              merge.data.freq += data.freq;
              merge.data.occurs_in.extend(data.occurs_in);
            }
            Entry::Vacant(entry) => {
              let mut merge = Merge::new(tp, (vocab_get(tp.0).unwrap(), vocab_get(tp.1).unwrap()));
              merge.data.freq = data.freq;
              merge.data.occurs_in.extend(data.occurs_in);
              entry.insert(merge);
            }
          }
        }
      }
      range_idx += wave_len;
    }
  }

  fn _set_vocab_idx(&mut self, start_idx: I) {
    self.start_vocab_idx.store(start_idx.to_u64(), std::sync::atomic::Ordering::Release);
  }

  fn _add_vocab_idx(&self) -> I {
    I::from_u64(self.start_vocab_idx.fetch_add(1, std::sync::atomic::Ordering::AcqRel))
  }

  fn push_merge_candidate(&mut self, tp: (I, I)) {
    let Some(merge) = self.pre_merges.get(&tp) else {
      return;
    };
    if merge.data.freq <= 0 {
      return;
    }
    self.merge_heap.push(MergeCandidate::from_merge(merge, self.config.tie_break));
  }

  fn rebuild_merge_heap(&mut self) {
    self.merge_heap = self
      .pre_merges
      .values()
      .filter(|merge| merge.data.freq > 0)
      .map(|merge| MergeCandidate::from_merge(merge, self.config.tie_break))
      .collect();
  }

  fn update_pre_merges(&mut self, merge: &Merge<C, I>, changes: AHashMap<(I, I), MergeData>) {
    let changed_tps = _update_merge_map(&mut self.pre_merges, merge, changes, Some(&self.vocab));
    for tp in changed_tps {
      self.push_merge_candidate(tp);
    }
  }

  fn merge(&mut self, merge: &Merge<C, I>, target_idx: I) -> AHashMap<(I, I), MergeData> {
    _merge(&mut self.words, merge, target_idx, self.config.parallel_merge_min_occurs_in)
  }

  fn _get_largest_merge(&mut self) -> Option<Merge<C, I>> {
    while let Some(candidate) = self.merge_heap.pop() {
      if candidate.freq <= 0 {
        continue;
      }
      let Some(merge) = self.pre_merges.get(&candidate.tp) else {
        continue;
      };
      if merge.data.freq != candidate.freq {
        continue;
      }
      if candidate.content.as_ref().is_some_and(|content| merge.content != *content) {
        continue;
      }

      return self.pre_merges.remove(&candidate.tp);
    }
    None
  }

  /// Apply one merge operation and return the newly assigned vocab index.
  ///
  /// This is the core training step once a merge candidate has been selected.
  #[hotpath::measure]
  pub fn _step(&mut self, merge: Merge<C, I>) -> I where C: Clone {
    let target_idx = self._add_vocab_idx();
    // if target = Some(j), this is a single char token, no need to merge.
    // but we have to add it to vocab.
    if merge.target.is_some() {
      self.vocab.insert(target_idx, merge.content.1.clone());
      return target_idx;
    }
    let changes = self.merge(&merge, target_idx);
    // println!("Merge {:?} (freq={}) into idx {}", merge.tp, merge.data.freq, target_idx);
    let merge = merge.with_target(target_idx);
    let merged_word = merge.merged_content();
    // self.vocab.entry(merge.tp.0).or_insert_with(|| merge.content.0.clone());
    // self.vocab.entry(merge.tp.1).or_insert_with(|| merge.content.1.clone());
    self.vocab.insert(target_idx, merged_word);
    assert_eq!(-changes.get(&merge.tp).map(|i| i.freq).unwrap_or(0), merge.data.freq);
    metrics::histogram!("bpe_trainer.changes").record(changes.len() as f64);
    self.update_pre_merges(&merge, changes);
    metrics::histogram!("bpe_trainer.occurs_in").record(merge.data.occurs_in.len() as f64);
    metrics::histogram!("bpe_trainer.freq").record(merge.data.freq as f64);
    self.merges.push(merge);
    target_idx
  }

  /// Convert a trained [`BpeTrainer`] into a [`BpeEncoder`].
  ///
  /// This re-encodes indices into the concrete `Idx` type used by encoders.
  pub fn finish(self) -> MyResult<BpeEncoder<C>>
  where
    C: Ord + Clone + Cachable + CharSplit,
  {
    let merges = self.merges
      .into_iter()
      .map(|m| {
        let tp = (m.tp.0.to_u64() as Idx, m.tp.1.to_u64() as Idx);
        let target = m.target.unwrap().to_u64() as Idx;
        (tp, target)
      })
      .collect();
    let vocab = self.vocab.into_iter().map(|(i, w)| (i.to_u64() as Idx, w)).collect();
    BpeEncoder::new(vocab, merges, self.special_tokens)
  }

  /// Emit internal metrics about the trainer state.
  pub fn _metrics(&self) {
    metrics::counter!("bpe_trainer.vocab_size").absolute(self.vocab.len() as u64);
    metrics::gauge!("bpe_trainer.pre_merges_count").set(self.pre_merges.len() as f64);
    metrics::gauge!("bpe_trainer.words_count").set(self.words.len() as f64);
  }

  /// Initialize trainer state and run merge steps until `vocab_size` is reached.
  #[hotpath::measure]
  pub fn train_until(&mut self, vocab_size: usize) -> MyResult<()>
  where
    C: Clone,
  {
    self._build_pre_merges();
    self._metrics();
    while self.vocab.len() < vocab_size {
      let Some(merge) = self._get_largest_merge() else {
        return Err(MyError::TrainStep);
      };
      self._step(merge);
      if self.vocab.len() % 100 == 0 {
        self._metrics();
      }
    }
    self._metrics();
    Ok(())
  }
}

impl<C, I> Train for BpeTrainer<C, I>
where
  Self: CanTrain<C, I>,
{
  fn new(special_tokens: Vec<String>) -> Self {
    Self::new(vec![], special_tokens)
  }

  fn add_words(&mut self, words: &mut dyn Iterator<Item = (&str, Freq)>) {
    let special_tokens = self.special_tokens.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let vocab_start_idx = self._byte_vocab_start_idx.unwrap();
    self.words = Self::_words_to_tokens(words, vocab_start_idx, &special_tokens, Some(&self.byte_vocab));
  }

  fn vocab_size(&self) -> usize {
    self.vocab.len()
  }

  fn init_training(&mut self) {
    self._build_pre_merges();
    self._metrics();
  }

  fn train(&mut self, vocab_size: usize) -> MyResult<()> {
    self.train_until(vocab_size)
  }

  #[hotpath::measure]
  fn step(&mut self) -> MyResult<()> {
    // Find the most frequent merge. Hugging Face's BPE trainer resolves equal
    // frequencies by choosing the smallest pair ids, so keep that ordering here.
    let merge = self._get_largest_merge();
    if let Some(merge) = merge {
      self._step(merge);
      if self.vocab_size() % 100 == 0 {
        self._metrics();
      }
      Ok(())
    } else {
      Err(MyError::TrainStep)
    }

  }
}


#[cfg(test)]
mod tests {
  use crate::{pretokenizer::DEFAULT_EOT, spec::gpt2::Gpt2Spec};

  use super::*;

  fn _test_bpe_merge(pretokens: &[(&str, Freq)], merges: &[((&str, &str), Vec<(&str, &str, MergeData)>)]) {
    fn pretoken(s: &str, freq: Freq) -> PreToken<u8, Idx> {
      let idxs = s.bytes().map(|b| b as Idx - 'a' as Idx).collect::<Vec<_>>();
      PreToken {
        src: s.to_word(),
        idxs,
        freq,
      }
    }
    fn lookup(bpe: &BpeTrainer<u8, Idx>, s: &str) -> Option<Idx> {
      bpe.vocab.iter().find_map(|(i, w)| {
        if w.as_ref() == s.as_bytes() {
          Some(*i)
        } else {
          None
        }
      })
    }
    fn display(bpe: &BpeTrainer<u8, Idx>, changes: &AHashMap<(Idx, Idx), MergeData>) -> String {
      let mut parts = Vec::new();
      let target = ("__target__").to_word();
      let changes = changes.iter().collect::<BTreeMap<_, _>>();
      for (tp, data) in changes {
        let left = bpe.vocab.get(&tp.0).unwrap_or(&target).debug_display();
        let right = bpe.vocab.get(&tp.1).unwrap_or(&target).debug_display();
        parts.push(format!("({:?}, {:?}, MergeData::new({}).occurs_in({:?}))", left, right, data.freq, data.occurs_in_vec()));
      }
      format!("{{\n  {}\n}}", parts.join(",\n  "))
    }

    let mut bpe = BpeTrainer::default();
    bpe.vocab.extend(
      ('a' ..= 'z').enumerate().map(|(i, c)| (i as Idx, c.to_string().to_word()))
    );
    bpe._set_vocab_idx(100);
    bpe.words.extend(
      pretokens.iter().map(|(s, f)| pretoken(s, *f))
    );
    bpe.init_training();
    for (m, expected) in merges {
      let merge_tp = (
        lookup(&bpe, m.0).unwrap(), lookup(&bpe, m.1).unwrap()
      );
      let merge = bpe.pre_merges.get(&merge_tp).unwrap().clone();
      let target = bpe._add_vocab_idx();
      let changes = bpe.merge(&merge, target);
      assert_eq!(merge.data.freq, -changes.get(&merge_tp).cloned().unwrap().freq);
      if expected.is_empty() {
        continue;
      }
      let expected = expected.into_iter().map(|(a, b, data)| {
        let tp_idx = (lookup(&bpe, a).unwrap_or(target), lookup(&bpe, b).unwrap_or(target));
        let mut data = data.clone();
        // Lazy occurrence sets keep positive affected-word additions exact, but
        // do not maintain exact removals or zero-net changes.
        if data.freq <= 0 {
          data.occurs_in.clear();
        }
        (tp_idx, data)
      }).collect::<AHashMap<_, _>>();
      assert_eq!(changes, expected, "\nExpected changes:\n{}\nActual changes:\n{}", display(&bpe, &expected), display(&bpe, &changes));
    }
  }

  fn pre_merge_snapshot<C, I>(
    bpe: &BpeTrainer<C, I>,
  ) -> BTreeMap<(I, I), (String, String, Option<I>, Freq, Vec<u64>)>
  where
    I: Copy + Ord,
    Word<C>: WordDebugExt,
  {
    bpe
      .pre_merges
      .iter()
      .map(|(tp, merge)| {
        (
          *tp,
          (
            merge.content.0.debug_display(),
            merge.content.1.debug_display(),
            merge.target,
            merge.data.freq,
            merge.data.occurs_in_vec(),
          ),
        )
      })
      .collect()
  }

  fn assert_unicode_trainers_equal(
    actual: &BpeTrainer<Character, CharIdx>, expected: &BpeTrainer<Character, CharIdx>,
  ) {
    let vocab = |trainer: &BpeTrainer<Character, CharIdx>| {
      trainer
        .vocab
        .iter()
        .map(|(idx, word)| (*idx, word.debug_display()))
        .collect::<Vec<_>>()
    };
    let merges = |trainer: &BpeTrainer<Character, CharIdx>| {
      trainer
        .merges
        .iter()
        .map(|merge| {
          (
            merge.tp,
            merge.target,
            merge.content.0.debug_display(),
            merge.content.1.debug_display(),
            merge.data.freq,
            merge.data.occurs_in_vec(),
          )
        })
        .collect::<Vec<_>>()
    };
    let words = |trainer: &BpeTrainer<Character, CharIdx>| {
      trainer
        .words
        .iter()
        .map(|word| (word.src.debug_display(), word.idxs.clone(), word.freq))
        .collect::<Vec<_>>()
    };

    assert_eq!(vocab(actual), vocab(expected));
    assert_eq!(merges(actual), merges(expected));
    assert_eq!(words(actual), words(expected));
    actual.validate_model().unwrap();
    expected.validate_model().unwrap();
  }

  #[test]
  fn test_bpe_merge() {
    _test_bpe_merge(&[("abcd", 5), ("abcdbcd", 30), ("abcbcdab", 200)], &[(("b", "c"), vec![
      ("a", "b", MergeData::new(-235).add_occurs_in([0, 1])),
      ("a", "bc", MergeData::new(235).add_occurs_in([0, 1, 2])),
      ("b", "c", MergeData::new(-465).add_occurs_in([0, 1, 2])),
      ("c", "b", MergeData::new(-200).add_occurs_in([2])),
      ("c", "d", MergeData::new(-265).add_occurs_in([0, 1, 2])),
      ("d", "b", MergeData::new(-30).add_occurs_in([1])),
      ("d", "bc", MergeData::new(30).add_occurs_in([1])),
      ("bc", "b", MergeData::new(0).add_occurs_in([2])),
      ("bc", "d", MergeData::new(265).add_occurs_in([0, 1, 2])),
      ("bc", "bc", MergeData::new(200).add_occurs_in([2])),
    ])]);

    _test_bpe_merge(&[("wherever", 10)],
    &[(("h", "e"), vec![
      ("e", "r", MergeData::new(-10).add_occurs_in([])),
      ("h", "e", MergeData::new(-10).add_occurs_in([0])),
      ("w", "h", MergeData::new(-10).add_occurs_in([0])),
      ("w", "he", MergeData::new(10).add_occurs_in([0])),
      ("he", "r", MergeData::new(10).add_occurs_in([0])),
    ])]);

    _test_bpe_merge(&[("aaa", 10), ("aaaa", 1)],
    &[(("a", "a"), vec![
      ("a", "a", MergeData::new(-23).add_occurs_in([0, 1])),
      ("aa", "a", MergeData::new(10).add_occurs_in([0])),
      ("aa", "aa", MergeData::new(1).add_occurs_in([1])),
    ])]);
  }

  #[test]
  fn test_bpe_step() {
    let mut bpe = BpeTrainer::<u8, Idx>::from_words(vec![
      ("ababc", 5),
      ("ababcbabc", 30),
      ("abcbabcab", 200),
    ], &vec![]);
    assert!(bpe.words.len() > 0);
    bpe.init_training();
    assert!(bpe.pre_merges.len() > 0);
    for _ in 0..3 {
      bpe.step().unwrap();
    }
    let result_vocab = bpe.vocab.into_iter().map(|(i, w)| (i, w.debug_display())).skip(256).collect::<Vec<_>>();
    assert_eq!(
      result_vocab,
      vec![
        (256, "ab".to_string()),
        (257, "abc".to_string()),
        (258, "babc".to_string()),
      ]
    );
    let result_merges = bpe.merges.into_iter().map(|m| {
      let left = m.content.0.debug_display();
      let right = m.content.1.debug_display();
      (left, right, m.data.freq)
    }).collect::<Vec<_>>();
    assert_eq!(
      result_merges,
      vec![
        ("a".to_string(), "b".to_string(), 700),
        ("ab".to_string(), "c".to_string(), 465),
        ("b".to_string(), "abc".to_string(), 230),
      ]
    );
  }

  #[test]
  fn test_bpe_step_parallel_merge_matches_sequential() {
    fn train(parallel_merge_min_occurs_in: Option<usize>) -> (Vec<(Idx, String)>, Vec<(String, String, Freq)>, Vec<Vec<Idx>>) {
      let mut bpe = BpeTrainer::<u8, Idx>::from_words_with_config(
        vec![
          ("ababc", 5),
          ("ababcbabc", 30),
          ("abcbabcab", 200),
        ],
        &vec![],
        BpeTrainerConfig {
          parallel_merge_min_occurs_in,
          ..BpeTrainerConfig::default()
        },
      );
      bpe.init_training();
      for _ in 0..3 {
        bpe.step().unwrap();
      }
      let vocab = bpe
        .vocab
        .iter()
        .map(|(i, w)| (*i, w.debug_display()))
        .skip(256)
        .collect::<Vec<_>>();
      let merges = bpe
        .merges
        .iter()
        .map(|m| (m.content.0.debug_display(), m.content.1.debug_display(), m.data.freq))
        .collect::<Vec<_>>();
      let words = bpe.words.iter().map(|w| w.idxs.clone()).collect::<Vec<_>>();
      (vocab, merges, words)
    }

    assert!(!crate::bpe::utils::should_parallel_merge(3, 3, None));
    assert!(crate::bpe::utils::should_parallel_merge(3, 3, Some(1)));
    assert_eq!(train(None), train(Some(1)));
  }

  #[test]
  fn test_parallel_initialization_matches_sequential() {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();

    let mut byte_sequential = BpeTrainer::<u8, Idx>::from_words(
      [("aaaa", 3), ("aa", 5), ("bbbbbb", 2)],
      &[],
    );
    let mut byte_parallel = BpeTrainer::<u8, Idx>::from_words(
      [("aaaa", 3), ("aa", 5), ("bbbbbb", 2)],
      &[],
    );
    byte_sequential._build_pre_merges_with_options(false, 4);
    pool.install(|| byte_parallel._build_pre_merges_with_options(true, 4));

    assert_eq!(pre_merge_snapshot(&byte_parallel), pre_merge_snapshot(&byte_sequential));
    let aa = (b'a' as Idx, b'a' as Idx);
    assert_eq!(byte_parallel.pre_merges.get(&aa).unwrap().data.freq, 14);
    assert_eq!(byte_parallel.pre_merges.get(&aa).unwrap().data.occurs_in_vec(), [0, 1]);

    let words = [("你你", 3), ("你", 5), ("你好你", 7)];
    let mut unicode_sequential = BpeTrainer::<Character, CharIdx>::from_words(words, &[]);
    let mut unicode_parallel = BpeTrainer::<Character, CharIdx>::from_words(words, &[]);
    unicode_sequential._build_pre_merges_with_options(false, 2);
    pool.install(|| unicode_parallel._build_pre_merges_with_options(true, 2));

    assert_eq!(pre_merge_snapshot(&unicode_parallel), pre_merge_snapshot(&unicode_sequential));
    let initial_ni = (CharIdx::Idx(u32::MAX), CharIdx::Char('你'));
    assert_eq!(unicode_parallel.pre_merges.get(&initial_ni).unwrap().data.freq, 25);

    unicode_sequential.step().unwrap();
    unicode_parallel.step().unwrap();
    unicode_sequential._build_pre_merges_with_options(false, 2);
    pool.install(|| unicode_parallel._build_pre_merges_with_options(true, 2));

    assert_eq!(pre_merge_snapshot(&unicode_parallel), pre_merge_snapshot(&unicode_sequential));
    assert!(!unicode_parallel.pre_merges.contains_key(&initial_ni));

    let signed_words = [("", 9), ("aa", 3), ("aa", 0), ("aa", -3), ("bb", 1)];
    let mut signed_sequential = BpeTrainer::<u8, Idx>::from_words(signed_words, &[]);
    let mut signed_parallel = BpeTrainer::<u8, Idx>::from_words(signed_words, &[]);
    signed_sequential._build_pre_merges_with_options(false, 1);
    pool.install(|| signed_parallel._build_pre_merges_with_options(true, 1));
    assert_eq!(pre_merge_snapshot(&signed_parallel), pre_merge_snapshot(&signed_sequential));
    assert_eq!(signed_parallel.pre_merges.get(&aa).unwrap().data.freq, 0);
    assert_eq!(signed_parallel.pre_merges.get(&aa).unwrap().data.occurs_in_vec(), [1, 2, 3]);
    signed_parallel.step().unwrap();
    assert_eq!(signed_parallel.merges.last().unwrap().content.0.debug_display(), "b");
    assert_eq!(signed_parallel.merges.last().unwrap().content.1.debug_display(), "b");
    assert!(signed_parallel.step().is_err());

    let signed_unicode_words = [("你", 3), ("你", 0), ("你", -3), ("ab", 1)];
    let mut signed_unicode_sequential = BpeTrainer::<Character, CharIdx>::from_words(
      signed_unicode_words,
      &[],
    );
    let mut signed_unicode_parallel = BpeTrainer::<Character, CharIdx>::from_words(
      signed_unicode_words,
      &[],
    );
    signed_unicode_sequential._build_pre_merges_with_options(false, 1);
    pool.install(|| signed_unicode_parallel._build_pre_merges_with_options(true, 1));
    assert_eq!(
      pre_merge_snapshot(&signed_unicode_parallel),
      pre_merge_snapshot(&signed_unicode_sequential),
    );
    let signed_initial_ni = (CharIdx::Idx(u32::MAX), CharIdx::Char('你'));
    assert_eq!(signed_unicode_parallel.pre_merges.get(&signed_initial_ni).unwrap().data.freq, 0);
    signed_unicode_parallel.step().unwrap();
    assert_eq!(signed_unicode_parallel.merges.last().unwrap().content.0.debug_display(), "a");
    assert_eq!(signed_unicode_parallel.merges.last().unwrap().content.1.debug_display(), "b");
    assert!(signed_unicode_parallel.step().is_err());
  }

  #[test]
  fn test_parallel_initialization_preserves_training_across_thread_counts() {
    let words = [
      ("你好你好", 3),
      ("您好", 3),
      ("世界", 2),
      ("你世", 2),
      ("界好", 2),
      ("abab", 1),
    ];

    for tie_break in [TieBreak::SmallestPairId, TieBreak::LargestContent] {
      let config = BpeTrainerConfig {
        tie_break,
        ..BpeTrainerConfig::default()
      };
      let mut sequential = BpeTrainer::<Character, CharIdx>::from_words_with_config(
        words,
        &[],
        config,
      );
      sequential._build_pre_merges_with_options(false, 2);
      while sequential.step().is_ok() {}

      for threads in [1, 2, 4] {
        let mut parallel = BpeTrainer::<Character, CharIdx>::from_words_with_config(
          words,
          &[],
          config,
        );
        rayon::ThreadPoolBuilder::new()
          .num_threads(threads)
          .build()
          .unwrap()
          .install(|| parallel._build_pre_merges_with_options(true, 2));
        while parallel.step().is_ok() {}

        assert_unicode_trainers_equal(&parallel, &sequential);
      }
    }
  }

  #[test]
  fn test_bpe_step_tie_breaks_by_smallest_pair_id() {
    let mut bpe = BpeTrainer::<u8, Idx>::from_words(vec![
      ("ab", 1),
      ("cd", 1),
    ], &vec![]);
    bpe.init_training();

    bpe.step().unwrap();

    let merge = bpe.merges.last().unwrap();
    assert_eq!(merge.content.0.debug_display(), "a");
    assert_eq!(merge.content.1.debug_display(), "b");
  }

  #[test]
  fn test_bpe_step_can_tie_break_by_largest_content() {
    let mut bpe = BpeTrainer::<u8, Idx>::new_with_config(
      vec![],
      vec![],
      BpeTrainerConfig {
        initial_alphabet: InitialAlphabet::RawBytes,
        tie_break: TieBreak::LargestContent,
        ..BpeTrainerConfig::default()
      },
    );
    bpe.add_words(&mut vec![
      ("ab", 1),
      ("cd", 1),
    ].into_iter());
    bpe.init_training();

    bpe.step().unwrap();

    let merge = bpe.merges.last().unwrap();
    assert_eq!(merge.content.0.debug_display(), "c");
    assert_eq!(merge.content.1.debug_display(), "d");
  }

  #[test]
  fn test_unicode_initial_units_precede_dependent_merges() {
    for (tie_break, expected_tail) in [
      (TieBreak::SmallestPairId, ["你", "好", "你好"]),
      (TieBreak::LargestContent, ["好", "你", "你好"]),
    ] {
      let mut bpe = BpeTrainer::<Character, CharIdx>::from_words_with_config(
        [("你好", 1)],
        &[],
        BpeTrainerConfig {
          tie_break,
          ..BpeTrainerConfig::default()
        },
      );

      for vocab_size in 257..=259 {
        // Repeated calls rebuild the candidate heap and must not materialize a
        // Unicode unit more than once.
        bpe.train_until(vocab_size).unwrap();
        let tail = bpe
          .vocab
          .iter()
          .filter_map(|(idx, token)| match idx {
            CharIdx::Idx(idx) if *idx >= 256 => Some(token.debug_display()),
            _ => None,
          })
          .collect::<Vec<_>>();
        assert_eq!(tail, expected_tail[..vocab_size - 256]);
        assert_eq!(bpe.merges.len(), vocab_size.saturating_sub(258));
        bpe.validate_model().unwrap();
      }

      let merge = bpe.merges.first().unwrap();
      assert_eq!(merge.content.0.debug_display(), "你");
      assert_eq!(merge.content.1.debug_display(), "好");
      assert_eq!(merge.merged_content().debug_display(), "你好");
      let model = bpe.validate_model().unwrap();
      let model_merge = model.merges().first().unwrap();
      assert!(matches!(model_merge.tp, (CharIdx::Idx(_), CharIdx::Idx(_))));
      assert!(model_merge.data.occurs_in.is_empty());
    }
  }

  #[test]
  fn test_unicode_initial_unit_frequency_aggregates_repeated_positions() {
    let mut bpe = BpeTrainer::<Character, CharIdx>::from_words(
      [("你你", 3), ("你", 5)],
      &[],
    );
    let initial_unit = (CharIdx::Idx(u32::MAX), CharIdx::Char('你'));
    let repeated_pair = (CharIdx::Char('你'), CharIdx::Char('你'));

    bpe.init_training();

    assert_eq!(bpe.pre_merges.get(&initial_unit).unwrap().data.freq, 11);
    assert_eq!(bpe.pre_merges.get(&repeated_pair).unwrap().data.freq, 3);

    bpe.step().unwrap();
    bpe.init_training();

    assert!(!bpe.pre_merges.contains_key(&initial_unit));
    assert_eq!(bpe.pre_merges.get(&repeated_pair).unwrap().data.freq, 3);
  }

  #[test]
  fn test_unicode_initial_unit_priority_does_not_depend_on_sentinel_content() {
    let input = format!("{}你", char::MAX);
    let mut bpe = BpeTrainer::<Character, CharIdx>::from_words_with_config(
      [(input, 1)],
      &[],
      BpeTrainerConfig {
        tie_break: TieBreak::LargestContent,
        ..BpeTrainerConfig::default()
      },
    );

    bpe.train_until(258).unwrap();

    assert!(bpe.merges.is_empty());
    bpe.validate_model().unwrap();
  }

  #[test]
  fn test_unicode_initial_unit_wins_unrelated_frequency_tie() {
    let mut bpe = BpeTrainer::<Character, CharIdx>::from_words(
      [("ab", 1), ("你", 1)],
      &[],
    );

    bpe.train_until(257).unwrap();

    assert_eq!(bpe.vocab.get(&CharIdx::Idx(256)).unwrap().debug_display(), "你");
    assert!(bpe.merges.is_empty());
    bpe.validate_model().unwrap();
  }

  #[test]
  fn test_validation_rejects_merge_before_its_dependency() {
    let mut bpe = BpeTrainer::<u8, Idx>::new(vec![], vec![]);
    let a: Word<u8> = "a".to_word();
    let b: Word<u8> = "b".to_word();
    let c: Word<u8> = "c".to_word();
    let ab: Word<u8> = "ab".to_word();
    let abc: Word<u8> = "abc".to_word();
    bpe.vocab.insert(256, ab.clone());
    bpe.vocab.insert(257, abc);
    bpe.merges = vec![
      Merge::new((256, b'c' as Idx), (ab.clone(), c)).with_target(257),
      Merge::new((b'a' as Idx, b'b' as Idx), (a, b)).with_target(256),
    ];

    let error = bpe.validate_model().unwrap_err();

    assert!(matches!(error, MyError::InvalidBpeModel(_)));
    assert!(error.to_string().contains("merge 0 left operand ab"));
  }

  #[test]
  fn test_validation_rejects_operand_id_content_mismatch() {
    let mut bpe = BpeTrainer::<u8, Idx>::new(vec![], vec![]);
    let a: Word<u8> = "a".to_word();
    let b: Word<u8> = "b".to_word();
    let ab: Word<u8> = "ab".to_word();
    bpe.vocab.insert(256, ab);
    bpe.merges = vec![
      Merge::new((b'a' as Idx, b'c' as Idx), (a, b)).with_target(256),
    ];

    let error = bpe.validate_model().unwrap_err();

    assert!(matches!(error, MyError::InvalidBpeModel(_)));
    assert!(error.to_string().contains("merge 0 right operand id resolves to c, expected b"));
  }

  #[test]
  fn test_validation_rejects_tokens_that_normalize_to_same_unit_content() {
    let mut bpe = BpeTrainer::<Character, CharIdx>::new(vec![], vec![]);
    let unicode: Word<Character> = "é".to_word();
    let bytes = vec![Character::Byte(0xc3), Character::Byte(0xa9)].to_word();
    bpe.vocab.insert(CharIdx::Idx(256), unicode);
    bpe.vocab.insert(CharIdx::Idx(257), bytes);

    let error = bpe.validate_model().unwrap_err();

    assert!(matches!(error, MyError::InvalidBpeModel(_)));
    assert!(error.to_string().contains("duplicate vocabulary token"));
  }

  #[test]
  fn test_bpe_from_words() {
    const NAME: &str = "tinystories_sample_5M";
    // const NAME: &str = "TinyStoriesV2-GPT4-train";
    let input = std::fs::read_to_string(format!("fixtures/_words.{NAME}.json")).unwrap();
    let words: BTreeMap<String, Freq> = serde_json::from_str(&input).unwrap();
    let mut bpe = BpeTrainer::from_words(words, &vec![DEFAULT_EOT.to_string()]);
    bpe.init_training();
    let vocab_size = match NAME {
      "tinystories_sample_5M" => 2000,
      _ => 10000,
    };
    while bpe.vocab.len() < vocab_size {
      bpe.step().unwrap();
      // let m = &bpe.merges.last().unwrap();
      // println!("{} {} => {}", _printable(&m.content.0), _printable(&m.content.1), m.data.freq);
    }
    std::fs::create_dir_all(format!("out/models/{NAME}")).ok();
    let model = bpe.validate_model().unwrap();
    model.save_vocab_json(&Gpt2Spec, std::fs::File::create(format!("out/models/{NAME}/vocab.json")).unwrap()).unwrap();
    model.save_merges_txt(&Gpt2Spec, std::fs::File::create(format!("out/models/{NAME}/merges.txt")).unwrap()).unwrap();

    let merges_txt = std::fs::read_to_string(format!("out/models/{NAME}/merges.txt")).unwrap();
    let merges_expect_txt = std::fs::read_to_string(format!("fixtures/merges.{NAME}.txt")).unwrap();
    assert_eq!(merges_txt, merges_expect_txt);
  }

  #[test]
  fn test_bpe_from_words_uni() {
    // const NAME: &str = "tinystories_sample_5M";
    // const NAME: &str = "TinyStoriesV2-GPT4-train";
    const NAME: &str = "TinyStories_all_data_zh_1M-sample";
    let spec = crate::spec::unitoken::UnitokenSpec;
    let input = std::fs::read_to_string(format!("fixtures/_words.{NAME}.json")).unwrap();
    let words: BTreeMap<String, Freq> = serde_json::from_str(&input).unwrap();
    let mut bpe = BpeTrainer::<Character, CharIdx>::from_words_with_config(
      words,
      &vec![DEFAULT_EOT.to_string()],
      BpeTrainerConfig {
        initial_alphabet: InitialAlphabet::RawBytes,
        tie_break: TieBreak::LargestContent,
        ..BpeTrainerConfig::default()
      },
    );
    bpe.init_training();
    let vocab_size = match NAME {
      "tinystories_sample_5M" | "TinyStories_all_data_zh_1M-sample" => 2000,
      _ => 10000,
    };
    while bpe.vocab.len() < vocab_size {
      bpe.step().unwrap();
      // let m = &bpe.merges.last().unwrap();
      // println!("{} {} => {}", _printable(&m.content.0), _printable(&m.content.1), m.data.freq);
    }
    std::fs::create_dir_all(format!("out/models/{NAME}")).ok();
    let model = bpe.validate_model().unwrap();
    model.save_vocab_json(&spec, std::fs::File::create(format!("out/models/{NAME}/vocab.uni.json")).unwrap()).unwrap();
    model.save_merges_txt(&spec, std::fs::File::create(format!("out/models/{NAME}/merges.uni.txt")).unwrap()).unwrap();

    let merges_txt = std::fs::read_to_string(format!("out/models/{NAME}/merges.uni.txt")).unwrap();
    let merges_expect_txt = std::fs::read_to_string(format!("fixtures/merges.{NAME}.uni.txt")).unwrap();
    let merges = merges_txt.trim_end().lines().collect::<Vec<_>>();
    assert_eq!(merges, merges_expect_txt.lines().take(merges.len()).collect::<Vec<_>>());
  }
}
