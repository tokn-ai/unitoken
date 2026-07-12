use std::{cmp::{Ordering, Reverse}, collections::{BinaryHeap, BTreeMap, BTreeSet, HashMap, hash_map::Entry}, hash::Hash, ops::Range, sync::atomic::AtomicU64};

use ahash::{AHashMap, AHashSet};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{MyError, MyResult, traits::{CanStrToWord, CanToWord, CanTrain, Train}};

use super::*;

#[cfg(any(test, feature = "analysis"))]
#[doc(hidden)]
pub mod analysis;

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
  /// Keep occurrence postings for only a resident top-K pair window.
  ///
  /// `None` preserves the full exact occurrence map. A positive value enables
  /// K-to-2K hysteresis with cold-winner hydration scans.
  pub hot_pair_window_size: Option<usize>,
}

impl BpeTrainerConfig {
  pub fn hf_byte_level() -> Self {
    Self {
      initial_alphabet: InitialAlphabet::ByteLevel,
      tie_break: TieBreak::SmallestPairId,
      parallel_merge_min_occurs_in: None,
      hot_pair_window_size: None,
    }
  }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HotPairWindowStats {
  pub hydration_scans: u64,
  pub hydrated_word_entries: u64,
  pub batch_prunes: u64,
  pub prune_evictions: u64,
  pub peak_resident_pairs: usize,
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
  hot_occurrences: AHashMap<(I, I), AHashSet<u64>>,
  hot_admission_threshold: Option<Freq>,
  hot_window_stats: HotPairWindowStats,
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
struct PairPartial {
  freq: Freq,
  occurs_in: Vec<u64>,
}

// Map-side state for one whole-word range. Both scheduling modes use this
// representation so frequency and occurrence semantics have one implementation.
struct PreMergePartial<I> {
  initial_freqs: AHashMap<I, Freq>,
  pairs: AHashMap<(I, I), PairPartial>,
}

struct PreMergeRange {
  words: Range<usize>,
  units: usize,
}

impl<I> Default for PreMergePartial<I> {
  fn default() -> Self {
    Self {
      initial_freqs: AHashMap::new(),
      pairs: AHashMap::new(),
    }
  }
}

#[inline]
fn collect_pre_merge_partial<C, I>(
  words: &[PreToken<C, I>], word_offset: usize, vocab: &BTreeMap<I, Word<C>>,
  collect_occurrences: bool,
) -> PreMergePartial<I>
where
  I: Copy + Eq + Hash + Ord,
{
  let mut partial = PreMergePartial::default();
  for (local_word_idx, word) in words.iter().enumerate() {
    let word_idx = (word_offset + local_word_idx) as u64;
    for unit in word.idxs.iter().copied() {
      if !vocab.contains_key(&unit) {
        *partial.initial_freqs.entry(unit).or_default() += word.freq;
      }
    }
    for tp in word.idxs.iter().copied().zip(word.idxs.iter().skip(1).copied()) {
      let entry = partial.pairs.entry(tp).or_default();
      entry.freq += word.freq;
      // A pair records each word once, even when it occurs repeatedly in that word.
      if collect_occurrences && entry.occurs_in.last().copied() != Some(word_idx) {
        entry.occurs_in.push(word_idx);
      }
    }
  }
  partial
}

#[inline]
fn collect_pre_merge_range<C, I>(
  words: &[PreToken<C, I>], range: &PreMergeRange, vocab: &BTreeMap<I, Word<C>>,
  collect_occurrences: bool,
) -> PreMergePartial<I>
where
  I: Copy + Eq + Hash + Ord,
{
  collect_pre_merge_partial(
    &words[range.words.clone()],
    range.words.start,
    vocab,
    collect_occurrences,
  )
}

// Lazily partition the inventory so multi-terabyte inputs do not require a
// corpus-sized range table before initialization can begin.
struct PreMergeRanges<'a, C, I> {
  words: &'a [PreToken<C, I>],
  max_units: usize,
  next_word: usize,
}

impl<'a, C, I> PreMergeRanges<'a, C, I> {
  fn new(words: &'a [PreToken<C, I>], max_units: usize) -> Self {
    Self {
      words,
      max_units: max_units.max(1),
      next_word: 0,
    }
  }
}

impl<C, I> Iterator for PreMergeRanges<'_, C, I> {
  type Item = PreMergeRange;

  fn next(&mut self) -> Option<Self::Item> {
    if self.next_word >= self.words.len() {
      return None;
    }

    let start = self.next_word;
    let mut end = start;
    let mut units = 0usize;
    for word in &self.words[start..] {
      let word_units = word.idxs.len().max(1);
      if end > start && units.saturating_add(word_units) > self.max_units {
        break;
      }
      units = units.saturating_add(word_units);
      end += 1;
    }
    self.next_word = end;

    Some(PreMergeRange {
      words: start..end,
      units,
    })
  }
}

// Serially materializes ordered partials into the trainer's canonical merge
// state. Keeping Unicode resolution here avoids duplicating it in each worker.
struct PreMergeReducer<'a, C, I> {
  vocab: &'a BTreeMap<I, Word<C>>,
  pre_merges: &'a mut HashMap<(I, I), Merge<C, I>>,
  vocab_contents: Option<BTreeSet<Word<C>>>,
  materialized_initial_units: AHashSet<I>,
  initial_sentinel: I,
  empty_word: Word<C>,
}

fn resolve_vocab_word<C, I>(vocab: &BTreeMap<I, Word<C>>, unit: I) -> Word<C>
where
  C: CanStrToWord,
  I: IdxLike + HasChar<C>,
{
  vocab
    .get(&unit)
    .cloned()
    .or_else(|| unit.idx_to_word())
    .ok_or_else(|| MyError::OovIdx(unit.to_u64()))
    .unwrap()
}

impl<'a, C, I> PreMergeReducer<'a, C, I>
where
  C: CanStrToWord + Ord,
  I: IdxLike + HasChar<C>,
{
  fn new(
    vocab: &'a BTreeMap<I, Word<C>>, pre_merges: &'a mut HashMap<(I, I), Merge<C, I>>,
  ) -> Self {
    Self {
      vocab,
      pre_merges,
      vocab_contents: None,
      materialized_initial_units: AHashSet::new(),
      initial_sentinel: I::from_u64(u64::MAX),
      empty_word: Vec::<C>::new().to_word(),
    }
  }

  #[inline]
  fn apply_initial(&mut self, unit: I, freq: Freq) {
    if self.materialized_initial_units.contains(&unit) {
      return;
    }
    let tp = (self.initial_sentinel, unit);
    if let Some(merge) = self.pre_merges.get_mut(&tp) {
      merge.data.freq += freq;
      return;
    }

    let content = resolve_vocab_word(self.vocab, unit);
    // Unicode units remain CharIdx::Char in the word inventory after they
    // receive a numeric vocab id, so compare content when training resumes.
    if self
      .vocab_contents
      .get_or_insert_with(|| self.vocab.values().cloned().collect())
      .contains(&content)
    {
      self.materialized_initial_units.insert(unit);
      return;
    }

    let mut merge = Merge::new(tp, (self.empty_word.clone(), content)).with_target(unit);
    merge.data.freq = freq;
    self.pre_merges.insert(tp, merge);
  }

  #[inline]
  fn apply_pair(&mut self, tp: (I, I), data: PairPartial) {
    let vocab = self.vocab;
    match self.pre_merges.entry(tp) {
      Entry::Occupied(mut entry) => {
        let merge = entry.get_mut();
        merge.data.freq += data.freq;
        merge.data.occurs_in.extend(data.occurs_in);
      }
      Entry::Vacant(entry) => {
        let mut merge = Merge::new(
          tp,
          (resolve_vocab_word(vocab, tp.0), resolve_vocab_word(vocab, tp.1)),
        );
        merge.data.freq = data.freq;
        merge.data.occurs_in.extend(data.occurs_in);
        entry.insert(merge);
      }
    }
  }

  #[inline]
  fn apply(&mut self, partial: PreMergePartial<I>) {
    for (unit, freq) in partial.initial_freqs {
      self.apply_initial(unit, freq);
    }
    for (tp, data) in partial.pairs {
      self.apply_pair(tp, data);
    }
  }
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
    let sp_set = special_tokens.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let mut bpe = Self::new_with_config(Vec::new(), special_tokens.to_vec(), config);
    let vocab_start_idx = bpe._byte_vocab_start_idx.unwrap();
    let tokens = Self::_words_to_tokens(
      words,
      vocab_start_idx,
      &sp_set,
      Some(&bpe.byte_vocab),
    );
    bpe.words = tokens;
    bpe
  }

  /// Create a trainer from already pre-tokenized words.
  ///
  /// This initializes vocab with `special_tokens` and a 256-entry byte vocabulary.
  pub fn new(words: Vec<PreToken<C, I>>, special_tokens: Vec<String>) -> Self {
    Self::new_with_config(words, special_tokens, BpeTrainerConfig::default())
  }

  pub fn new_with_config(words: Vec<PreToken<C, I>>, special_tokens: Vec<String>, config: BpeTrainerConfig) -> Self {
    assert!(
      config.hot_pair_window_size != Some(0),
      "hot_pair_window_size must be positive",
    );
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
  /// Frequency of the most recently completed pair merge.
  ///
  /// Materializing an initial Unicode unit is not a pair merge and does not
  /// change this value.
  pub fn last_merge_freq(&self) -> Option<Freq> {
    self.merges.last().map(|merge| merge.data.freq)
  }

  pub fn hot_pair_window_stats(&self) -> Option<&HotPairWindowStats> {
    self.config.hot_pair_window_size.map(|_| &self.hot_window_stats)
  }

  pub fn hot_resident_pairs(&self) -> usize {
    self.hot_occurrences.len()
  }

  pub fn hot_occurrence_capacity(&self) -> usize {
    self
      .hot_occurrences
      .values()
      .map(|occurrences| occurrences.capacity())
      .sum()
  }

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
      hot_occurrences: AHashMap::new(),
      hot_admission_threshold: None,
      hot_window_stats: HotPairWindowStats::default(),
      special_tokens: Vec::new(),
      words: Vec::new(),
    }
  }
}

impl<C, I: IdxLike> BpeTrainer<C, I>
where
  Word<C>: WordDebugExt,
  I: HasChar<C>,
  C: CanStrToWord + Clone + Ord + Send + Sync,
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
    self.hot_occurrences.clear();
    self.hot_admission_threshold = None;
    self.hot_window_stats = HotPairWindowStats::default();
    let collect_occurrences = self.config.hot_pair_window_size.is_none();
    self._build_pre_merges_batched(parallel, chunk_units, collect_occurrences);
    self.rebuild_merge_heap();
    if self.config.hot_pair_window_size.is_some() {
      self.refill_hot_occurrences(None);
    }
  }

  fn _build_pre_merges_batched(
    &mut self, parallel: bool, chunk_units: usize, collect_occurrences: bool,
  ) {
    let chunk_units = chunk_units.max(1);
    let words = &self.words;
    let vocab = &self.vocab;
    let mut reducer = PreMergeReducer::new(vocab, &mut self.pre_merges);

    if !parallel {
      for range in PreMergeRanges::new(words, chunk_units) {
        reducer.apply(collect_pre_merge_range(
          words,
          &range,
          vocab,
          collect_occurrences,
        ));
      }
      return;
    }

    let ranges_per_wave = rayon::current_num_threads().clamp(1, PARALLEL_INIT_MAX_BATCHES);
    let mut ranges = PreMergeRanges::new(words, chunk_units).peekable();
    while let Some(first_range) = ranges.next() {
      let oversized = first_range.units > chunk_units;
      let mut range_wave = Vec::with_capacity(ranges_per_wave);
      range_wave.push(first_range);
      if !oversized {
        while range_wave.len() < ranges_per_wave {
          let Some(next_range) = ranges.peek() else {
            break;
          };
          // Keep an oversized word intact so boundary pairs and occurrence dedup
          // remain local to one partial.
          if next_range.units > chunk_units {
            break;
          }
          range_wave.push(ranges.next().unwrap());
        }
      }

      let partials = range_wave
        .par_iter()
        .map(|range| collect_pre_merge_range(
          words,
          range,
          vocab,
          collect_occurrences,
        ))
        .collect::<Vec<_>>();
      for partial in partials {
        reducer.apply(partial);
      }
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

  fn ranked_top_pairs(&self) -> Vec<MergeCandidate<C, I>> {
    let limit = self
      .config
      .hot_pair_window_size
      .expect("ranked_top_pairs requires a hot window");
    let mut top = BinaryHeap::<Reverse<MergeCandidate<C, I>>>::with_capacity(
      limit.saturating_add(1),
    );
    for merge in self
      .pre_merges
      .values()
      .filter(|merge| merge.target.is_none() && merge.data.freq > 0)
    {
      top.push(Reverse(MergeCandidate::from_merge(
        merge,
        self.config.tie_break,
      )));
      if top.len() > limit {
        top.pop();
      }
    }
    let mut ranked = top
      .into_iter()
      .map(|Reverse(candidate)| candidate)
      .collect::<Vec<_>>();
    ranked.sort_unstable_by(|left, right| {
      right.cmp(left).then_with(|| left.tp.cmp(&right.tp))
    });
    ranked
  }

  fn refill_hot_occurrences(&mut self, required: Option<(I, I)>) {
    let ranked = self.ranked_top_pairs();
    let desired = ranked
      .iter()
      .map(|candidate| candidate.tp)
      .collect::<AHashSet<_>>();
    if let Some(required) = required {
      assert!(
        desired.contains(&required),
        "a cold pair winner must be present in the exact top-K refill",
      );
    }
    let mut hydrated = desired
      .iter()
      .copied()
      .filter(|tp| !self.hot_occurrences.contains_key(tp))
      .map(|tp| (tp, AHashSet::new()))
      .collect::<AHashMap<_, _>>();
    for (word_idx, word) in self.words.iter().enumerate() {
      for tp in word.idxs.iter().copied().zip(word.idxs.iter().skip(1).copied()) {
        if let Some(occurrences) = hydrated.get_mut(&tp) {
          occurrences.insert(word_idx as u64);
        }
      }
    }
    self.hot_occurrences.extend(hydrated);
    self.hot_admission_threshold = ranked.last().map(|candidate| candidate.freq);
    self.hot_window_stats.hydration_scans += 1;
    self.hot_window_stats.hydrated_word_entries = self
      .hot_window_stats
      .hydrated_word_entries
      .saturating_add(self.words.len() as u64);
    self.observe_hot_window_peak();
  }

  fn update_hot_occurrences(
    &mut self, target_idx: I, changes: &AHashMap<(I, I), MergeData>,
  ) {
    for (tp, data) in changes {
      let Some(occurrences) = self.hot_occurrences.get_mut(tp) else {
        continue;
      };
      if data.freq > 0 {
        occurrences.extend(data.occurs_in.iter().copied());
      }
    }

    let threshold = self
      .hot_admission_threshold
      .expect("a pair merge follows an initialized hot window");
    for (tp, data) in changes {
      if data.freq <= 0
        || (tp.0 != target_idx && tp.1 != target_idx)
        || data.freq < threshold
      {
        continue;
      }
      debug_assert!(!self.hot_occurrences.contains_key(tp));
      self.hot_occurrences.insert(*tp, data.occurs_in.clone());
    }
    self.observe_hot_window_peak();
  }

  fn prune_hot_occurrences(&mut self) {
    let window_size = self
      .config
      .hot_pair_window_size
      .expect("prune_hot_occurrences requires a hot window");
    if self.hot_occurrences.len() <= window_size.saturating_mul(2) {
      return;
    }

    let mut ranked = self
      .hot_occurrences
      .keys()
      .filter_map(|tp| {
        let merge = self.pre_merges.get(tp)?;
        (merge.target.is_none() && merge.data.freq > 0)
          .then(|| MergeCandidate::from_merge(merge, self.config.tie_break))
      })
      .collect::<Vec<_>>();
    ranked.sort_unstable_by(|left, right| {
      right.cmp(left).then_with(|| left.tp.cmp(&right.tp))
    });
    ranked.truncate(window_size);
    self.hot_admission_threshold = ranked.last().map(|candidate| candidate.freq);
    let retained = ranked
      .into_iter()
      .map(|candidate| candidate.tp)
      .collect::<AHashSet<_>>();
    let before = self.hot_occurrences.len();
    self.hot_occurrences.retain(|tp, _| retained.contains(tp));
    self.hot_window_stats.batch_prunes += 1;
    self.hot_window_stats.prune_evictions = self
      .hot_window_stats
      .prune_evictions
      .saturating_add(before.saturating_sub(self.hot_occurrences.len()) as u64);
  }

  fn observe_hot_window_peak(&mut self) {
    self.hot_window_stats.peak_resident_pairs = self
      .hot_window_stats
      .peak_resident_pairs
      .max(self.hot_occurrences.len());
  }

  fn update_pre_merges(
    &mut self, merge: &Merge<C, I>, target_idx: I,
    changes: AHashMap<(I, I), MergeData>,
  ) {
    let bounded = self.config.hot_pair_window_size.is_some();
    if bounded {
      self.update_hot_occurrences(target_idx, &changes);
    }
    let changed_tps = _update_merge_map(
      &mut self.pre_merges,
      merge,
      changes,
      Some(&self.vocab),
      !bounded,
    );
    for tp in changed_tps {
      self.push_merge_candidate(tp);
    }
    if bounded {
      self.prune_hot_occurrences();
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
      let is_pair = merge.target.is_none();
      if self.config.hot_pair_window_size.is_some()
        && is_pair
        && !self.hot_occurrences.contains_key(&candidate.tp)
      {
        self.refill_hot_occurrences(Some(candidate.tp));
      }

      let mut merge = self.pre_merges.remove(&candidate.tp).unwrap();
      if self.config.hot_pair_window_size.is_some() && is_pair {
        merge.data.occurs_in = self
          .hot_occurrences
          .remove(&candidate.tp)
          .expect("a selected pair must have resident occurrences");
      }
      return Some(merge);
    }
    None
  }

  /// Apply one merge operation and return the newly assigned vocab index.
  ///
  /// This is the core training step once a merge candidate has been selected.
  #[hotpath::measure]
  pub fn _step(&mut self, merge: Merge<C, I>) -> I where C: Clone {
    self._step_with_observer(merge, |_, _, _| {})
  }

  fn _step_with_observer<F>(&mut self, merge: Merge<C, I>, observe_changes: F) -> I
  where
    C: Clone,
    F: FnOnce(&Self, I, Option<&AHashMap<(I, I), MergeData>>),
  {
    let target_idx = self._add_vocab_idx();
    // if target = Some(j), this is a single char token, no need to merge.
    // but we have to add it to vocab.
    if merge.target.is_some() {
      self.vocab.insert(target_idx, merge.content.1.clone());
      observe_changes(self, target_idx, None);
      return target_idx;
    }
    let changes = self.merge(&merge, target_idx);
    // println!("Merge {:?} (freq={}) into idx {}", merge.tp, merge.data.freq, target_idx);
    let mut merge = merge.with_target(target_idx);
    let merged_word = merge.merged_content();
    // self.vocab.entry(merge.tp.0).or_insert_with(|| merge.content.0.clone());
    // self.vocab.entry(merge.tp.1).or_insert_with(|| merge.content.1.clone());
    self.vocab.insert(target_idx, merged_word);
    assert_eq!(-changes.get(&merge.tp).map(|i| i.freq).unwrap_or(0), merge.data.freq);
    metrics::histogram!("bpe_trainer.changes").record(changes.len() as f64);
    observe_changes(self, target_idx, Some(&changes));
    self.update_pre_merges(&merge, target_idx, changes);
    metrics::histogram!("bpe_trainer.occurs_in").record(merge.data.occurs_in.len() as f64);
    metrics::histogram!("bpe_trainer.freq").record(merge.data.freq as f64);
    if self.config.hot_pair_window_size.is_some() {
      merge.data.occurs_in = AHashSet::new();
    }
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

  fn build_pre_merges_naive<C, I>(bpe: &mut BpeTrainer<C, I>)
  where
    C: CanStrToWord + Clone + Ord + Send + Sync,
    I: IdxLike + HasChar<C>,
    Word<C>: WordDebugExt,
  {
    bpe.pre_merges.clear();
    bpe.merge_heap.clear();
    {
      let words = &bpe.words;
      let vocab = &bpe.vocab;
      let pre_merges = &mut bpe.pre_merges;
      let mut vocab_contents = None;
      let mut materialized_initial_units = AHashSet::new();
      let resolve = |unit: I| {
        vocab
          .get(&unit)
          .cloned()
          .or_else(|| unit.idx_to_word())
          .ok_or_else(|| MyError::OovIdx(unit.to_u64()))
          .unwrap()
      };
      let initial_sentinel = I::from_u64(u64::MAX);
      let empty_word = Vec::<C>::new().to_word();

      for (word_idx, word) in words.iter().enumerate() {
        for unit in word.idxs.iter().copied() {
          if vocab.contains_key(&unit) || materialized_initial_units.contains(&unit) {
            continue;
          }
          let tp = (initial_sentinel, unit);
          if let Some(merge) = pre_merges.get_mut(&tp) {
            merge.data.freq += word.freq;
            continue;
          }

          let content = resolve(unit);
          if vocab_contents
            .get_or_insert_with(|| vocab.values().cloned().collect::<BTreeSet<_>>())
            .contains(&content)
          {
            materialized_initial_units.insert(unit);
            continue;
          }
          let mut merge = Merge::new(tp, (empty_word.clone(), content)).with_target(unit);
          merge.data.freq = word.freq;
          pre_merges.insert(tp, merge);
        }

        for tp in word.idxs.iter().copied().zip(word.idxs.iter().skip(1).copied()) {
          let merge = pre_merges
            .entry(tp)
            .or_insert_with(|| Merge::new(tp, (resolve(tp.0), resolve(tp.1))));
          merge.add(word_idx as u64, word.freq);
        }
      }
    }
    bpe.rebuild_merge_heap();
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
  fn test_hot_pair_window_matches_exact_training() {
    fn train_byte(
      tie_break: TieBreak, hot_pair_window_size: Option<usize>,
    ) -> (Vec<(Idx, String)>, Vec<(String, String, Freq)>, Vec<Vec<Idx>>) {
      let mut trainer = BpeTrainer::<u8, Idx>::from_words_with_config(
        [
          ("cab", 11),
          ("eab", 9),
          ("gab", 7),
          ("abi", 5),
          ("abj", 3),
          ("abk", 1),
        ],
        &[],
        BpeTrainerConfig {
          tie_break,
          hot_pair_window_size,
          ..BpeTrainerConfig::default()
        },
      );
      trainer.init_training();
      while trainer.step().is_ok() {}
      (
        trainer
          .vocab
          .iter()
          .map(|(idx, word)| (*idx, word.debug_display()))
          .collect(),
        trainer
          .merges
          .iter()
          .map(|merge| (
            merge.content.0.debug_display(),
            merge.content.1.debug_display(),
            merge.data.freq,
          ))
          .collect(),
        trainer.words.iter().map(|word| word.idxs.clone()).collect(),
      )
    }

    fn train_unicode(
      tie_break: TieBreak, hot_pair_window_size: Option<usize>,
    ) -> (Vec<(CharIdx, String)>, Vec<(String, String, Freq)>, Vec<Vec<CharIdx>>) {
      let mut trainer = BpeTrainer::<Character, CharIdx>::from_words_with_config(
        [("你好你好", 7), ("您好", 5), ("世界", 3), ("你世", 2)],
        &[],
        BpeTrainerConfig {
          tie_break,
          hot_pair_window_size,
          ..BpeTrainerConfig::default()
        },
      );
      trainer.init_training();
      while trainer.step().is_ok() {}
      (
        trainer
          .vocab
          .iter()
          .map(|(idx, word)| (*idx, word.debug_display()))
          .collect(),
        trainer
          .merges
          .iter()
          .map(|merge| (
            merge.content.0.debug_display(),
            merge.content.1.debug_display(),
            merge.data.freq,
          ))
          .collect(),
        trainer.words.iter().map(|word| word.idxs.clone()).collect(),
      )
    }

    for tie_break in [TieBreak::SmallestPairId, TieBreak::LargestContent] {
      assert_eq!(train_byte(tie_break, Some(2)), train_byte(tie_break, None));
      assert_eq!(
        train_unicode(tie_break, Some(2)),
        train_unicode(tie_break, None),
      );
    }
  }

  #[test]
  fn test_hot_pair_window_bounds_and_releases_occurrences() {
    let mut trainer = BpeTrainer::<u8, Idx>::from_words_with_config(
      [
        ("cab", 11),
        ("eab", 9),
        ("gab", 7),
        ("abi", 5),
        ("abj", 3),
        ("abk", 1),
      ],
      &[],
      BpeTrainerConfig {
        hot_pair_window_size: Some(1),
        ..BpeTrainerConfig::default()
      },
    );

    trainer.init_training();
    assert!(trainer.pre_merges.values().all(|merge| merge.data.occurs_in.is_empty()));
    assert_eq!(trainer.hot_resident_pairs(), 1);
    assert_eq!(trainer.hot_pair_window_stats().unwrap().hydration_scans, 1);

    while trainer.step().is_ok() {
      assert!(trainer.merges.iter().all(|merge| merge.data.occurs_in.is_empty()));
    }

    let stats = trainer.hot_pair_window_stats().unwrap();
    assert!(stats.hydration_scans > 1, "fixture must exercise a cold winner refill");
    assert_eq!(stats.peak_resident_pairs, 1);
    trainer.validate_model().unwrap();
  }

  #[test]
  fn test_hot_pair_window_batch_prunes_to_exact_limit() {
    let mut trainer = BpeTrainer::<u8, Idx>::from_words_with_config(
      [("abcdef", 1)],
      &[],
      BpeTrainerConfig {
        hot_pair_window_size: Some(1),
        ..BpeTrainerConfig::default()
      },
    );
    trainer.init_training();
    trainer.hot_occurrences = trainer
      .pre_merges
      .keys()
      .copied()
      .map(|tp| (tp, AHashSet::new()))
      .collect();

    trainer.prune_hot_occurrences();

    assert_eq!(trainer.hot_resident_pairs(), 1);
    let stats = trainer.hot_pair_window_stats().unwrap();
    assert_eq!(stats.batch_prunes, 1);
    assert_eq!(stats.prune_evictions, 4);
  }

  #[test]
  fn test_pre_merge_ranges_are_bounded_without_splitting_words() {
    let bpe = BpeTrainer::<u8, Idx>::from_words(
      [("", 1), ("a", 1), ("bbb", 1), ("ccccc", 1), ("dd", 1)],
      &[],
    );
    let ranges = PreMergeRanges::new(&bpe.words, 4)
      .map(|range| (range.words, range.units))
      .collect::<Vec<_>>();

    assert_eq!(ranges, [(0..2, 2), (2..3, 3), (3..4, 5), (4..5, 2)]);
  }

  #[test]
  fn test_batched_initialization_matches_naive_reference() {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();

    let mut byte_reference = BpeTrainer::<u8, Idx>::from_words(
      [("aaaa", 3), ("aa", 5), ("bbbbbb", 2)],
      &[],
    );
    let mut byte_sequential = BpeTrainer::<u8, Idx>::from_words(
      [("aaaa", 3), ("aa", 5), ("bbbbbb", 2)],
      &[],
    );
    let mut byte_parallel = BpeTrainer::<u8, Idx>::from_words(
      [("aaaa", 3), ("aa", 5), ("bbbbbb", 2)],
      &[],
    );
    build_pre_merges_naive(&mut byte_reference);
    byte_sequential._build_pre_merges_with_options(false, 4);
    pool.install(|| byte_parallel._build_pre_merges_with_options(true, 4));

    let byte_expected = pre_merge_snapshot(&byte_reference);
    assert_eq!(pre_merge_snapshot(&byte_sequential), byte_expected);
    assert_eq!(pre_merge_snapshot(&byte_parallel), byte_expected);
    assert_eq!(byte_parallel.pre_merges.len(), 2);
    let aa = (b'a' as Idx, b'a' as Idx);
    assert_eq!(byte_parallel.pre_merges.get(&aa).unwrap().data.freq, 14);
    assert_eq!(byte_parallel.pre_merges.get(&aa).unwrap().data.occurs_in_vec(), [0, 1]);
    let bb = (b'b' as Idx, b'b' as Idx);
    assert_eq!(byte_parallel.pre_merges.get(&bb).unwrap().data.freq, 10);
    assert_eq!(byte_parallel.pre_merges.get(&bb).unwrap().data.occurs_in_vec(), [2]);

    let words = [("你你", 3), ("你", 5), ("你好你", 7)];
    let mut unicode_reference = BpeTrainer::<Character, CharIdx>::from_words(words, &[]);
    let mut unicode_sequential = BpeTrainer::<Character, CharIdx>::from_words(words, &[]);
    let mut unicode_parallel = BpeTrainer::<Character, CharIdx>::from_words(words, &[]);
    build_pre_merges_naive(&mut unicode_reference);
    unicode_sequential._build_pre_merges_with_options(false, 2);
    pool.install(|| unicode_parallel._build_pre_merges_with_options(true, 2));

    let unicode_expected = pre_merge_snapshot(&unicode_reference);
    assert_eq!(pre_merge_snapshot(&unicode_sequential), unicode_expected);
    assert_eq!(pre_merge_snapshot(&unicode_parallel), unicode_expected);
    assert_eq!(unicode_parallel.pre_merges.len(), 5);
    let initial_ni = (CharIdx::Idx(u32::MAX), CharIdx::Char('你'));
    assert_eq!(unicode_parallel.pre_merges.get(&initial_ni).unwrap().data.freq, 25);
    let initial_hao = (CharIdx::Idx(u32::MAX), CharIdx::Char('好'));
    assert_eq!(unicode_parallel.pre_merges.get(&initial_hao).unwrap().data.freq, 7);
    for (tp, freq, occurs_in) in [
      ((CharIdx::Char('你'), CharIdx::Char('你')), 3, vec![0]),
      ((CharIdx::Char('你'), CharIdx::Char('好')), 7, vec![2]),
      ((CharIdx::Char('好'), CharIdx::Char('你')), 7, vec![2]),
    ] {
      let merge = unicode_parallel.pre_merges.get(&tp).unwrap();
      assert_eq!(merge.data.freq, freq);
      assert_eq!(merge.data.occurs_in_vec(), occurs_in);
    }

    unicode_reference.step().unwrap();
    unicode_sequential.step().unwrap();
    unicode_parallel.step().unwrap();
    build_pre_merges_naive(&mut unicode_reference);
    unicode_sequential._build_pre_merges_with_options(false, 2);
    pool.install(|| unicode_parallel._build_pre_merges_with_options(true, 2));

    let unicode_expected = pre_merge_snapshot(&unicode_reference);
    assert_eq!(pre_merge_snapshot(&unicode_sequential), unicode_expected);
    assert_eq!(pre_merge_snapshot(&unicode_parallel), unicode_expected);
    assert_eq!(unicode_parallel.pre_merges.len(), 4);
    assert!(!unicode_parallel.pre_merges.contains_key(&initial_ni));

    let signed_words = [("", 9), ("aa", 3), ("aa", 0), ("aa", -3), ("bb", 1)];
    let mut signed_reference = BpeTrainer::<u8, Idx>::from_words(signed_words, &[]);
    let mut signed_sequential = BpeTrainer::<u8, Idx>::from_words(signed_words, &[]);
    let mut signed_parallel = BpeTrainer::<u8, Idx>::from_words(signed_words, &[]);
    build_pre_merges_naive(&mut signed_reference);
    signed_sequential._build_pre_merges_with_options(false, 1);
    pool.install(|| signed_parallel._build_pre_merges_with_options(true, 1));
    let signed_expected = pre_merge_snapshot(&signed_reference);
    assert_eq!(pre_merge_snapshot(&signed_sequential), signed_expected);
    assert_eq!(pre_merge_snapshot(&signed_parallel), signed_expected);
    assert_eq!(signed_parallel.pre_merges.len(), 2);
    assert_eq!(signed_parallel.pre_merges.get(&aa).unwrap().data.freq, 0);
    assert_eq!(signed_parallel.pre_merges.get(&aa).unwrap().data.occurs_in_vec(), [1, 2, 3]);
    assert_eq!(signed_parallel.pre_merges.get(&bb).unwrap().data.freq, 1);
    assert_eq!(signed_parallel.pre_merges.get(&bb).unwrap().data.occurs_in_vec(), [4]);
    signed_parallel.step().unwrap();
    assert_eq!(signed_parallel.merges.last().unwrap().content.0.debug_display(), "b");
    assert_eq!(signed_parallel.merges.last().unwrap().content.1.debug_display(), "b");
    assert!(signed_parallel.step().is_err());

    let signed_unicode_words = [("你", 3), ("你", 0), ("你", -3), ("ab", 1)];
    let mut signed_unicode_reference = BpeTrainer::<Character, CharIdx>::from_words(
      signed_unicode_words,
      &[],
    );
    let mut signed_unicode_sequential = BpeTrainer::<Character, CharIdx>::from_words(
      signed_unicode_words,
      &[],
    );
    let mut signed_unicode_parallel = BpeTrainer::<Character, CharIdx>::from_words(
      signed_unicode_words,
      &[],
    );
    build_pre_merges_naive(&mut signed_unicode_reference);
    signed_unicode_sequential._build_pre_merges_with_options(false, 1);
    pool.install(|| signed_unicode_parallel._build_pre_merges_with_options(true, 1));
    let signed_unicode_expected = pre_merge_snapshot(&signed_unicode_reference);
    assert_eq!(pre_merge_snapshot(&signed_unicode_sequential), signed_unicode_expected);
    assert_eq!(
      pre_merge_snapshot(&signed_unicode_parallel),
      signed_unicode_expected,
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
      let mut reference = BpeTrainer::<Character, CharIdx>::from_words_with_config(
        words,
        &[],
        config,
      );
      build_pre_merges_naive(&mut reference);
      while reference.step().is_ok() {}

      let mut sequential = BpeTrainer::<Character, CharIdx>::from_words_with_config(
        words,
        &[],
        config,
      );
      sequential._build_pre_merges_with_options(false, 2);
      while sequential.step().is_ok() {}
      assert_unicode_trainers_equal(&sequential, &reference);

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

        assert_unicode_trainers_equal(&parallel, &reference);
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
  fn test_last_merge_freq_ignores_unicode_initial_units() {
    let mut bpe = BpeTrainer::<Character, CharIdx>::from_words(
      [("你好", 7)],
      &[],
    );
    bpe.init_training();
    assert_eq!(bpe.last_merge_freq(), None);

    bpe.step().unwrap();
    assert_eq!(bpe.last_merge_freq(), None);
    bpe.step().unwrap();
    assert_eq!(bpe.last_merge_freq(), None);
    bpe.step().unwrap();
    assert_eq!(bpe.last_merge_freq(), Some(7));

    let model = bpe.validate_model().unwrap();
    assert_eq!(model.last_merge_freq(), Some(7));
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
