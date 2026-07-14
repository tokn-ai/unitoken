use std::{collections::{BTreeMap, HashMap}, io::BufWriter, path::Path};
use std::collections::hash_map::Entry;

use ahash::AHashSet;
use moka::sync::Cache;
use npyz::WriterBuilder;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator as _};

use crate::{
  MyError, MyResult, bigram::VocabBigramIndex,
  pretokenizer::{_read_file_to_buffer, PreTokenPiece, PreTokenizer}, spec::Spec, traits::{CanEncode, CanStrToWord, Decode, Encode},
};

use super::*;

fn validate_unit_model_contract<C: CharSplit>(
  vocab: &BTreeMap<Idx, Word<C>>,
  merges: &[((Idx, Idx), Idx)],
) -> MyResult<()>
where
  Word<C>: WordDebugExt,
{
  for token in vocab.values() {
    if !C::is_valid_model_vocab_word(token) {
      return Err(MyError::InvalidBpeModel(format!(
        "Unicode vocabulary token {} contains fallback bytes; only singleton fallback byte tokens are allowed",
        token.debug_display(),
      )));
    }
  }

  for (rank, ((left, right), target)) in merges.iter().copied().enumerate() {
    for (side, idx) in [("left", left), ("right", right), ("target", target)] {
      let Some(word) = vocab.get(&idx) else {
        continue;
      };
      if !C::is_valid_model_merge_word(word) {
        return Err(MyError::InvalidBpeModel(format!(
          "Unicode merge {rank} {side} token {} contains a fallback byte",
          word.debug_display(),
        )));
      }
    }
  }
  Ok(())
}

#[derive(Clone, Copy, Debug)]
struct EncoderModelCapabilities {
  can_split_on_vocab_bigrams: bool,
  can_batch_encode: bool,
}

fn encoder_model_capabilities<C: Eq>(
  vocab: &BTreeMap<Idx, Word<C>>,
  merges: &[((Idx, Idx), Idx)],
) -> EncoderModelCapabilities {
  // A cut is safe only when every merge target describes the literal source
  // units spanned by its operands. Otherwise an apparently absent bigram can
  // still be crossed by a hand-authored merge.
  let can_split_on_vocab_bigrams = merges.iter().all(|((left, right), target)| {
    let (Some(left), Some(right), Some(target)) = (
      vocab.get(left),
      vocab.get(right),
      vocab.get(target),
    ) else {
      return false;
    };
    target.len() == left.len() + right.len()
      && target.starts_with(left)
      && target.ends_with(right)
  });

  let mut target_ranks = BTreeMap::new();
  let mut merge_pairs = AHashSet::new();
  let mut can_batch_encode = true;
  // `_encode_words` visits each rank once, so dependencies must precede their
  // consumers and pair/target definitions must be unique.
  for (rank, (pair, target)) in merges.iter().copied().enumerate() {
    can_batch_encode &= merge_pairs.insert(pair);
    can_batch_encode &= target_ranks.insert(target, rank).is_none();
  }
  for (rank, ((left, right), _)) in merges.iter().copied().enumerate() {
    for operand in [left, right] {
      if target_ranks.get(&operand).is_some_and(|dependency_rank| *dependency_rank >= rank) {
        can_batch_encode = false;
      }
    }
  }

  EncoderModelCapabilities {
    can_split_on_vocab_bigrams,
    can_batch_encode,
  }
}

#[non_exhaustive]
pub struct BpeBuilder<S = Vec<u8>> {
  pub vocab: Option<BTreeMap<Idx, S>>,
  pub merges: Option<Vec<((Idx, Idx), Idx)>>,
  pub merges_raw: Option<Vec<(S, S)>>,
  pub special_tokens: Option<Vec<String>>,
  pub vocab_size: Option<usize>,
  pub pat_str: Option<String>,
  pub split_on_vocab_bigrams: bool,
}

impl<S> BpeBuilder<S> {
  #[must_use]
  /// Create a new builder with all fields unset.
  pub fn new() -> Self {
    Self {
      vocab: None,
      merges: None,
      merges_raw: None,
      special_tokens: None,
      vocab_size: None,
      pat_str: None,
      split_on_vocab_bigrams: true,
    }
  }

  #[must_use]
  /// Set the vocabulary map from token id to token bytes.
  pub fn set_vocab(self, vocab: BTreeMap<Idx, S>) -> Self {
    Self {
      vocab: Some(vocab),
      ..self
    }
  }

  #[must_use]
  /// Set merges in fully-decoded form: `((left_id, right_id), merged_id)`.
  pub fn set_merges(self, merges: Vec<((Idx, Idx), Idx)>) -> Self {
    Self {
      merges: Some(merges),
      ..self
    }
  }

  #[must_use]
  /// Set merges in raw/token form: `(left_token, right_token)`.
  ///
  /// These will be resolved against the loaded vocabulary during [`Self::build`].
  pub fn set_merges_raw(self, merges_raw: Vec<(S, S)>) -> Self {
    Self {
      merges_raw: Some(merges_raw),
      ..self
    }
  }

  #[must_use]
  /// Convenience alias for [`Self::set_vocab_size(Some(size))`].
  pub fn vocab_size(self, size: usize) -> Self {
    Self {
      vocab_size: Some(size),
      ..self
    }
  }

  #[must_use]
  /// Set an optional vocab size cap.
  ///
  /// Note: the cap may be enforced by the output spec or by the caller.
  pub fn set_vocab_size(self, size: Option<usize>) -> Self {
    Self {
      vocab_size: size,
      ..self
    }
  }

  #[must_use]
  /// Convenience alias for [`Self::set_special_tokens(Some(sp))`].
  pub fn special_tokens(self, sp: Vec<String>) -> Self {
    Self {
      special_tokens: Some(sp),
      ..self
    }
  }

  #[must_use]
  /// Set the special tokens list.
  ///
  /// If `None`, special tokens will be inferred from the beginning of the vocab (when possible).
  pub fn set_special_tokens(self, sp: Option<Vec<String>>) -> Self {
    Self {
      special_tokens: sp,
      ..self
    }
  }

  #[must_use]
  /// Set the regex pattern used by the pre-tokenizer.
  ///
  /// If unset, the crate's default GPT-2-style pattern is used.
  pub fn set_pat_str(self, pat_str: Option<String>) -> Self {
    Self {
      pat_str,
      ..self
    }
  }

  #[must_use]
  /// Enable or disable vocab-derived bigram partitioning during encoding.
  ///
  /// Disabling this optimization preserves token ids while keeping each PAT
  /// word intact for atomic BPE. This avoids the partition scan but may
  /// increase BPE work, so byte workloads should benchmark both settings.
  pub fn set_split_on_vocab_bigrams(self, enabled: bool) -> Self {
    Self {
      split_on_vocab_bigrams: enabled,
      ..self
    }
  }
}

impl<S> Default for BpeBuilder<S> {
  fn default() -> Self {
    Self::new()
  }
}

impl BpeBuilder {
  #[must_use]
  /// Set the vocabulary from a `Word<C>` representation.
  pub fn set_vocab_c<C: CharSplit>(self, vocab: BTreeMap<Idx, Word<C>>) -> Self {
    Self {
      vocab: Some(vocab.into_iter().map(|(k, v)| (k, CharSplit::to_vec_u8(&v))).collect()),
      ..self
    }
  }

  #[must_use]
  /// Load a vocab file using `spec` and store it in this builder.
  pub fn load_vocab_file<C: CharSplit, SPEC: Spec<C, Idx> + ?Sized>(self, filename: impl AsRef<Path>, spec: &SPEC) -> MyResult<Self> {
    println!("Loading vocab file: {}", filename.as_ref().display());
    let file = std::fs::File::open(filename)?;
    spec.decode_vocab(&mut std::io::BufReader::new(file))
      .map(|vocab| self.set_vocab_c(vocab))
  }

  #[must_use]
  /// Load a merges file using `spec` and store it as raw merges in this builder.
  pub fn load_merges_file<C: Clone + CharSplit, SPEC: Spec<C, Idx> + ?Sized>(self, filename: impl AsRef<Path>, spec: &SPEC) -> MyResult<Self> {
    println!("Loading merges file: {}", filename.as_ref().display());
    let file = std::fs::File::open(filename)?;
    let merges = spec.decode_merges_raw(&mut std::io::BufReader::new(file))?;
    let merges_raw = merges.into_iter()
      .map(|m| (CharSplit::to_vec_u8(&m.content.0), CharSplit::to_vec_u8(&m.content.1)))
      .collect::<Vec<_>>();
    Ok(self.set_merges_raw(merges_raw))
  }

  #[must_use]
  /// Build a [`BpeEncoder`] using the configured vocab/merges.
  ///
  /// If only raw merges are provided, they will be resolved against the vocabulary.
  /// Returns an error if a merge references an out-of-vocabulary token.
  pub fn build<C: Clone + Ord + CharSplit + CanStrToWord + Cachable, SPEC: Spec<C, Idx> + ?Sized>(self, _spec: &SPEC) -> MyResult<BpeEncoder<C>>
  where
    Word<C>: WordDebugExt
  {
    let vocab = self.vocab.unwrap_or_default().into_iter().map(|(k, v)| {
      (k, C::from_vec_u8(&v))
    }).collect::<BTreeMap<_, _>>();
    let vocab_rev = vocab.iter().map(|(k, v)| (v.clone(), *k)).collect::<BTreeMap<_, _>>();
    let merges = if let Some(merges) = self.merges {
      merges
    } else if let Some(merges_raw) = self.merges_raw {
       merges_raw.into_iter().map(|(a, b)| {
        let a_w = C::from_vec_u8(&a);
        let b_w = C::from_vec_u8(&b);
        let mut merged = a;
        merged.extend(b);
        let merged_w = C::from_vec_u8(&merged);
        let a_idx = *vocab_rev.get(&a_w).ok_or_else(|| MyError::Oov(a_w.debug_display()))?;
        let b_idx = *vocab_rev.get(&b_w).ok_or_else(|| MyError::Oov(a_w.debug_display()))?;
        let m_idx = *vocab_rev.get(&merged_w).ok_or_else(|| MyError::Oov(a_w.debug_display()))?;
        Ok(((a_idx, b_idx), m_idx))
      }).collect::<MyResult<Vec<((Idx, Idx), Idx)>>>()?
    } else {
      Vec::new()
    };
    let special_tokens = self.special_tokens.unwrap_or_else(|| BpeEncoder::get_special_tokens_from_vocab(&vocab).unwrap_or_default());
    BpeEncoder::new_with_pat_and_vocab_bigrams(
      vocab,
      merges,
      special_tokens,
      self.pat_str.as_deref(),
      self.split_on_vocab_bigrams,
    )
  }
}

#[non_exhaustive]
#[derive(Clone)]
pub struct BpeEncoder<C = u8> {
  pub vocab_bytes: BTreeMap<C, Idx>,
  pub vocab_rev: BTreeMap<Word<C>, Idx>,
  pub vocab: BTreeMap<Idx, Word<C>>,
  pub decode_vocab_bytes: Vec<Box<[u8]>>,
  pub special_tokens: BTreeMap<String, Idx>,
  pub pre_tokenizer: PreTokenizer,
  pub merges: Vec<((Idx, Idx), Idx)>,
  /// with freq represents rank, or `merge.data.freq=-i` for i-th merge.
  /// with [`occurs_in={0}`](MergeData::occurs_in), in order to handle first word in [`Self::_encode_word`].
  pub pre_merge_map: HashMap<(Idx, Idx), Merge<C, Idx>>,
  /// Cache of already-pretokenized words.
  pub cache: Cache<String, Word<Idx>>,
  can_batch_encode: bool,
}

impl<C: Ord + Cachable> BpeEncoder<C>
where
  Word<C>: WordDebugExt,
  C: CanStrToWord + CharSplit,
{
  // fn _load_vocab<R: std::io::Read>(spec: &dyn Spec<C, Idx>, mut reader: R) -> MyResult<BTreeMap<Idx, Word<C>>> {
  //   spec.decode_vocab(&mut reader)
  // }

  // fn _load_merges<R: std::io::Read>(spec: &dyn Spec<C, Idx>, mut reader: R, vocab: &BTreeMap<Idx, Word<C>>) -> MyResult<Vec<Merge<C, Idx>>> {
  //   spec.decode_merges(&mut reader, vocab)
  // }

  // pub fn new_from_file<P1: AsRef<Path>, P2: AsRef<Path>>(
  //   spec: &dyn Spec<C, Idx>, vocab_path: P1, merges_path: P2, special_tokens: Option<Vec<String>>, vocab_size: Option<usize>,
  // ) -> MyResult<Self>
  // where
  //   C: Clone
  // {
  //   let vocab = Self::_load_vocab(spec, std::fs::File::open(vocab_path)?)?;
  //   let merges = Self::_load_merges(spec, std::fs::File::open(merges_path)?, &vocab)?;
  //   let merges = merges.into_iter().map(|m| (m.tp, m.target.unwrap())).take(vocab_size.unwrap_or(usize::MAX)).collect();
  //   let special_tokens = match special_tokens {
  //     Some(tokens) => tokens,
  //     None => Self::get_special_tokens_from_vocab(&vocab)?,
  //   };
  //   Self::new(vocab, merges, special_tokens)
  // }

  /// Infer special tokens from the prefix of the vocabulary.
  ///
  /// The convention used here is: starting at index 0, collect consecutive entries whose token
  /// length is greater than 1. This matches common BPE vocab layouts.
  pub fn get_special_tokens_from_vocab(vocab: &BTreeMap<Idx, Word<C>>) -> MyResult<Vec<String>> {
    let mut special_tokens = Vec::new();
    for index in 0..vocab.len() {
      match vocab.get(&(index as Idx)) {
        Some(token) if token.len() > 1 => special_tokens.push(token.to_string_lossy()),
        _ => break,
      }
    }
    Ok(special_tokens)
  }

  #[hotpath::measure]
  /// Save a sequence of token ids to a `.npy` file.
  pub fn save_idxs_npy<P: AsRef<Path>>(&self, file_path: P, idxs: Vec<Idx>) -> MyResult<()> {
    let mut file = std::fs::File::create(file_path)?;
    let mut writer = npyz::WriteOptions::new()
      .default_dtype()
      .shape(&[idxs.len() as u64])
      .writer(BufWriter::new(&mut file))
      .begin_1d()?;

    writer.extend(idxs)?;
    writer.finish()?;
    Ok(())
  }

  #[cfg(feature = "fmt-npz")]
  #[hotpath::measure]
  /// Save a sequence of token ids to a `.npz` file containing an `idx` array.
  pub fn save_idxs_npz<P: AsRef<Path>>(&self, file_path: P, idxs: Vec<Idx>) -> MyResult<()> {
    let mut file = std::fs::File::create(file_path)?;
    let mut npz = npyz::npz::NpzWriter::new(BufWriter::new(&mut file));
    let mut writer = npz.array("idx", Default::default())?
      .default_dtype()
      .shape(&[idxs.len() as u64])
      .begin_nd()?;

    writer.extend(idxs)?;
    writer.finish()?;
    Ok(())
  }

  /// Construct an encoder from a vocab and merge table.
  ///
  /// - `vocab`: token id → token content
  /// - `merges`: merge rules in rank order as `((left, right), merged)`
  /// - `special_tokens`: list of special tokens; first one is used as the pre-tokenizer EOT marker.
  pub fn new(vocab: BTreeMap<Idx, Word<C>>, merges: Vec<((Idx, Idx), Idx)>, special_tokens: Vec<String>) -> MyResult<Self>
  where
    C: Clone
  {
    Self::new_with_pat(vocab, merges, special_tokens, None)
  }

  /// Construct an encoder from a vocab, merge table, and optional pre-tokenizer pattern.
  pub fn new_with_pat(
    vocab: BTreeMap<Idx, Word<C>>,
    merges: Vec<((Idx, Idx), Idx)>,
    special_tokens: Vec<String>,
    pat_str: Option<&str>,
  ) -> MyResult<Self>
  where
    C: Clone
  {
    Self::new_with_pat_and_vocab_bigrams(
      vocab,
      merges,
      special_tokens,
      pat_str,
      true,
    )
  }

  fn new_with_pat_and_vocab_bigrams(
    vocab: BTreeMap<Idx, Word<C>>,
    merges: Vec<((Idx, Idx), Idx)>,
    special_tokens: Vec<String>,
    pat_str: Option<&str>,
    split_on_vocab_bigrams: bool,
  ) -> MyResult<Self>
  where
    C: Clone,
  {
    let vocab_rev = vocab
      .iter()
      .map(|(k, v)| (v.clone(), *k))
      .collect::<BTreeMap<_, _>>();
    let vocab_bytes = vocab
      .iter()
      .filter_map(|(k, v)| {
        if v.len() == 1 {
          Some((v[0].clone(), *k))
        } else {
          None
        }
      })
      .collect();
    let resolved_special_tokens = special_tokens.iter().map(|token| {
      let word = token.as_str().to_word();
      let idx = *vocab_rev.get(&word).ok_or_else(|| MyError::Oov(word.debug_display()))?;
      Ok((token.clone(), idx))
    }).collect::<MyResult<Vec<_>>>()?;
    let special_token_ids = resolved_special_tokens
      .iter()
      .map(|(_, idx)| *idx)
      .collect::<AHashSet<_>>();
    let merge_target_ids = merges
      .iter()
      .map(|(_, target)| *target)
      .collect::<AHashSet<_>>();
    // Keep merge-target specials indexed: callers can narrow the public
    // special-token regex, making the same text an ordinary BPE word.
    let excluded_vocab_bigram_ids = special_token_ids
      .difference(&merge_target_ids)
      .copied()
      .collect::<AHashSet<_>>();
    validate_unit_model_contract::<C>(&vocab, &merges)?;
    let capabilities = encoder_model_capabilities(&vocab, &merges);
    let vocab_bigram_index = if split_on_vocab_bigrams && capabilities.can_split_on_vocab_bigrams {
      C::build_vocab_bigram_index(&vocab, &excluded_vocab_bigram_ids)
    } else {
      VocabBigramIndex::disabled()
    };
    let pre_merge_map = merges.iter().copied().enumerate().map(|(i, (tp, target))| {
      let mut merge = Merge::new(tp, (
        vocab.get(&tp.0).ok_or_else(|| MyError::OovIdx(tp.0.to_u64())).cloned()?,
        vocab.get(&tp.1).ok_or_else(|| MyError::OovIdx(tp.1.to_u64())).cloned()?,
      )).with_target(target);
      merge.add(0, -(i as Freq));
      Ok((tp, merge))
    }).collect::<MyResult<_>>()?;
    let mut decode_vocab_bytes = vec![Box::<[u8]>::default(); vocab.keys().max().map(|idx| *idx as usize + 1).unwrap_or_default()];
    for (idx, word) in &vocab {
      decode_vocab_bytes[*idx as usize] = CharSplit::to_vec_u8(word).into_boxed_slice();
    }
    let end_of_text = special_tokens.first().cloned();
    let pre_tokenizer = PreTokenizer::try_new(&special_tokens, end_of_text.as_deref(), pat_str)?
      .with_vocab_bigram_index(vocab_bigram_index);
    let special_tokens = resolved_special_tokens.into_iter().collect();
    let max_cap = vocab.len() as u64 * 500;
    Ok(Self {
      vocab_bytes,
      vocab_rev,
      vocab,
      decode_vocab_bytes,
      merges,
      pre_merge_map,
      special_tokens,
      pre_tokenizer,
      cache: Cache::new(max_cap),
      can_batch_encode: capabilities.can_batch_encode,
    })
  }
}

#[hotpath::measure_all]
impl<C> BpeEncoder<C>
where
  C: Ord + Clone + Cachable + CharSplit,
  Word<C>: WordDebugExt,
  C: CanStrToWord,
{
  /// Convert an input word into initial byte/char indices (before merges).
  ///
  /// Returns a [`PreToken`] containing the source token and its initial ids.
  pub fn _pretoken(&self, word: Word<C>, freq: Freq) -> MyResult<PreToken<C, Idx>> {
    let mut idxs = Vec::new();
    for c in word.iter() {
      if let Some(idx) = self.vocab_bytes.get(c) {
        idxs.push(*idx);
        continue;
      }
      // if c is char and not in vocab_bytes, try split it into bytes
      let Some(split) = c.char_split() else {
        return Err(MyError::OovBytes(std::slice::from_ref(c).to_word().debug_display()));
      };
      for b in split {
        if let Some(idx) = self.vocab_bytes.get(&b) {
          idxs.push(*idx);
        } else {
          return Err(MyError::OovBytes(std::slice::from_ref(c).to_word().debug_display()));
        }
      }
    }
    Ok(PreToken { src: word, idxs, freq })
  }

  fn encode_word_uncached(&self, input: &str) -> MyResult<Word<Idx>> {
    self._encode_word(&input.to_word())
  }

  fn encode_word_cached(&self, input: &str) -> MyResult<Word<Idx>> {
    if let Some(result) = self.cache.get(input) {
      return Ok(result);
    }
    let result = self.encode_word_uncached(input)?;
    self.cache.insert(input.to_string(), result.clone());
    Ok(result)
  }

  fn encode_words_uncached(&self, input: &[&str]) -> MyResult<Vec<Word<Idx>>> {
    let words = input.iter().map(|word| word.to_word()).collect::<Vec<_>>();
    if self.can_batch_encode {
      self._encode_words(&words)
    } else {
      words.iter().map(|word| self._encode_word(word)).collect()
    }
  }

  fn _new_pre_merge_map(&self) -> HashMap<(Idx, Idx), Merge<C, Idx>> {
    let mut pre_merges = self.pre_merge_map.clone();
    pre_merges.iter_mut().for_each(|i| {
      i.1.data.freq = 0;
      i.1.data.occurs_in.clear();
    });
    pre_merges
  }

  /// this would merge pairs in all [`Word`] in `input`,
  /// in the same order and similar precedure of trainer.
  ///
  /// this method would be useful if you would like to build cache,
  /// or have large number of words to be encode at one time.
  ///
  /// See [`Self::encode_words`] for cached version
  pub fn _encode_words(&self, input: &[Word<C>]) -> MyResult<Vec<Word<Idx>>> {
    if input.len() == 0 {
      return Ok(Vec::new());
    }
    let mut words = input
      .iter()
      .map(|w| self._pretoken(w.clone(), 1))
      .collect::<Result<Vec<_>, _>>()?;
    let mut pre_merges = self._new_pre_merge_map();

    // init
    for (i, word) in words.iter().enumerate() {
      for (j1, j2) in word.idxs.iter().copied().zip(word.idxs.iter().skip(1).copied()) {
        let tp = (j1, j2);
        if let Some(merge) = pre_merges.get_mut(&tp) {
          merge.add(i as u64, 1);
        }
      }
    }

    // merge
    for (tp, target) in &self.merges {
      let Some(merge) = pre_merges.remove(&tp) else {
        continue;
      };
      let changes = _merge(&mut words, &merge, *target, None);
      _update_merge_map(&mut pre_merges, &merge, changes, None);
    }

    Ok(words.into_iter().map(|i| i.idxs.to_word()).collect())
  }

  // #[hotpath::measure]
  /// Encode many words, using an internal cache to avoid recomputing repeats.
  ///
  /// The output ordering matches the input iteration order.
  pub fn encode_words_impl<S: AsRef<str>, I: IntoIterator<Item = S>>(&self, input: I) -> MyResult<Vec<Word<Idx>>> {
    let input = input.into_iter().collect::<Vec<_>>();
    let mut word_by_text: ahash::AHashMap<&str, usize> = ahash::AHashMap::new();
    let mut words = Vec::new();
    let mut encoded_words = Vec::new();
    let mut order = Vec::with_capacity(input.len());

    for word in &input {
      let word = word.as_ref();
      let word_idx = match word_by_text.entry(word) {
        Entry::Occupied(entry) => *entry.get(),
        Entry::Vacant(entry) => {
          let word_idx = words.len();
          words.push(word);
          encoded_words.push(self.cache.get(word));
          entry.insert(word_idx);
          word_idx
        }
      };
      order.push(word_idx);
    }

    let missing_word_ids = encoded_words
      .iter()
      .enumerate()
      .filter_map(|(idx, encoded)| encoded.is_none().then_some(idx))
      .collect::<Vec<_>>();
    let missing_words = missing_word_ids
      .iter()
      .map(|idx| words[*idx])
      .collect::<Vec<_>>();
    let encoded_missing = self.encode_words_uncached(&missing_words)?;
    for (word_idx, encoded) in missing_word_ids.into_iter().zip(encoded_missing) {
      self.cache.insert(words[word_idx].to_string(), encoded.clone());
      encoded_words[word_idx] = Some(encoded);
    }

    Ok(order.into_iter().map(|word_idx| {
      encoded_words[word_idx].as_ref().unwrap().clone()
    }).collect())
  }

  /// encode a single word without cache.
  /// see [`Self::encode_word`] for cached version.
  pub fn _encode_word(&self, input: &Word<C>) -> MyResult<Word<Idx>> {
    let mut queue = BTreeMap::new();
    let mut words = vec![self._pretoken(input.clone(), 1)?];
    for (i1, i2) in words[0].idxs.iter().copied().zip(words[0].idxs.iter().skip(1).copied()) {
      let tp = (i1, i2);
      if let Some(merge) = self.pre_merge_map.get(&tp) {
        queue.insert((merge.data.freq, tp), merge);
      }
    }
    while let Some((_, merge)) = queue.pop_last() {
      let changes = _merge(&mut words, merge, merge.target.unwrap(), None);
      for (tp, data) in changes {
        if data.occurs_in.is_empty() {
          continue;
        }
        let Some(merge) = self.pre_merge_map.get(&tp) else {
          continue;
        };
        if data.freq < 0 {
          queue.remove(&(merge.data.freq, tp));
        } else {
          queue.insert((merge.data.freq, tp), merge);
        }
      }
    }
    Ok(words.into_iter().next().unwrap().idxs.to_word())
  }

  fn encode_string_ordered(&self, input: &str) -> MyResult<Vec<Idx>> {
    let mut special_piece_by_text: ahash::AHashMap<&str, usize> = ahash::AHashMap::new();
    let mut word_piece_by_text: ahash::AHashMap<&str, usize> = ahash::AHashMap::new();
    let mut pieces: Vec<Word<Idx>> = Vec::new();
    let mut ordered_pieces = Vec::with_capacity(input.len() / 4);
    let mut final_len = 0;
    self.pre_tokenizer.for_each_piece(input, |piece| {
      let piece_idx = match piece {
        PreTokenPiece::Special(token) => {
          match special_piece_by_text.entry(token) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
              let idx = *self.special_tokens.get(token)
                .ok_or_else(|| MyError::Oov(token.to_string()))?;
              let piece_idx = pieces.len();
              pieces.push(Arc::from([idx]));
              entry.insert(piece_idx);
              piece_idx
            }
          }
        }
        PreTokenPiece::Word(word) => {
          match word_piece_by_text.entry(word) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
              let encoded = self.encode_word_cached(word)?;
              let piece_idx = pieces.len();
              pieces.push(encoded);
              entry.insert(piece_idx);
              piece_idx
            }
          }
        }
      };
      final_len += pieces[piece_idx].len();
      ordered_pieces.push(piece_idx);
      Ok(())
    })?;

    let mut encoded = Vec::with_capacity(final_len);
    for piece_idx in ordered_pieces {
      encoded.extend_from_slice(&pieces[piece_idx]);
    }
    Ok(encoded)
  }

  // #[hotpath::measure]
  fn _create_cache_from_words(
    &self, input: Vec<String>
  ) -> MyResult<OrderMap<String, Arc<[Idx]>>> {
    let words = input.iter().map(String::as_str).collect::<Vec<_>>();
    let encoded = self.encode_words_uncached(&words)?;
    Ok(input.into_iter().zip(encoded).rev().collect())
  }

  /// Replace the already-pretokenized word cache with the provided entries.
  ///
  /// This is mainly useful for warm-starting an encoder when encoding large corpora.
  pub fn with_cache(mut self, cache: OrderMap<String, Arc<[Idx]>>) -> Self {
    let max_cap = cache.len() as u64 * 3 / 2;
    self.cache = Cache::new(max_cap);
    for (k, v) in cache {
      self.cache.insert(k, v);
    }
    self
  }

  #[deprecated(note = "use `encode_file` instead")]
  /// Encode a file after building a cache from the file's word inventory.
  ///
  /// Deprecated: prefer [`Encode::encode_file`] / [`Self::encode_file_impl`].
  pub fn encode_file_with_cache<P: AsRef<Path>>(
    &self, path: P, num_chunks: usize,
  ) -> MyResult<Vec<Idx>> {
    let words = self.pre_tokenizer.get_words_from_file(&path, num_chunks)?;
    let input = words.into_iter().map(|(k, _)| k).collect::<Vec<_>>();
    let cache = self._create_cache_from_words(input)?;
    let bpe_with_cache = self.clone().with_cache(cache);
    bpe_with_cache.encode_file(path.as_ref(), num_chunks)
  }

  /// Encode a file into token ids.
  ///
  /// The file is split into `num_chunks` aligned on the pre-tokenizer's EOT marker.
  pub fn encode_file_impl<P: AsRef<Path>>(
    &self, path: P, num_chunks: usize,
  ) -> MyResult<Vec<Idx>> {
    let boundaries = self.pre_tokenizer.find_chunk_boundaries(&path, num_chunks)?;
    let path = path.as_ref().to_path_buf();

    debug!("Start encoding file in {num_chunks} chunks...");
    let mut segments_tokens_index = boundaries.into_par_iter()
      .enumerate()
      .map(|(index, (offset, len))| {
        let buffer = _read_file_to_buffer(&path, offset, len)?;
        let content = String::from_utf8_lossy(&buffer);
        self.encode_string(&content).map(|v| (index, v))
      }).collect::<MyResult<Vec<_>>>()?;

    debug!("Finished encoding segments, merging results...");
    segments_tokens_index.sort_by(|(ida, _), (idb, _)| { ida.cmp(idb) });

    let result = segments_tokens_index.into_iter().map(|(_, idxs)| idxs).flatten().collect::<Vec<_>>();
    Ok(result)
  }
}

impl<C> Encode<Idx> for BpeEncoder<C>
where
  BpeEncoder<C>: CanEncode<C, Idx>,
{
  fn pre_tokenizer(&self) -> &PreTokenizer {
    &self.pre_tokenizer
  }

  #[hotpath::measure]
  fn encode_word(&self, input: &str) -> MyResult<Word<Idx>> {
    self.encode_word_cached(input)
  }

  fn encode_words(&self, words: &[&str]) -> MyResult<Vec<Word<Idx>>> {
    self.encode_words_impl(words)
  }

  #[hotpath::measure]
  fn encode_string(&self, input: &str) -> MyResult<Vec<Idx>> {
    self.encode_string_ordered(input)
  }

  fn encode_file(
    &self, path: &Path, num_chunks: usize,
  ) -> MyResult<Vec<Idx>> {
    self.encode_file_impl(path, num_chunks)
  }
}

impl<C> Decode<Idx> for BpeEncoder<C>
where
  BpeEncoder<C>: CanEncode<C, Idx>,
  C: Clone,
{
  fn decode(&self, idxs: &[Idx]) -> MyResult<String> {
    BpeEncoder::<C>::decode(self, idxs)
  }
}

#[hotpath::measure_all]
impl<C: Clone> BpeEncoder<C>
where
  Word<C>: WordDebugExt,
  C: CharSplit,
{
  /// Convert token ids back into vocab entries.
  pub fn _decode(&self, idxs: &[Idx]) -> MyResult<Vec<Word<C>>> {
    let mut result = Vec::with_capacity(idxs.len());
    for idx in idxs {
      if let Some(word) = self.vocab.get(idx) {
        result.push(word.clone());
      } else {
        return Err(MyError::OovIdx(idx.to_u64()));
      }
    }
    Ok(result)
  }

  /// Decode token ids back into a UTF-8 string.
  ///
  /// This concatenates decoded token bytes and performs a lossy UTF-8 conversion.
  pub fn decode(&self, idxs: &[Idx]) -> MyResult<String> {
    let mut result = Vec::with_capacity(idxs.len().saturating_mul(4));
    for idx in idxs {
      let bytes = self.decode_vocab_bytes
        .get(*idx as usize)
        .filter(|bytes| !bytes.is_empty())
        .ok_or_else(|| MyError::OovIdx(idx.to_u64()))?;
      result.extend_from_slice(bytes);
    }
    Ok(String::from_utf8(result).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).to_string()))
  }
}

#[cfg(test)]
mod tests {
  use crate::{bigram::Bigram, spec::{gpt2::Gpt2Spec, unitoken::UnitokenSpec}, traits::CanEncode};

  use super::*;

  fn _setup_bpe<C>(name: &str, spec: &dyn Spec<C, Idx>) -> BpeEncoder<C>
  where
    BpeEncoder<C>: CanEncode<C, Idx>
  {
    let bpe = BpeBuilder::new()
      .load_merges_file(format!("fixtures/merges.{name}.txt"), spec).unwrap()
      .load_vocab_file(format!("fixtures/vocab.{name}.json"), spec).unwrap()
      // .special_tokens(vec![DEFAULT_EOT.to_string()])
      .build(spec).unwrap();

    // let vocab = BpeEncoder::_load_vocab(&spec, std::fs::File::open(format!("fixtures/vocab.{name}.json")).unwrap()).unwrap();
    // let merges = BpeEncoder::_load_merges(&spec, std::fs::File::open(format!("fixtures/merges.{name}.txt")).unwrap(), &vocab).unwrap();
    // let merges = merges.into_iter().map(|m| (m.tp, m.target.unwrap())).collect();
    // BpeEncoder::new(vocab, merges, vec![DEFAULT_EOT.to_string()]).unwrap()
    bpe
  }

  fn segments<'a>(index: &VocabBigramIndex, input: &'a str) -> Vec<&'a str> {
    let mut split_points = Vec::new();
    index.split_points(input, &mut split_points);
    let mut start = 0;
    split_points.into_iter().chain(std::iter::once(input.len())).map(|end| {
      let segment = &input[start..end];
      start = end;
      segment
    }).collect()
  }

  fn byte_boundary_encoder() -> BpeEncoder<u8> {
    let vocab = BTreeMap::from([
      (0, "a".to_word()),
      (1, "b".to_word()),
      (2, "c".to_word()),
      (3, "d".to_word()),
      (4, "e".to_word()),
      (5, "f".to_word()),
      (6, "g".to_word()),
      (7, "h".to_word()),
      (8, "i".to_word()),
      (9, "ab".to_word()),
      (10, "abc".to_word()),
      (11, "de".to_word()),
    ]);
    BpeEncoder::new(
      vocab,
      vec![
        ((0, 1), 9),
        ((9, 2), 10),
        ((3, 4), 11),
      ],
      vec![],
    ).unwrap()
  }

  #[test]
  fn test_vocab_bigram_index_scans_long_tokens() {
    let byte_vocab = BTreeMap::from([
      (0, "abc".to_word()),
      (1, "de".to_word()),
      (2, "special".to_word()),
    ]);
    let byte_index = u8::build_vocab_bigram_index(&byte_vocab, &AHashSet::from_iter([2]));
    assert!(byte_index.contains_byte(Bigram::new(b'a', b'b')));
    assert!(byte_index.contains_byte(Bigram::new(b'b', b'c')));
    assert!(byte_index.contains_byte(Bigram::new(b'd', b'e')));
    assert!(!byte_index.contains_byte(Bigram::new(b's', b'p')));

    let unicode_vocab: BTreeMap<Idx, Word<Character>> = BTreeMap::from([
      (0, "你好世".to_word()),
      (1, "a你".to_word()),
    ]);
    let unicode_index = Character::build_vocab_bigram_index(&unicode_vocab, &AHashSet::new());
    let unicode_bigrams = unicode_index.unicode_bigrams().unwrap();
    assert!(unicode_bigrams.contains(&Bigram::new('你', '好')));
    assert!(unicode_bigrams.contains(&Bigram::new('好', '世')));
    assert!(unicode_bigrams.contains(&Bigram::new('a', '你')));
  }

  #[test]
  fn test_vocab_bigram_splitter_preserves_utf8_boundaries() {
    let byte_index = VocabBigramIndex::byte();
    assert_eq!(segments(&byte_index, "short"), ["short"]);
    assert_eq!(segments(&byte_index, "é"), ["é"]);
    let byte_input = format!("{}é你bb", "a".repeat(28));
    let byte_segments = segments(&byte_index, &byte_input);
    assert_eq!(byte_segments.concat(), byte_input);
    assert_eq!(byte_segments.len(), 32);
    assert!(byte_segments.iter().all(|segment| segment.chars().count() == 1));

    let unicode_index = VocabBigramIndex::unicode(AHashSet::from_iter([
      Bigram::new('你', '好'),
      Bigram::new('世', '界'),
    ]));
    assert_eq!(segments(&unicode_index, "你好甲世界"), ["你好", "甲", "世界"]);
    assert_eq!(segments(&unicode_index, "𠮷你é"), ["𠮷", "你", "é"]);
  }

  #[test]
  fn test_encode_word_caches_only_the_atomic_word() {
    let bpe = byte_boundary_encoder();
    let input = format!("abcdef{}", "g".repeat(26));
    let mut expected = vec![10, 11, 5];
    expected.extend(std::iter::repeat_n(6, 26));
    assert_eq!(bpe.encode_word(&input).unwrap().as_ref(), expected);
    assert_eq!(bpe.cache.get(&input).unwrap().as_ref(), expected);
    assert!(bpe.cache.get("abc").is_none());
    assert!(bpe.cache.get("de").is_none());
    assert!(bpe.cache.get("f").is_none());

    let cache = bpe._create_cache_from_words(vec![input.clone()]).unwrap();
    assert_eq!(cache.keys().collect::<Vec<_>>(), [&input]);
  }

  #[test]
  fn test_encode_word_reuses_the_atomic_word_cache() {
    let bpe = byte_boundary_encoder();
    bpe.cache.insert("ac".to_string(), Arc::from([8, 0]));

    assert_eq!(bpe.encode_word("ac").unwrap().as_ref(), [8, 0]);
  }

  #[test]
  fn test_vocab_segments_preserve_unsplit_encoding() {
    let bpe = byte_boundary_encoder();
    let short_input = "abcdefg";
    let input = format!("abcdef{}", "g".repeat(26));
    let longer_input = format!("{input}h");
    let unsplit = bpe._encode_word(&input.as_str().to_word()).unwrap();
    let atomic = bpe.encode_word(&input).unwrap();
    assert_eq!(atomic, unsplit);
    assert_eq!(bpe.encode_string(short_input).unwrap(), bpe.encode_word(short_input).unwrap().as_ref());
    assert_eq!(bpe.encode_string(&input).unwrap(), unsplit.as_ref());
    assert_eq!(bpe.encode_words(&[&input, &longer_input, &input]).unwrap(), [
      bpe.encode_word(&input).unwrap(),
      bpe.encode_word(&longer_input).unwrap(),
      bpe.encode_word(&input).unwrap(),
    ]);
  }

  #[test]
  fn test_vocab_bigram_splitting_can_be_disabled() {
    let vocab: BTreeMap<Idx, Word<u8>> = BTreeMap::from([
      (0, "a".to_word()),
      (1, "b".to_word()),
      (2, "ab".to_word()),
      (3, "x".to_word()),
    ]);
    let merges = vec![((0, 1), 2)];
    let enabled: BpeEncoder<u8> = BpeBuilder::new()
      .set_vocab_c(vocab.clone())
      .set_merges(merges.clone())
      .build(&UnitokenSpec)
      .unwrap();
    let disabled: BpeEncoder<u8> = BpeBuilder::new()
      .set_vocab_c(vocab)
      .set_merges(merges)
      .set_split_on_vocab_bigrams(false)
      .build(&UnitokenSpec)
      .unwrap();
    let input = format!("ab{}", "x".repeat(32));

    assert_eq!(enabled.pre_tokenizer.get_words(&input).unwrap(), BTreeMap::from([
      ("ab", 1),
      ("x", 32),
    ]));
    assert_eq!(disabled.pre_tokenizer.get_words(&input).unwrap(), BTreeMap::from([
      (input.as_str(), 1),
    ]));

    let expected = disabled.encode_word(&input).unwrap();
    assert_eq!(enabled.encode_string(&input).unwrap(), expected.as_ref());
    assert_eq!(disabled.encode_string(&input).unwrap(), expected.as_ref());
  }

  #[test]
  fn test_unicode_pretokenization_preserves_ascii_cjk_and_fallback_encoding() {
    let vocab: BTreeMap<Idx, Word<Character>> = BTreeMap::from([
      (0, "a".to_word()),
      (1, "你".to_word()),
      (2, "好".to_word()),
      (3, "世".to_word()),
      (4, "界".to_word()),
      (5, vec![Character::Byte(0xf0)].to_word()),
      (6, vec![Character::Byte(0xa0)].to_word()),
      (7, vec![Character::Byte(0xae)].to_word()),
      (8, vec![Character::Byte(0xb7)].to_word()),
      (9, "你好".to_word()),
      (10, "世界".to_word()),
    ]);
    let bpe = BpeEncoder::new(
      vocab,
      vec![
        ((1, 2), 9),
        ((3, 4), 10),
      ],
      vec![],
    ).unwrap();
    let input = "a你好𠮷世界";

    assert_eq!(bpe.pre_tokenizer.get_words(input).unwrap(), BTreeMap::from([
      ("a", 1),
      ("你好", 1),
      ("世界", 1),
      ("𠮷", 1),
    ]));
    let atomic = bpe.encode_word(input).unwrap();
    let encoded = bpe.encode_string(input).unwrap();
    assert_eq!(atomic.as_ref(), [0, 9, 5, 6, 7, 8, 10]);
    assert_eq!(encoded, atomic.as_ref());
    assert_eq!(bpe.decode(&atomic).unwrap(), input);
  }

  #[test]
  fn test_disabling_vocab_bigrams_keeps_configured_unicode_bigrams() {
    let vocab: BTreeMap<Idx, Word<Character>> = BTreeMap::from([
      (0, "你".to_word()),
      (1, "好".to_word()),
      (2, "世".to_word()),
      (3, "界".to_word()),
      (4, "你好".to_word()),
    ]);
    let mut bpe: BpeEncoder<Character> = BpeBuilder::new()
      .set_vocab_c(vocab)
      .set_merges(vec![((0, 1), 4)])
      .set_split_on_vocab_bigrams(false)
      .build(&UnitokenSpec)
      .unwrap();
    bpe.pre_tokenizer = bpe
      .pre_tokenizer
      .clone()
      .with_unicode_bigrams(AHashSet::from_iter([('你', '好')]));

    assert_eq!(bpe.pre_tokenizer.get_words("你好世界").unwrap(), BTreeMap::from([
      ("世", 1),
      ("你好", 1),
      ("界", 1),
    ]));
    assert_eq!(bpe.encode_string("你好世界").unwrap(), [4, 2, 3]);
  }

  #[test]
  fn test_atomic_word_encoding_respects_competing_merge_ranks() {
    let vocab: BTreeMap<Idx, Word<u8>> = BTreeMap::from([
      (0, "a".to_word()),
      (1, "b".to_word()),
      (2, "c".to_word()),
      (3, "d".to_word()),
      (4, "bc".to_word()),
      (5, "ab".to_word()),
      (6, "abc".to_word()),
      (7, "x".to_word()),
    ]);
    let bpe = BpeEncoder::new(
      vocab,
      vec![
        ((1, 2), 4),
        ((0, 1), 5),
        ((5, 2), 6),
      ],
      vec![],
    ).unwrap();

    assert_eq!(bpe.encode_word("abc").unwrap().as_ref(), [0, 4]);
    assert_eq!(bpe.encode_words(&["abc"]).unwrap()[0].as_ref(), [0, 4]);

    let input = format!("abcd{}", "x".repeat(28));
    let mut expected = vec![0, 4, 3];
    expected.extend(std::iter::repeat_n(7, 28));
    assert_eq!(bpe.encode_string(&input).unwrap(), expected);
  }

  #[test]
  fn test_atomic_word_does_not_promote_orphan_multiunit_vocab_tokens() {
    let vocab: BTreeMap<Idx, Word<u8>> = BTreeMap::from([
      (0, "a".to_word()),
      (1, "b".to_word()),
      (2, "c".to_word()),
      (3, "ab".to_word()),
      (4, "x".to_word()),
    ]);
    let bpe = BpeEncoder::new(vocab, vec![], vec![]).unwrap();

    let input = format!("abc{}", "x".repeat(29));
    let mut expected = vec![0, 1, 2];
    expected.extend(std::iter::repeat_n(4, 29));
    assert_eq!(bpe.encode_word("ab").unwrap().as_ref(), [0, 1]);
    assert_eq!(bpe.encode_words(&["ab"]).unwrap()[0].as_ref(), [0, 1]);
    assert_eq!(bpe.encode_string(&input).unwrap(), expected);
  }

  #[test]
  fn test_encode_word_does_not_apply_pat_or_special_token_routing() {
    let vocab: BTreeMap<Idx, Word<u8>> = BTreeMap::from([
      (0, "a".to_word()),
      (1, "b".to_word()),
      (2, " ".to_word()),
      (3, "ab".to_word()),
    ]);
    let bpe = BpeEncoder::new_with_pat(vocab, vec![], vec!["ab".to_string()], Some(r"[^\s]")).unwrap();

    assert_eq!(bpe.encode_word("a b").unwrap().as_ref(), [0, 2, 1]);
    assert_eq!(bpe.encode_string("a b").unwrap(), [0, 1]);
    assert_eq!(bpe.encode_word("ab").unwrap().as_ref(), [0, 1]);
    assert_eq!(bpe.encode_string("ab").unwrap(), [3]);
  }

  #[test]
  fn test_special_routing_is_independent_of_atomic_word_cache_order() {
    let vocab: BTreeMap<Idx, Word<u8>> = BTreeMap::from([
      (0, "a".to_word()),
      (1, "b".to_word()),
      (2, "x".to_word()),
      (3, "y".to_word()),
      (4, "z".to_word()),
      (5, "q".to_word()),
      (6, "ab".to_word()),
      (7, "zabq".to_word()),
    ]);
    let bpe = BpeEncoder::new(vocab, vec![], vec!["ab".to_string()]).unwrap();

    assert_eq!(bpe.encode_word("ab").unwrap().as_ref(), [0, 1]);
    assert_eq!(bpe.cache.get("ab").unwrap().as_ref(), [0, 1]);
    assert_eq!(bpe.encode_string("ab").unwrap(), [6]);

    let bpe = BpeEncoder::new(
      bpe.vocab.clone(),
      vec![],
      vec!["ab".to_string()],
    ).unwrap();
    assert_eq!(bpe.encode_string("ab").unwrap(), [6]);
    assert!(bpe.cache.get("ab").is_none());
    assert_eq!(bpe.encode_word("ab").unwrap().as_ref(), [0, 1]);
  }

  #[test]
  fn test_special_and_word_pieces_with_the_same_text_do_not_alias() {
    let vocab: BTreeMap<Idx, Word<u8>> = BTreeMap::from([
      (0, "a".to_word()),
      (1, "b".to_word()),
      (2, "!".to_word()),
      (3, "?".to_word()),
      (4, "ab".to_word()),
    ]);
    let mut bpe = BpeEncoder::new_with_pat(
      vocab,
      vec![],
      vec!["ab".to_string()],
      Some(r"ab|."),
    ).unwrap();
    bpe.pre_tokenizer.re_special_tokens = fancy_regex::Regex::new(r"ab(?=!)").unwrap();

    assert_eq!(bpe.encode_string("ab!ab?").unwrap(), [4, 2, 0, 1, 3]);
  }

  #[test]
  fn test_merge_target_special_remains_safe_in_an_ordinary_word() {
    let vocab: BTreeMap<Idx, Word<Character>> = BTreeMap::from([
      (0, "你".to_word()),
      (1, "好".to_word()),
      (2, "你好".to_word()),
      (3, "?".to_word()),
    ]);
    let mut bpe = BpeEncoder::new_with_pat(
      vocab,
      vec![((0, 1), 2)],
      vec!["你好".to_string()],
      Some(r"\p{L}+|."),
    ).unwrap();
    bpe.pre_tokenizer.re_special_tokens = fancy_regex::Regex::new(r"你好(?=!)").unwrap();

    assert_eq!(bpe.encode_string("你好?").unwrap(), [2, 3]);
  }

  #[test]
  fn test_pretokenized_words_reuse_the_atomic_word_cache() {
    fn encoder() -> BpeEncoder<u8> {
      BpeEncoder::new(
        BTreeMap::from([
          (0, "a".to_word()),
          (1, "b".to_word()),
          (2, "c".to_word()),
          (3, "ab".to_word()),
          (4, "x".to_word()),
        ]),
        vec![],
        vec![],
      ).unwrap()
    }

    let input = format!("cab{}", "x".repeat(29));
    let warmed = encoder().with_cache(OrderMap::from_iter([
      ("ab".to_string(), Arc::from([4, 2])),
    ]));
    assert!(warmed.pre_tokenizer.get_words(&input).unwrap().contains_key("ab"));
    assert_eq!(warmed.encode_string(&input).unwrap()[..3], [2, 4, 2]);
    assert_eq!(warmed.cache.get("ab").unwrap().as_ref(), [4, 2]);
  }

  #[test]
  fn test_malformed_merge_disables_vocab_bigram_splitting() {
    let vocab: BTreeMap<Idx, Word<u8>> = BTreeMap::from([
      (0, "a".to_word()),
      (1, "b".to_word()),
      (2, "z".to_word()),
    ]);
    let bpe = BpeEncoder::new(vocab, vec![((0, 1), 2)], vec![]).unwrap();
    let input = format!("ab{}", "a".repeat(30));

    assert_eq!(bpe.pre_tokenizer.get_words(&input).unwrap(), BTreeMap::from([
      (input.as_str(), 1),
    ]));
    assert!(bpe.can_batch_encode);
    assert_eq!(bpe.encode_word("ab").unwrap().as_ref(), [2]);
    assert_eq!(bpe.encode_words(&["ab"]).unwrap()[0].as_ref(), [2]);
  }

  #[test]
  fn test_encode_words_falls_back_for_forward_merge_dependencies() {
    let vocab: BTreeMap<Idx, Word<u8>> = BTreeMap::from([
      (0, "a".to_word()),
      (1, "b".to_word()),
      (2, "c".to_word()),
      (3, "ab".to_word()),
      (4, "abc".to_word()),
      (5, "x".to_word()),
    ]);
    let bpe = BpeEncoder::new(
      vocab,
      vec![
        ((3, 2), 4),
        ((0, 1), 3),
      ],
      vec![],
    ).unwrap();
    let input = format!("abc{}", "x".repeat(29));
    let mut expected = vec![4];
    expected.extend(std::iter::repeat_n(5, 29));

    assert!(!bpe.can_batch_encode);
    assert_eq!(bpe.encode_word(&input).unwrap().as_ref(), expected);
    assert_eq!(bpe.encode_words(&[&input]).unwrap()[0].as_ref(), expected);
  }

  #[test]
  fn test_unicode_encoder_rejects_fallback_byte_tokens_and_merges() {
    let mixed_vocab = BTreeMap::from([
      (0, vec![Character::Byte(0x80)].to_word()),
      (1, "a".to_word()),
      (2, vec![Character::Byte(0x80), Character::Unicode('a')].to_word()),
    ]);
    let error = match BpeEncoder::new(mixed_vocab, vec![], vec![]) {
      Ok(_) => panic!("mixed Unicode vocab token should be rejected"),
      Err(error) => error,
    };
    assert!(error.to_string().contains("only singleton fallback byte tokens are allowed"));

    let byte_merge_vocab = BTreeMap::from([
      (0, vec![Character::Byte(0xc3)].to_word()),
      (1, vec![Character::Byte(0xa9)].to_word()),
      (2, "é".to_word()),
    ]);
    let error = match BpeEncoder::new(byte_merge_vocab, vec![((0, 1), 2)], vec![]) {
      Ok(_) => panic!("Unicode fallback byte merge should be rejected"),
      Err(error) => error,
    };
    assert!(error.to_string().contains("merge 0 left token"));
  }

  #[test]
  fn test_bpe_encode_words() {
    const NAME: &str = "tinystories_sample_5M";
    // const NAME: &str = "TinyStoriesV2-GPT4-train";
    let input: BTreeMap<String, Freq> = serde_json::from_str(&std::fs::read_to_string(format!("fixtures/_words.{NAME}.json")).unwrap()).unwrap();
    let input = input.into_iter().map(|(k, _)| k.to_word()).collect::<Vec<_>>();
    let bpe = _setup_bpe(NAME, &Gpt2Spec);
    let result = bpe._encode_words(&input).unwrap();
    assert_eq!(result.len(), input.len());

    let result2 = input.iter().map(|w| bpe._encode_word(w).unwrap()).collect::<Vec<_>>();
    assert_eq!(result, result2);
    // for ((i, src), (r1, r2)) in input.iter().enumerate().zip(result.iter().zip(result2.iter())) {
    //   assert_eq!(r1, r2, "[{i}] src={}", src.display());
    // }
  }

  #[test]
  fn test_cache() {
    const NAME: &str = "tinystories_sample_5M";
    let input: BTreeMap<String, Freq> = serde_json::from_str(&std::fs::read_to_string(format!("fixtures/_words.{NAME}.json")).unwrap()).unwrap();
    let input = input.iter().map(|(k, _)| k).collect::<Vec<_>>();
    let mut bpe = _setup_bpe(NAME, &Gpt2Spec);
    bpe.cache = Cache::new(input.len() as u64 * 6 / 5);
    let result1 = bpe.encode_words_impl(&input).unwrap();
    let result2 = bpe.encode_words_impl(&input).unwrap();
    assert_eq!(result1, result2);
    assert!(input.iter().all(|word| bpe.cache.get(word.as_str()).is_some()));
  }

  #[test]
  fn test_encode_string() {
    const NAME: &str = "tinystories_sample_5M";
    let bpe = _setup_bpe(NAME, &Gpt2Spec);
    let input = std::fs::read_to_string(format!("fixtures/{NAME}.txt")).unwrap();
    let result = bpe.encode_string(&input).unwrap();
    assert_eq!(result.len(), 1424317);
  }

  #[test]
  fn test_bpe_encode_file() {
    const NAME: &str = "tinystories_sample_5M";
    let bpe = _setup_bpe(NAME, &Gpt2Spec);
    let result = bpe.encode_file(
      format!("fixtures/{NAME}.txt").as_ref(),
      1,
    ).unwrap();
    // assert!(result.len() == 1269588);
    // let total_index: usize = result.iter().map(|idxs| idxs.len()).sum();
    assert_eq!(result.len(), 1424317);
  }

  #[test]
  fn test_bpe_encode_file_uni() {
    const NAME: &str = "TinyStories_all_data_zh_1M-sample";
    let bpe = _setup_bpe::<Character>(&format!("{NAME}.uni"), &UnitokenSpec);
    let result = bpe.encode_file(
      format!("fixtures/{NAME}.txt").as_ref(),
      1,
    ).unwrap();
    assert_eq!(result.len(), 886572);
    let decoded = bpe.decode(&result).unwrap();
    let input = std::fs::read_to_string(format!("fixtures/{NAME}.txt")).unwrap();
    assert_eq!(decoded.len(), 5292796);
    assert_eq!(decoded, input);
  }
}
