use std::{collections::{BTreeMap, HashMap}, io::BufWriter, path::Path, usize};

use moka::sync::Cache;
use npyz::WriterBuilder;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator as _};

use crate::{
  MyError, MyResult,
  pretokenizer::{_read_file_to_buffer, PreTokenizer}, spec::Spec, traits::{CanEncode, CanStrToWord, Decode, Encode},
};

use super::*;

pub struct BpeBuilder<S = Vec<u8>> {
  pub vocab: Option<BTreeMap<Idx, S>>,
  pub merges: Option<Vec<((Idx, Idx), Idx)>>,
  pub merges_raw: Option<Vec<(S, S)>>,
  pub special_tokens: Option<Vec<String>>,
  pub vocab_size: Option<usize>,
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
    BpeEncoder::new(vocab, merges, special_tokens)
  }
}

#[derive(Clone)]
pub struct BpeEncoder<C = u8> {
  pub vocab_bytes: BTreeMap<C, Idx>,
  pub vocab_rev: BTreeMap<Word<C>, Idx>,
  pub vocab: BTreeMap<Idx, Word<C>>,
  pub special_tokens: BTreeMap<String, Idx>,
  pub pre_tokenizer: PreTokenizer,
  pub merges: Vec<((Idx, Idx), Idx)>,
  /// with freq represents rank, or `merge.data.freq=-i` for i-th merge.
  /// with [`occurs_in={0}`](MergeData::occurs_in), in order to handle first word in [`Self::_encode_word`].
  pub pre_merge_map: HashMap<(Idx, Idx), Merge<C, Idx>>,
  pub cache: Cache<String, Word<Idx>>,
}

impl<C: Ord + Cachable> BpeEncoder<C>
where
  Word<C>: WordDebugExt,
  C: CanStrToWord,
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
    let pre_merge_map = merges.iter().copied().enumerate().map(|(i, (tp, target))| {
      let mut merge = Merge::new(tp, (
        vocab.get(&tp.0).ok_or_else(|| MyError::OovIdx(tp.0.to_u64())).cloned()?,
        vocab.get(&tp.1).ok_or_else(|| MyError::OovIdx(tp.1.to_u64())).cloned()?,
      )).with_target(target);
      merge.add(0, -(i as Freq));
      Ok((tp, merge))
    }).collect::<MyResult<_>>()?;
    let end_of_text = special_tokens.first().cloned();
    let pre_tokenizer = PreTokenizer::new(&special_tokens, end_of_text.as_deref());
    let special_tokens = special_tokens.into_iter().map(|s| {
      let w = s.to_word();
      let idx = *vocab_rev.get(&w).ok_or_else(|| MyError::Oov(w.debug_display()))?;
      Ok((s, idx))
    }).collect::<MyResult<_>>()?;
    let max_cap = vocab.len() as u64 * 500;
    Ok(Self {
      vocab_bytes,
      vocab_rev,
      vocab,
      merges,
      pre_merge_map,
      special_tokens,
      pre_tokenizer,
      cache: Cache::new(max_cap),
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
      let changes = _merge(&mut words, &merge, *target);
      _update_merge_map(&mut pre_merges, &merge, changes, None);
    }

    Ok(words.into_iter().map(|i| i.idxs.to_word()).collect())
  }

  // #[hotpath::measure]
  /// Encode many words, using an internal cache to avoid recomputing repeats.
  ///
  /// The output ordering matches the input iteration order.
  pub fn encode_words_impl<S: AsRef<str>, I: IntoIterator<Item = S>>(&self, input: I) -> MyResult<Vec<Word<Idx>>> {
    let mut results = BTreeMap::new();
    let mut to_encode = Vec::new();
    let mut query = Vec::new();
    let input_len = input.into_iter().enumerate().map(|(i, w)| {
      let w = w.as_ref();
      if let Some(cached) = self.cache.get(w) {
        results.insert(i, cached);
      } else {
        to_encode.push(w.to_word());
        query.push((i, w.to_string()));
      }
    }).count();
    let encoded = self._encode_words(&to_encode)?;
    for ((i, w), (_, e)) in query.into_iter().zip(to_encode.into_iter().zip(encoded.into_iter())) {
      self.cache.insert(w, e.clone());
      results.insert(i, e);
    }
    let final_results = results.values().cloned().collect::<Vec<_>>();
    assert_eq!(final_results.len(), input_len);
    Ok(final_results)
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
      let changes = _merge(&mut words, merge, merge.target.unwrap());
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

  // #[hotpath::measure]
  fn encode_tokens_index(&self, tokens_index: &HashMap<&str, Vec<usize>>, special_tokens_index: &HashMap<&str, Vec<usize>>) -> MyResult<Vec<Idx>> {
    let tokens_num = tokens_index.iter().map(|(_, doc_idxs)| doc_idxs.len()).sum::<usize>();
    let special_tokens_num = special_tokens_index.iter().map(|(_, doc_idxs)| doc_idxs.len()).sum::<usize>();
    let total = tokens_num + special_tokens_num;
    let mut result: Vec<&[Idx]> = vec![&[]; total];

    let output = self.encode_words_impl(tokens_index.keys())?;
    for (doc_idxs, w) in tokens_index.values().zip(output.iter()) {
      for doc_idx in doc_idxs.iter() {
        result[*doc_idx] = &w;
      }
    }

    let special_output = special_tokens_index.iter()
      .map(|(token, _)| {
        let idx = self.special_tokens.get(*token).ok_or_else(|| MyError::Oov(token.to_string()))?;
        Ok([*idx])
      })
      .collect::<MyResult<Vec<_>>>()?;
    for ((_token, doc_idxs), w) in special_tokens_index.iter().zip(special_output.iter()) {
      for doc_idx in doc_idxs.iter() {
        result[*doc_idx] = w.as_slice();
      }
    }

    let final_result = result.into_iter().map(|w| w.to_vec()).flatten().collect::<Vec<_>>();
    Ok(final_result)
  }

  // #[hotpath::measure]
  fn _create_cache_from_words(
    &self, input: Vec<String>
  ) -> MyResult<OrderMap<String, Arc<[Idx]>>> {
    let words = input.iter().map(|s| s.to_word()).collect::<Vec<_>>();
    let encoded = self._encode_words(&words)?;
    let cache = OrderMap::from_iter(input.into_iter().zip(encoded.into_iter()).rev().map(|(k, v)| (k, v)));
    Ok(cache)
  }

  /// Replace the internal encode cache with the provided entries.
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
    if let Some(result) = self.cache.get(input) {
      return Ok(result);
    }
    let result = self._encode_word(&input.to_word())?;
    self.cache.insert(input.to_string(), result.clone());
    Ok(result)
  }

  fn encode_words(&self, words: &[&str]) -> MyResult<Vec<Word<Idx>>> {
    self.encode_words_impl(words)
  }

  #[hotpath::measure]
  fn encode_string(&self, input: &str) -> MyResult<Vec<Idx>> {
    let (tokens_index, special_tokens_index) = self.pre_tokenizer.get_tokens_index_from_segment(input)?;
    self.encode_tokens_index(&tokens_index, &special_tokens_index)
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
    let words = self._decode(idxs)?;
    let mut result = Vec::new();
    for word in words {
      for c in word.iter() {
        c.char_split_u8(&mut result);
      }
    }
    Ok(String::from_utf8_lossy(&result).to_string())
  }
}

#[cfg(test)]
mod tests {
  use crate::{spec::{gpt2::Gpt2Spec, uni::UniSpec}, traits::CanEncode};

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
    println!("input size: {}, cache size: {}", input.len(), bpe.cache.weighted_size())
  }

  #[test]
  fn test_encode_string() {
    const NAME: &str = "tinystories_sample_5M";
    let bpe = _setup_bpe(NAME, &Gpt2Spec);
    let input = std::fs::read_to_string(format!("fixtures/{NAME}.txt")).unwrap();
    let result = bpe.encode_string(&input).unwrap();
    assert!(result.len() == 1424324);
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
    assert!(result.len() == 1424324);
  }

  #[test]
  fn test_bpe_encode_file_uni() {
    const NAME: &str = "TinyStories_all_data_zh_1M-sample";
    let bpe = _setup_bpe::<Character>(&format!("{NAME}.uni"), &UniSpec);
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
