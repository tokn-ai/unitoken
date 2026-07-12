use std::{collections::BTreeMap, hash::Hash, io::{Read, Write}};

use ahash::AHashMap;
use hashbrown::HashMap;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
  MyError, MyResult,
  bpe::Freq,
  pretokenizer::{
    PreTokenizer, count_unicode_bigrams, for_each_pretoken, for_each_regular_chunk,
    is_unicode_bigram_script, select_unicode_bigrams, UnicodeBigramSelection,
  },
};

type WordCounts = HashMap<String, Freq, ahash::RandomState>;
type BorrowedWordCounts<'a> = HashMap<&'a str, Freq, ahash::RandomState>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceBatchOptions {
  pub max_records: usize,
  pub max_bytes: usize,
}

impl Default for SourceBatchOptions {
  fn default() -> Self {
    Self {
      max_records: 4096,
      max_bytes: 64 * 1024 * 1024,
    }
  }
}

impl SourceBatchOptions {
  pub fn validate(self) -> MyResult<Self> {
    if self.max_records == 0 {
      return Err(MyError::SourceBatch("max_records must be at least 1"));
    }
    if self.max_bytes == 0 {
      return Err(MyError::SourceBatch("max_bytes must be at least 1"));
    }
    Ok(self)
  }
}

fn checked_add<K: Eq + Hash>(counts: &mut AHashMap<K, Freq>, key: K, value: Freq) -> MyResult<()> {
  let current = counts.entry(key).or_default();
  *current = current.checked_add(value).ok_or(MyError::FrequencyOverflow)?;
  Ok(())
}

fn merge_counts<K: Eq + Hash>(target: &mut AHashMap<K, Freq>, source: AHashMap<K, Freq>) -> MyResult<()> {
  for (key, value) in source {
    checked_add(target, key, value)?;
  }
  Ok(())
}

fn count_bigrams_into(
  pre_tokenizer: &PreTokenizer,
  text: &str,
  counts: &mut AHashMap<(char, char), Freq>,
) -> MyResult<()> {
  if text.is_empty() {
    return Ok(());
  }
  if pre_tokenizer.re_special_tokens.as_str() == "$^" {
    return count_unicode_bigrams(text, counts, is_unicode_bigram_script);
  }
  for_each_regular_chunk(text, &pre_tokenizer.re_special_tokens, |chunk| {
    count_unicode_bigrams(chunk, counts, is_unicode_bigram_script)
  })
}

fn count_words_borrowed<'a>(
  pre_tokenizer: &PreTokenizer,
  text: &'a str,
  counts: &mut BorrowedWordCounts<'a>,
) -> MyResult<()> {
  if text.is_empty() {
    return Ok(());
  }
  if pre_tokenizer.re_special_tokens.as_str() == "$^" {
    return count_words_in_regular_text_borrowed(pre_tokenizer, text, counts);
  }
  for_each_regular_chunk(text, &pre_tokenizer.re_special_tokens, |chunk| {
    count_words_in_regular_text_borrowed(pre_tokenizer, chunk, counts)
  })
}

fn count_words_in_regular_text_borrowed<'a>(
  pre_tokenizer: &PreTokenizer,
  text: &'a str,
  counts: &mut BorrowedWordCounts<'a>,
) -> MyResult<()> {
  for_each_pretoken(
    text,
    &pre_tokenizer.re_pat,
    pre_tokenizer.unicode_bigrams.as_ref(),
    pre_tokenizer.unicode_bigram_mixed_boundary,
    |word| {
      let frequency = counts.entry(word).or_insert(0);
      *frequency = frequency.checked_add(1).ok_or(MyError::FrequencyOverflow)?;
      Ok(())
    },
  )
}

fn merge_borrowed_word_counts<'a>(
  mut left: BorrowedWordCounts<'a>,
  mut right: BorrowedWordCounts<'a>,
) -> MyResult<BorrowedWordCounts<'a>> {
  if left.len() < right.len() {
    std::mem::swap(&mut left, &mut right);
  }
  for (word, frequency) in right {
    let current = left.entry(word).or_default();
    *current = current.checked_add(frequency).ok_or(MyError::FrequencyOverflow)?;
  }
  Ok(left)
}

fn merge_borrowed_into_word_counts(
  target: &mut WordCounts,
  source: BorrowedWordCounts<'_>,
) -> MyResult<()> {
  target.reserve(source.len());
  for (word, frequency) in source {
    let current = target.entry_ref(word).or_insert(0);
    *current = current.checked_add(frequency).ok_or(MyError::FrequencyOverflow)?;
  }
  Ok(())
}

fn merge_word_counts(target: &mut WordCounts, source: WordCounts) -> MyResult<()> {
  for (word, frequency) in source {
    let current = target.entry(word).or_default();
    *current = current.checked_add(frequency).ok_or(MyError::FrequencyOverflow)?;
  }
  Ok(())
}

#[derive(Clone)]
#[cfg_attr(feature = "py", pyo3::pyclass(from_py_object))]
pub struct BigramCounter {
  pre_tokenizer: PreTokenizer,
  counts: AHashMap<(char, char), Freq>,
}

impl BigramCounter {
  pub fn new(pre_tokenizer: PreTokenizer) -> Self {
    Self {
      pre_tokenizer,
      counts: AHashMap::new(),
    }
  }

  pub fn add_text(&mut self, text: &str) -> MyResult<()> {
    count_bigrams_into(&self.pre_tokenizer, text, &mut self.counts)
  }

  pub fn add_batch<S: AsRef<str> + Sync>(&mut self, texts: &[S]) -> MyResult<()> {
    let batch_counts = texts
      .par_iter()
      .try_fold(AHashMap::new, |mut counts, text| {
        count_bigrams_into(&self.pre_tokenizer, text.as_ref(), &mut counts)?;
        Ok::<_, MyError>(counts)
      })
      .try_reduce(AHashMap::new, |mut left, right| {
        merge_counts(&mut left, right)?;
        Ok::<_, MyError>(left)
      })?;
    merge_counts(&mut self.counts, batch_counts)
  }

  pub fn add_source<I, S>(&mut self, source: I, options: SourceBatchOptions) -> MyResult<()>
  where
    I: IntoIterator<Item = S>,
    S: AsRef<str> + Sync,
  {
    let options = options.validate()?;
    let mut batch = Vec::new();
    let mut bytes: usize = 0;
    for text in source {
      let text_bytes = text.as_ref().len();
      if !batch.is_empty()
        && (batch.len() >= options.max_records
          || bytes.saturating_add(text_bytes) > options.max_bytes)
      {
        self.add_batch(&batch)?;
        batch.clear();
        bytes = 0;
      }
      bytes = bytes.checked_add(text_bytes).ok_or(MyError::SourceBatch("batch byte size overflow"))?;
      batch.push(text);
    }
    if !batch.is_empty() {
      self.add_batch(&batch)?;
    }
    Ok(())
  }

  pub fn merge(&mut self, other: Self) -> MyResult<()> {
    merge_counts(&mut self.counts, other.counts)
  }

  pub fn selected(&self, top_k: usize, min_freq: Freq) -> Vec<(char, char)> {
    let mut selected = self
      .selection(top_k, min_freq)
      .bigrams
      .into_iter()
      .collect::<Vec<_>>();
    selected.sort_unstable();
    selected
  }

  /// Select Unicode bigrams and preserve their effective frequency boundary.
  pub fn selection(&self, top_k: usize, min_freq: Freq) -> UnicodeBigramSelection {
    select_unicode_bigrams(self.counts.clone(), top_k, min_freq)
  }

  pub fn counts(&self) -> &AHashMap<(char, char), Freq> {
    &self.counts
  }
}

#[derive(Clone)]
#[cfg_attr(feature = "py", pyo3::pyclass(from_py_object))]
pub struct WordCounter {
  pre_tokenizer: PreTokenizer,
  counts: WordCounts,
}

impl WordCounter {
  pub fn new(pre_tokenizer: PreTokenizer) -> Self {
    Self {
      pre_tokenizer,
      counts: WordCounts::default(),
    }
  }

  pub fn add_text(&mut self, text: &str) -> MyResult<()> {
    let mut counts = BorrowedWordCounts::default();
    count_words_borrowed(&self.pre_tokenizer, text, &mut counts)?;
    merge_borrowed_into_word_counts(&mut self.counts, counts)
  }

  pub fn add_batch<'a, S: AsRef<str> + Sync>(&mut self, texts: &'a [S]) -> MyResult<()> {
    let batch_counts = texts
      .par_iter()
      .try_fold(BorrowedWordCounts::default, |mut counts, text| {
        count_words_borrowed(&self.pre_tokenizer, text.as_ref(), &mut counts)?;
        Ok::<_, MyError>(counts)
      })
      .try_reduce(BorrowedWordCounts::default, merge_borrowed_word_counts)?;
    merge_borrowed_into_word_counts(&mut self.counts, batch_counts)
  }

  pub fn add_source<I, S>(&mut self, source: I, options: SourceBatchOptions) -> MyResult<()>
  where
    I: IntoIterator<Item = S>,
    S: AsRef<str> + Sync,
  {
    let options = options.validate()?;
    let mut batch = Vec::new();
    let mut bytes: usize = 0;
    for text in source {
      let text_bytes = text.as_ref().len();
      if !batch.is_empty()
        && (batch.len() >= options.max_records
          || bytes.saturating_add(text_bytes) > options.max_bytes)
      {
        self.add_batch(&batch)?;
        batch.clear();
        bytes = 0;
      }
      bytes = bytes.checked_add(text_bytes).ok_or(MyError::SourceBatch("batch byte size overflow"))?;
      batch.push(text);
    }
    if !batch.is_empty() {
      self.add_batch(&batch)?;
    }
    Ok(())
  }

  pub fn merge(&mut self, other: Self) -> MyResult<()> {
    merge_word_counts(&mut self.counts, other.counts)
  }

  pub fn words(&self) -> BTreeMap<String, Freq> {
    self.counts.iter().map(|(word, frequency)| (word.clone(), *frequency)).collect()
  }

  pub fn len(&self) -> usize {
    self.counts.len()
  }

  pub fn is_empty(&self) -> bool {
    self.counts.is_empty()
  }

  pub fn clear(&mut self) {
    self.counts.clear();
  }

  pub fn take_counts(&mut self) -> WordCounts {
    std::mem::take(&mut self.counts)
  }

  pub fn save<W: Write>(&self, writer: W) -> MyResult<()> {
    serde_json::to_writer(writer, &self.counts)?;
    Ok(())
  }

  pub fn load<R: Read>(pre_tokenizer: PreTokenizer, reader: R) -> MyResult<Self> {
    let counts = serde_json::from_reader(reader)?;
    Ok(Self { pre_tokenizer, counts })
  }

  pub fn counts(&self) -> &WordCounts {
    &self.counts
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::pretokenizer::parse_unicode_bigrams;

  #[test]
  fn bigram_counter_batches_and_merges() {
    let pre_tokenizer = PreTokenizer::new(&[], None);
    let mut left = BigramCounter::new(pre_tokenizer.clone());
    left.add_source(
      ["你好世界", "你好"],
      SourceBatchOptions { max_records: 1, max_bytes: 8 },
    ).unwrap();
    let mut right = BigramCounter::new(pre_tokenizer);
    right.add_text("世界").unwrap();
    left.merge(right).unwrap();

    assert_eq!(left.counts().get(&('你', '好')), Some(&2));
    assert_eq!(left.counts().get(&('世', '界')), Some(&2));
    let selection = left.selection(1, 1);
    assert_eq!(selection.cutoff_freq, Some(2));
    assert_eq!(selection.max_excluded_freq, Some(1));
    assert_eq!(selection.bigrams.len(), 2);
  }

  #[test]
  fn word_counter_uses_frozen_bigrams_and_skips_special_tokens() {
    let bigrams = parse_unicode_bigrams(&["你好".to_string()]).unwrap();
    let pre_tokenizer = PreTokenizer::new(&["<eot>".to_string()], Some("<eot>"))
      .with_unicode_bigrams(bigrams);
    let mut counter = WordCounter::new(pre_tokenizer);
    counter.add_batch(&["你好世界<eot>", "你好"]).unwrap();

    let words = counter.words();
    assert_eq!(words.get("你好"), Some(&2));
    assert_eq!(words.get("世"), Some(&1));
    assert_eq!(words.get("界"), Some(&1));
    assert!(!words.contains_key("<eot>"));
  }

  #[test]
  fn word_counter_borrowed_batch_matches_owned_pretokenization() {
    let bigrams = parse_unicode_bigrams(&["世界".to_string(), "你好".to_string()]).unwrap();
    let pre_tokenizer = PreTokenizer::new(&[], None).with_unicode_bigrams(bigrams);
    let texts = ["Hello 世界你好 world", "世界你好", "Hello"];
    let mut expected = BTreeMap::new();
    for text in texts {
      for (word, frequency) in pre_tokenizer.get_words_owned(text).unwrap() {
        *expected.entry(word).or_default() += frequency;
      }
    }

    let mut counter = WordCounter::new(pre_tokenizer);
    counter.add_batch(&texts).unwrap();

    assert_eq!(counter.words(), expected);
  }

  #[test]
  fn word_counter_streams_around_adjacent_special_tokens() {
    let pre_tokenizer = PreTokenizer::new(&["<eot>".to_string()], Some("<eot>"));
    let mut counter = WordCounter::new(pre_tokenizer);
    counter.add_text("<eot>Hello<eot><eot> world<eot>").unwrap();

    assert_eq!(
      counter.words(),
      [("Hello".to_string(), 1), (" world".to_string(), 1)].into_iter().collect(),
    );
  }

  #[test]
  fn word_counter_preserves_empty_custom_pattern_matches() {
    let pre_tokenizer = PreTokenizer::try_new(&[], None, Some("")).unwrap()
      .with_unicode_bigrams(ahash::AHashSet::new());
    let expected = pre_tokenizer.get_words_owned("ab").unwrap();
    let mut counter = WordCounter::new(pre_tokenizer);
    counter.add_text("ab").unwrap();

    assert_eq!(counter.words(), expected);
  }
}
