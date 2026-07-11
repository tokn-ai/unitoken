use std::{collections::BTreeMap, hash::Hash};

use ahash::AHashMap;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
  MyError, MyResult,
  bpe::Freq,
  pretokenizer::{
    PreTokenizer, _pretokenizer_counter_with_unicode_bigrams, count_unicode_bigrams,
    is_unicode_bigram_script, select_unicode_bigrams, split_special_tokens,
  },
};

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
  for part in split_special_tokens(text, &pre_tokenizer.re_special_tokens)?
    .into_iter()
    .filter(|part| !part.is_special())
  {
    count_unicode_bigrams(part.as_str(), counts, is_unicode_bigram_script)?;
  }
  Ok(())
}

fn count_words_into(
  pre_tokenizer: &PreTokenizer,
  text: &str,
  counts: &mut AHashMap<String, Freq>,
) -> MyResult<()> {
  for part in split_special_tokens(text, &pre_tokenizer.re_special_tokens)?
    .into_iter()
    .filter(|part| !part.is_special())
  {
    for (word, frequency) in _pretokenizer_counter_with_unicode_bigrams(
      part.as_str(),
      &pre_tokenizer.re_pat,
      pre_tokenizer.unicode_bigrams.as_ref(),
      pre_tokenizer.unicode_bigram_mixed_boundary,
    )? {
      checked_add(counts, word, frequency)?;
    }
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
    let mut selected = select_unicode_bigrams(self.counts.clone(), top_k, min_freq)
      .into_iter()
      .collect::<Vec<_>>();
    selected.sort_unstable();
    selected
  }

  pub fn counts(&self) -> &AHashMap<(char, char), Freq> {
    &self.counts
  }
}

#[derive(Clone)]
#[cfg_attr(feature = "py", pyo3::pyclass(from_py_object))]
pub struct WordCounter {
  pre_tokenizer: PreTokenizer,
  counts: AHashMap<String, Freq>,
}

impl WordCounter {
  pub fn new(pre_tokenizer: PreTokenizer) -> Self {
    Self {
      pre_tokenizer,
      counts: AHashMap::new(),
    }
  }

  pub fn add_text(&mut self, text: &str) -> MyResult<()> {
    count_words_into(&self.pre_tokenizer, text, &mut self.counts)
  }

  pub fn add_batch<S: AsRef<str> + Sync>(&mut self, texts: &[S]) -> MyResult<()> {
    let batch_counts = texts
      .par_iter()
      .try_fold(AHashMap::new, |mut counts, text| {
        count_words_into(&self.pre_tokenizer, text.as_ref(), &mut counts)?;
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

  pub fn take_counts(&mut self) -> AHashMap<String, Freq> {
    std::mem::take(&mut self.counts)
  }

  pub fn counts(&self) -> &AHashMap<String, Freq> {
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
}
