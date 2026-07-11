#[pyo3::pymodule(gil_used = false)]
mod _lib {
use std::{collections::BTreeMap, path::PathBuf, sync::{Arc, mpsc::sync_channel}};

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use ordermap::OrderMap;
use pyo3::{prelude::*, pymethods, types::{PyAny, PyIterator}};

use crate::{MyError, MyResult, bpe::{BpeEncoder, BpeTrainer, BpeTrainerConfig, CharIdx, CharSplit, Character, Idx, IdxLike, InitialAlphabet, TieBreak, Word, encoder::BpeBuilder, utils::ToWord}, counter::SourceBatchOptions, pretokenizer::{BoundaryMode, ChunkHint, ChunkOptions, UnicodeBigramMixedBoundary, parse_unicode_bigrams, unicode_bigram_to_string}, spec::{Spec, gpt2::Gpt2Spec, unitoken::UnitokenSpec}, traits::{CanEncode, CanStrToWord, Encoder, Train as _}};

#[pyclass(subclass)]
pub struct BpeTrainerBase;

#[allow(dead_code)]
/// this is just a reference for impl blocks, not directly used
pub trait BpeTrainerBaseImpl: Sized {
  fn new_py(special_tokens: Vec<String>) -> (Self, BpeTrainerBase);

  fn add_words(&mut self, py: Python, words: Vec<(String, i64)>);
  fn vocab_size(&self) -> usize;
  fn init_training(&mut self, py: Python);
  fn train_until(&mut self, py: Python, vocab_size: usize) -> PyResult<i64>;
  fn step(&mut self, py: Python) -> PyResult<i64>;
  fn get_vocab(&self) -> Vocabulary;
  fn validate_model(&self, py: Python) -> PyResult<()>;
  fn save_vocab(&self, py: Python, path: PathBuf, format: &str) -> PyResult<()>;
  fn save_merges_txt(&self, py: Python, path: PathBuf, format: &str) -> PyResult<()>;
}

fn trainer_config(
  initial_alphabet: Option<&str>,
  tie_break: Option<&str>,
  parallel_merge_min_occurs_in: Option<usize>,
) -> PyResult<BpeTrainerConfig> {
  let initial_alphabet = match initial_alphabet.unwrap_or("raw") {
    "raw" => InitialAlphabet::RawBytes,
    "byte_level" => InitialAlphabet::ByteLevel,
    value => return Err(pyo3::exceptions::PyValueError::new_err(format!("Unknown initial_alphabet: {value}"))),
  };
  let tie_break = match tie_break.unwrap_or("smallest_pair_id") {
    "smallest_pair_id" => TieBreak::SmallestPairId,
    "largest_content" => TieBreak::LargestContent,
    value => return Err(pyo3::exceptions::PyValueError::new_err(format!("Unknown tie_break: {value}"))),
  };
  Ok(BpeTrainerConfig {
    initial_alphabet,
    tie_break,
    parallel_merge_min_occurs_in,
  })
}

fn chunk_options(chunk_size: u64, boundary: &str) -> PyResult<ChunkOptions> {
  let boundary = BoundaryMode::parse(boundary)
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
  if chunk_size == 0 {
    return Err(pyo3::exceptions::PyValueError::new_err("chunk_size must be at least 1"));
  }
  Ok(ChunkOptions {
    hint: ChunkHint::Size(chunk_size),
    boundary,
  })
}

#[allow(non_camel_case_types)]
#[pyclass(extends = BpeTrainerBase)]
pub struct BpeTrainer_u8_Idx {
  pub inner: BpeTrainer<u8, Idx>,
}

#[pymethods]
impl BpeTrainer_u8_Idx {
  #[new]
  /// Create a new BPE trainer (byte-level) for Python.
  ///
  /// Returns `(trainer, base)` where `base` enables Python-side subclassing.
  #[pyo3(signature = (special_tokens, initial_alphabet=None, tie_break=None, parallel_merge_min_occurs_in=None))]
  pub fn new_py(
    special_tokens: Vec<String>,
    initial_alphabet: Option<&str>,
    tie_break: Option<&str>,
    parallel_merge_min_occurs_in: Option<usize>,
  ) -> PyResult<(Self, BpeTrainerBase)> {
    let config = trainer_config(initial_alphabet, tie_break, parallel_merge_min_occurs_in)?;
    Ok((
      Self {
        inner: BpeTrainer::new_with_config(vec![], special_tokens, config),
      },
      BpeTrainerBase {},
    ))
  }

  /// Add `(word, frequency)` pairs to the trainer's inventory.
  pub fn add_words(&mut self, py: Python, words: Vec<(String, i64)>) {
    py.detach(||
      self.inner.add_words(&mut words.iter().map(|(w, f)| (w.as_str(), *f)))
    )
  }

  /// Replace the trainer inventory by consuming a word counter.
  pub fn add_word_counter(&mut self, py: Python, mut counter: PyRefMut<'_, WordCounter>) {
    let counts = counter.take_counts();
    drop(counter);
    py.detach(||
      self.inner.add_words(&mut counts.iter().map(|(word, frequency)| (word.as_str(), *frequency)))
    )
  }

  /// Current vocabulary size.
  pub fn vocab_size(&self) -> usize {
    self.inner.vocab_size()
  }

  /// Initialize internal training state.
  pub fn init_training(&mut self, py: Python) {
    py.detach(|| self.inner.init_training())
  }

  /// Train until the vocabulary reaches `vocab_size`.
  ///
  /// Returns the updated vocabulary size.
  pub fn train_until(&mut self, py: Python, vocab_size: usize) -> PyResult<i64> {
    py.detach(|| self.inner.train_until(vocab_size)).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(self.inner.vocab_size() as i64)
  }

  /// Perform one training step.
  ///
  /// Returns the updated vocabulary size.
  pub fn step(&mut self, py: Python) -> PyResult<i64> {
    py.detach(|| self.inner.step()).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(self.inner.vocab_size() as i64)
  }

  /// Return a view of the current vocabulary.
  pub fn get_vocab(&self) -> Vocabulary {
    Vocabulary {
      inner: Box::new(VocabularyInner::new(&self.inner.vocab)),
    }
  }

  /// Validate the current vocabulary and merge history.
  pub fn validate_model(&self, py: Python) -> PyResult<()> {
    py.detach(|| self.inner.validate_model())
      .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
  }

  /// Save the vocabulary JSON using the requested format.
  pub fn save_vocab(&self, py: Python, path: PathBuf, format: &str) -> PyResult<()> {
    py.detach(|| {
      let mut file = std::fs::File::create(&path)?;
      let mut writer = std::io::BufWriter::new(&mut file);
      match format {
        "gpt2" => self.inner.save_vocab_json(&Gpt2Spec, &mut writer),
        "unitoken" => self.inner.save_vocab_json(&UnitokenSpec, &mut writer),
        _ => Err(MyError::SpecError(format!("Unknown format: {}", format))),
      }
    }).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
  }

  /// Save merges using the requested format.
  pub fn save_merges_txt(&self, py: Python, path: PathBuf, format: &str) -> PyResult<()> {
    py.detach(|| {
      let mut file = std::fs::File::create(&path)?;
      let mut writer = std::io::BufWriter::new(&mut file);
      match format {
        "gpt2" => self.inner.save_merges_txt(&Gpt2Spec, &mut writer),
        "unitoken" => self.inner.save_merges_txt(&UnitokenSpec, &mut writer),
        _ => Err(MyError::SpecError(format!("Unknown format: {}", format))),
      }
    }).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
  }
}

#[allow(non_camel_case_types)]
#[pyclass(extends = BpeTrainerBase)]
pub struct BpeTrainer_Character_CharIdx {
  pub inner: BpeTrainer<Character, CharIdx>,
}

#[pymethods]
impl BpeTrainer_Character_CharIdx {
  #[new]
  /// Create a new BPE trainer (character-level) for Python.
  ///
  /// Returns `(trainer, base)` where `base` enables Python-side subclassing.
  #[pyo3(signature = (special_tokens, initial_alphabet=None, tie_break=None, parallel_merge_min_occurs_in=None))]
  pub fn new_py(
    special_tokens: Vec<String>,
    initial_alphabet: Option<&str>,
    tie_break: Option<&str>,
    parallel_merge_min_occurs_in: Option<usize>,
  ) -> PyResult<(Self, BpeTrainerBase)> {
    let config = trainer_config(initial_alphabet, tie_break, parallel_merge_min_occurs_in)?;
    Ok((
      Self {
        inner: BpeTrainer::new_with_config(vec![], special_tokens, config),
      },
      BpeTrainerBase {},
    ))
  }

  /// Add `(word, frequency)` pairs to the trainer's inventory.
  pub fn add_words(&mut self, py: Python, words: Vec<(String, i64)>) {
    py.detach(||
      self.inner.add_words(&mut words.iter().map(|(w, f)| (w.as_str(), *f)))
    )
  }

  /// Replace the trainer inventory by consuming a word counter.
  pub fn add_word_counter(&mut self, py: Python, mut counter: PyRefMut<'_, WordCounter>) {
    let counts = counter.take_counts();
    drop(counter);
    py.detach(||
      self.inner.add_words(&mut counts.iter().map(|(word, frequency)| (word.as_str(), *frequency)))
    )
  }

  /// Current vocabulary size.
  pub fn vocab_size(&self) -> usize {
    self.inner.vocab_size()
  }

  /// Initialize internal training state.
  pub fn init_training(&mut self, py: Python) {
    py.detach(|| self.inner.init_training())
  }

  /// Train until the vocabulary reaches `vocab_size`.
  ///
  /// Returns the updated vocabulary size.
  pub fn train_until(&mut self, py: Python, vocab_size: usize) -> PyResult<i64> {
    py.detach(|| self.inner.train_until(vocab_size)).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(self.inner.vocab_size() as i64)
  }

  /// Perform one training step.
  ///
  /// Returns the updated vocabulary size.
  pub fn step(&mut self, py: Python) -> PyResult<i64> {
    py.detach(|| self.inner.step()).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(self.inner.vocab_size() as i64)
  }

  /// Return a view of the current vocabulary.
  pub fn get_vocab(&self) -> Vocabulary {
    Vocabulary {
      inner: Box::new(VocabularyInner::new(&self.inner.vocab)),
    }
  }

  /// Validate the current vocabulary and merge history.
  pub fn validate_model(&self, py: Python) -> PyResult<()> {
    py.detach(|| self.inner.validate_model())
      .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
  }

  /// Save the vocabulary JSON to `path`.
  ///
  /// Note: `"gpt2"` is not supported for the character tokenizer.
  pub fn save_vocab(&self, py: Python, path: PathBuf, format: &str) -> PyResult<()> {
    py.detach(|| {
      let mut file = std::fs::File::create(&path)?;
      let mut writer = std::io::BufWriter::new(&mut file);
      match format {
        "gpt2" => Err(MyError::SpecError("gpt2 format is not supported for the Unicode unit".to_string())),
        "unitoken" => self.inner.save_vocab_json(&UnitokenSpec, &mut writer),
        _ => Err(MyError::SpecError(format!("Unknown format: {}", format))),
      }
    }).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
  }

  /// Save merges to `path`.
  ///
  /// Note: `"gpt2"` is not supported for the character tokenizer.
  pub fn save_merges_txt(&self, py: Python, path: PathBuf, format: &str) -> PyResult<()> {
    py.detach(|| {
      let mut file = std::fs::File::create(&path)?;
      let mut writer = std::io::BufWriter::new(&mut file);
      match format {
        "gpt2" => Err(MyError::SpecError("gpt2 format is not supported for the Unicode unit".to_string())),
        "unitoken" => self.inner.save_merges_txt(&UnitokenSpec, &mut writer),
        _ => Err(MyError::SpecError(format!("Unknown format: {}", format))),
      }
    }).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
  }
}

pub struct VocabularyInner<C, I>(OrderMap<Word<C>, I>);

impl<C: std::hash::Hash + Eq, I: IdxLike> VocabularyInner<C, I> {
  /// Build a reverse map from token bytes to token id.
  pub fn new(vocab: &BTreeMap<I, Word<C>>) -> Self {
    Self(vocab.iter().map(|(i, c)| (c.clone(), i.clone())).collect())
  }
}

trait VocabularyImpl {
  fn len(&self) -> usize;
  fn get(&self, word: &str) -> Option<i64>;
  fn items(&self) -> Vec<(Vec<u8>, i64)>;
}

impl<C: CanStrToWord + CharSplit + std::hash::Hash + Eq, I: IdxLike> VocabularyImpl for VocabularyInner<C, I> {
  fn len(&self) -> usize {
    self.0.len()
  }

  fn get(&self, word: &str) -> Option<i64> {
    self.0.get(&word.to_word()).map(|i| i.to_u64() as i64)
  }

  fn items(&self) -> Vec<(Vec<u8>, i64)> {
    self.0.iter().map(|(w, i)| (CharSplit::to_vec_u8(w), i.to_u64() as i64)).collect()
  }
}

#[pyclass]
pub struct Vocabulary {
  inner: Box<dyn VocabularyImpl + Send + Sync>,
}

#[pymethods]
impl Vocabulary {
  #[getter]
  /// Number of entries in the vocabulary.
  pub fn len(&self) -> usize {
    self.inner.len()
  }

  /// Look up a word/token and return its id if present.
  pub fn get(&self, word: &str) -> Option<i64> {
    self.inner.get(word)
  }

  /// Return all `(token_bytes, id)` entries.
  pub fn items(&self) -> Vec<(Vec<u8>, i64)> {
    self.inner.items()
  }
}

#[pymodule_export]
pub use crate::pretokenizer::PreTokenizer;
#[pymodule_export]
pub use crate::counter::{BigramCounter, WordCounter};

#[pymethods]
impl PreTokenizer {
  #[new]
  /// Create a Python `PreTokenizer`.
  ///
  /// - `special_tokens`: special tokens to treat as indivisible.
  /// - `eot_token`: end-of-text token used for chunk boundary alignment.
  /// - `pat_str`: optional regex pattern; defaults to the crate's default.
  #[pyo3(signature = (special_tokens, eot_token=None, pat_str=None, unicode_bigrams=None, unicode_bigram_mixed_boundary="keep"))]
  pub fn new_py(
    special_tokens: Vec<String>,
    eot_token: Option<String>,
    pat_str: Option<String>,
    unicode_bigrams: Option<Vec<String>>,
    unicode_bigram_mixed_boundary: &str,
  ) -> PyResult<Self> {
    let mut pretokenizer = Self::try_new(&special_tokens, eot_token.as_deref(), pat_str.as_deref())
      .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let unicode_bigram_mixed_boundary = UnicodeBigramMixedBoundary::parse(unicode_bigram_mixed_boundary)
      .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    if let Some(unicode_bigrams) = unicode_bigrams {
      pretokenizer = pretokenizer.with_unicode_bigrams(
        parse_unicode_bigrams(&unicode_bigrams)
          .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
      );
    }
    pretokenizer = pretokenizer.with_unicode_bigram_mixed_boundary(unicode_bigram_mixed_boundary);
    Ok(pretokenizer)
  }

  #[pyo3(name = "find_chunk_boundaries", signature = (path, *, chunk_size=1048576, boundary="auto"))]
  /// Python wrapper for [`PreTokenizer::find_chunk_boundaries`].
  pub fn py_find_chunk_boundaries(
    &self, py: Python, path: PathBuf, chunk_size: u64, boundary: &str,
  ) -> PyResult<Vec<(u64, usize)>> {
    let options = chunk_options(chunk_size, boundary)?;
    py.detach(||
      self.find_chunk_boundaries_with_options(path, options)
    ).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
  }

  #[pyo3(name = "get_words_from_segment")]
  /// Python wrapper for [`PreTokenizer::get_words_from_segment`].
  pub fn py_get_words_from_segment(
    &self, py: Python, path: PathBuf, offset: u64, length: usize,
  ) -> PyResult<BTreeMap<String, i64>> {
    py.detach(||
      self.get_words_from_segment(path, offset, length)
    ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
  }

  #[pyo3(name = "get_words")]
  /// Pretokenize text and return a word-frequency mapping.
  pub fn py_get_words(&self, py: Python, text: &str) -> PyResult<BTreeMap<String, i64>> {
    py.detach(|| self.get_words_owned(text))
      .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
  }

  /// Create an empty mergeable Unicode bigram counter.
  pub fn bigram_counter(&self) -> BigramCounter {
    BigramCounter::new(self.clone())
  }

  /// Create an empty mergeable word counter.
  pub fn word_counter(&self) -> WordCounter {
    WordCounter::new(self.clone())
  }

  pub fn load_word_counter(&self, py: Python, path: PathBuf) -> PyResult<WordCounter> {
    let pre_tokenizer = self.clone();
    py.detach(|| {
      let file = std::fs::File::open(path)?;
      WordCounter::load(pre_tokenizer, std::io::BufReader::new(file))
    }).map_err(|error| pyo3::exceptions::PyIOError::new_err(error.to_string()))
  }

  #[pyo3(name = "get_words_from_file", signature = (path, *, chunk_size=1048576, boundary="auto"))]
  /// Python wrapper for [`PreTokenizer::get_words_from_file`].
  pub fn py_get_words_from_file(
    &self, py: Python, path: PathBuf, chunk_size: u64, boundary: &str,
  ) -> PyResult<BTreeMap<String, i64>> {
    let options = chunk_options(chunk_size, boundary)?;
    py.detach(||
      self.get_words_from_file_with_options(path, options)
    ).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
  }

  #[pyo3(name = "build_unicode_bigrams_from_file", signature = (path, *, chunk_size=1048576, boundary="auto", top_k=100000, min_freq=16))]
  pub fn py_build_unicode_bigrams_from_file(
    &self, py: Python, path: PathBuf, chunk_size: u64, boundary: &str, top_k: usize, min_freq: i64,
  ) -> PyResult<Vec<String>> {
    let options = chunk_options(chunk_size, boundary)?;
    let bigrams = py.detach(||
      self.build_unicode_bigram_set_from_file_with_options(path, options, top_k, min_freq)
    ).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let mut bigrams = bigrams.into_iter().collect::<Vec<_>>();
    bigrams.sort();
    Ok(bigrams.into_iter().map(unicode_bigram_to_string).collect())
  }
}

fn source_options(max_records: usize, max_bytes: usize) -> PyResult<SourceBatchOptions> {
  SourceBatchOptions { max_records, max_bytes }
    .validate()
    .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

fn should_flush_batch(batch_len: usize, batch_bytes: usize, next_bytes: usize, options: SourceBatchOptions) -> bool {
  batch_len > 0 && (batch_len >= options.max_records || batch_bytes.saturating_add(next_bytes) > options.max_bytes)
}

trait SourceCounter: Send {
  fn add_source_batch(&mut self, texts: &[String]) -> MyResult<()>;
}

impl SourceCounter for BigramCounter {
  fn add_source_batch(&mut self, texts: &[String]) -> MyResult<()> {
    self.add_batch(texts)
  }
}

impl SourceCounter for WordCounter {
  fn add_source_batch(&mut self, texts: &[String]) -> MyResult<()> {
    self.add_batch(texts)
  }
}

fn source_counter_error(error: MyError) -> PyErr {
  pyo3::exceptions::PyRuntimeError::new_err(error.to_string())
}

fn validate_prefetch(prefetch: i64) -> PyResult<usize> {
  if !(0..=1).contains(&prefetch) {
    return Err(pyo3::exceptions::PyValueError::new_err("prefetch must be 0 or 1"));
  }
  Ok(prefetch as usize)
}

fn for_each_source_batch(
  iterator: Bound<PyIterator>,
  options: SourceBatchOptions,
  mut consume: impl FnMut(Vec<String>) -> PyResult<()>,
) -> PyResult<()> {
  let mut batch = Vec::new();
  let mut batch_bytes = 0usize;
  for item in iterator {
    let text = item?.extract::<String>()?;
    if should_flush_batch(batch.len(), batch_bytes, text.len(), options) {
      consume(std::mem::take(&mut batch))?;
      batch_bytes = 0;
    }
    batch_bytes = batch_bytes.checked_add(text.len())
      .ok_or_else(|| pyo3::exceptions::PyOverflowError::new_err("batch byte size overflow"))?;
    batch.push(text);
  }
  if !batch.is_empty() {
    consume(batch)?;
  }
  Ok(())
}

fn add_source_sync<C: SourceCounter>(
  counter: &mut C,
  py: Python,
  source: &Bound<PyAny>,
  options: SourceBatchOptions,
) -> PyResult<()> {
  for_each_source_batch(source.try_iter()?, options, |batch| {
    py.detach(|| counter.add_source_batch(&batch)).map_err(source_counter_error)
  })
}

fn add_source_prefetched<C: SourceCounter>(
  counter: &mut C,
  py: Python,
  source: &Bound<PyAny>,
  options: SourceBatchOptions,
) -> PyResult<()> {
  let iterator = source.try_iter()?;
  std::thread::scope(|scope| {
    let (sender, receiver) = sync_channel::<Vec<String>>(0);
    let worker = scope.spawn(move || -> MyResult<()> {
      for batch in receiver {
        counter.add_source_batch(&batch)?;
      }
      Ok(())
    });

    let producer_result = for_each_source_batch(iterator, options, |batch| {
      py.detach(|| sender.send(batch))
        .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("source counter worker stopped"))
    });
    drop(sender);

    match py.detach(|| worker.join()) {
      Ok(Ok(())) => producer_result,
      Ok(Err(error)) => Err(source_counter_error(error)),
      Err(_) => Err(pyo3::exceptions::PyRuntimeError::new_err("source counter worker panicked")),
    }
  })
}

#[cfg(test)]
fn run_prefetched_batches<C: SourceCounter, E>(
  counter: &mut C,
  produce: impl FnOnce(&std::sync::mpsc::SyncSender<Vec<String>>) -> Result<(), E>,
) -> (Result<(), E>, std::thread::Result<MyResult<()>>) {
  std::thread::scope(|scope| {
    let (sender, receiver) = sync_channel::<Vec<String>>(0);
    let worker = scope.spawn(move || -> MyResult<()> {
      for batch in receiver {
        counter.add_source_batch(&batch)?;
      }
      Ok(())
    });

    let producer_result = produce(&sender);
    drop(sender);
    (producer_result, worker.join())
  })
}

fn add_python_source<C: SourceCounter>(
  counter: &mut C,
  py: Python,
  source: &Bound<PyAny>,
  max_records: usize,
  max_bytes: usize,
  prefetch: i64,
) -> PyResult<()> {
  let options = source_options(max_records, max_bytes)?;
  let prefetch = validate_prefetch(prefetch)?;
  match prefetch {
    0 => add_source_sync(counter, py, source, options),
    1 => add_source_prefetched(counter, py, source, options),
    _ => unreachable!(),
  }
}

#[cfg(test)]
mod source_prefetch_tests {
  use std::{sync::mpsc::sync_channel, time::Duration};

  use super::*;

  struct BlockingCounter {
    processing_started: std::sync::mpsc::SyncSender<()>,
    next_batch_started: std::sync::mpsc::Receiver<()>,
    calls: usize,
  }

  impl SourceCounter for BlockingCounter {
    fn add_source_batch(&mut self, _texts: &[String]) -> MyResult<()> {
      if self.calls == 0 {
        self.processing_started.send(()).unwrap();
        self.next_batch_started.recv_timeout(Duration::from_secs(1)).unwrap();
      }
      self.calls += 1;
      Ok(())
    }
  }

  #[test]
  fn prefetched_batches_overlap_production_and_processing() {
    let (processing_started_tx, processing_started_rx) = sync_channel(0);
    let (next_batch_started_tx, next_batch_started_rx) = sync_channel(0);
    let mut counter = BlockingCounter {
      processing_started: processing_started_tx,
      next_batch_started: next_batch_started_rx,
      calls: 0,
    };

    let (producer, worker) = run_prefetched_batches(&mut counter, |sender| {
      sender.send(vec!["first".to_string()]).unwrap();
      processing_started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
      next_batch_started_tx.send(()).unwrap();
      sender.send(vec!["second".to_string()]).unwrap();
      Ok::<_, ()>(())
    });

    assert_eq!(producer, Ok(()));
    assert!(worker.unwrap().is_ok());
    assert_eq!(counter.calls, 2);
  }
}

#[pymethods]
impl BigramCounter {
  #[new]
  pub fn new_py(pre_tokenizer: PreTokenizer) -> Self {
    Self::new(pre_tokenizer)
  }

  #[pyo3(name = "add_text")]
  pub fn py_add_text(&mut self, py: Python, text: &str) -> PyResult<()> {
    py.detach(|| self.add_text(text))
      .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
  }

  #[pyo3(name = "add_batch")]
  pub fn py_add_batch(&mut self, py: Python, texts: Vec<String>) -> PyResult<()> {
    py.detach(|| self.add_batch(&texts))
      .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
  }

  #[pyo3(name = "add_source", signature = (source, *, max_records=4096, max_bytes=67108864, prefetch=1))]
  pub fn py_add_source(
    &mut self, py: Python, source: &Bound<PyAny>, max_records: usize, max_bytes: usize, prefetch: i64,
  ) -> PyResult<()> {
    add_python_source(self, py, source, max_records, max_bytes, prefetch)
  }

  #[pyo3(name = "merge")]
  pub fn py_merge(&mut self, py: Python, other: BigramCounter) -> PyResult<()> {
    py.detach(|| self.merge(other))
      .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
  }

  #[pyo3(name = "selected")]
  pub fn py_selected(&self, top_k: usize, min_freq: i64) -> Vec<String> {
    BigramCounter::selected(self, top_k, min_freq)
      .into_iter()
      .map(unicode_bigram_to_string)
      .collect()
  }

  pub fn items(&self) -> Vec<(String, i64)> {
    let mut items = self.counts().iter()
      .map(|(bigram, frequency)| (unicode_bigram_to_string(*bigram), *frequency))
      .collect::<Vec<_>>();
    items.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    items
  }
}

#[pymethods]
impl WordCounter {
  #[new]
  pub fn new_py(pre_tokenizer: PreTokenizer) -> Self {
    Self::new(pre_tokenizer)
  }

  #[pyo3(name = "add_text")]
  pub fn py_add_text(&mut self, py: Python, text: &str) -> PyResult<()> {
    py.detach(|| self.add_text(text))
      .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
  }

  #[pyo3(name = "add_batch")]
  pub fn py_add_batch(&mut self, py: Python, texts: Vec<String>) -> PyResult<()> {
    py.detach(|| self.add_batch(&texts))
      .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
  }

  #[pyo3(name = "add_source", signature = (source, *, max_records=4096, max_bytes=67108864, prefetch=1))]
  pub fn py_add_source(
    &mut self, py: Python, source: &Bound<PyAny>, max_records: usize, max_bytes: usize, prefetch: i64,
  ) -> PyResult<()> {
    add_python_source(self, py, source, max_records, max_bytes, prefetch)
  }

  #[pyo3(name = "merge")]
  pub fn py_merge(&mut self, py: Python, other: WordCounter) -> PyResult<()> {
    py.detach(|| self.merge(other))
      .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
  }

  #[pyo3(name = "words")]
  pub fn py_words(&self) -> BTreeMap<String, i64> {
    WordCounter::words(self)
  }

  #[getter]
  #[pyo3(name = "len")]
  pub fn py_len(&self) -> usize {
    WordCounter::len(self)
  }

  #[pyo3(name = "clear")]
  pub fn py_clear(&mut self) {
    WordCounter::clear(self)
  }

  #[pyo3(name = "save")]
  pub fn py_save(&self, py: Python, path: PathBuf) -> PyResult<()> {
    py.detach(|| {
      let file = std::fs::File::create(path)?;
      self.save(std::io::BufWriter::new(file))
    }).map_err(|error| pyo3::exceptions::PyIOError::new_err(error.to_string()))
  }

}

#[pyclass]
pub struct BpeEncoderBase(Arc<dyn Encoder<Idx> + Send + Sync>);

fn _arc_to_vec<I: Copy>(i: Arc<[I]>) -> Vec<I> {
  i.iter().copied().collect()
}

fn new_bpe<C: Clone>(
  vocab: Option<BTreeMap<Vec<u8>, Idx>>,
  merges: Option<Vec<(Vec<u8>, Vec<u8>)>>,
  vocab_file: Option<PathBuf>,
  merges_file: Option<PathBuf>,
  special_tokens: Option<Vec<String>>,
  pat_str: Option<String>,
  spec: &dyn Spec<C, Idx>,
) -> MyResult<BpeEncoderBase>
where
  BpeEncoder<C>: CanEncode<C, Idx>
{
  let mut builder = BpeBuilder::new();
  if let Some(filename) = vocab_file {
    builder = builder.load_vocab_file(filename, spec)?;
  } else if let Some(vocab) = vocab {
    builder = builder.set_vocab(vocab.into_iter().map(|(k, v)| (v, k)).collect());
  } else {
    return Err(MyError::BpeBuilder("Either vocab_file or vocab must be provided".to_string()));
  }
  if let Some(filename) = merges_file {
    builder = builder.load_merges_file(filename, spec)?;
  } else if let Some(merges) = merges {
    builder = builder.set_merges_raw(merges);
  } else {
    return Err(MyError::BpeBuilder("Either merges_file or merges must be provided".to_string()));
  }
  builder= builder.set_special_tokens(special_tokens);
  builder = builder.set_pat_str(pat_str);
  let bpe = builder.build(spec)?;
  Ok(BpeEncoderBase(Arc::new(bpe)))
}

#[pymethods]
impl BpeEncoderBase {
  #[new]
  #[pyo3(signature = (format, unit, vocab, merges, vocab_file, merges_file, special_tokens, pat_str=None))]
  /// Create a Python BPE encoder.
  ///
  /// The encoder can be created from in-memory `vocab`/`merges` or from file paths.
  /// `format` and `unit` must be compatible.
  pub fn new_py(
    py: Python,
    format: &str, unit: &str,
    vocab: Option<BTreeMap<Vec<u8>, Idx>>,
    merges: Option<Vec<(Vec<u8>, Vec<u8>)>>,
    vocab_file: Option<PathBuf>,
    merges_file: Option<PathBuf>,
    special_tokens: Option<Vec<String>>,
    pat_str: Option<String>,
  ) -> PyResult<Self> {
    py.detach(||
      match (format, unit) {
        ("gpt2", "byte") => new_bpe::<u8>(vocab, merges, vocab_file, merges_file, special_tokens, pat_str, &Gpt2Spec),
        ("unitoken", "byte") => new_bpe::<u8>(vocab, merges, vocab_file, merges_file, special_tokens, pat_str, &UnitokenSpec),
        ("unitoken", "unicode") => new_bpe::<Character>(vocab, merges, vocab_file, merges_file, special_tokens, pat_str, &UnitokenSpec),
        _ => Err(MyError::SpecError(format!("format {format} is not compatible with unit {unit}"))),
      }
    ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
  }

  #[pyo3(name = "pre_tokenizer")]
  /// Return the underlying pre-tokenizer.
  pub fn py_pre_tokenizer(&self) -> PreTokenizer {
    self.0.pre_tokenizer().clone()
  }

  #[pyo3(name = "encode_word")]
  /// Encode a single word into token ids.
  pub fn py_encode_word(&self, py: Python, word: &str) -> PyResult<Vec<Idx>> {
    py.detach(||
      self.0.encode_word(word).map(_arc_to_vec)
    ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
  }

  #[pyo3(name = "encode_words")]
  /// Encode multiple words into token ids.
  pub fn py_encode_words(&self, py: Python, words: Vec<String>) -> PyResult<Vec<Vec<Idx>>> {
    py.detach(|| {
      let words = words.iter().map(|i| i.as_str()).collect::<Vec<_>>();
      let result = self.0.encode_words(&words)?;
      Ok(result.into_iter().map(_arc_to_vec).collect())
    }).map_err(|e: MyError| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
  }

  #[pyo3(name = "encode_to_numpy")]
  /// Encode an arbitrary string into a NumPy `uint32` array of token ids.
  pub fn py_encode_to_numpy<'py>(&self, py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyArray1<Idx>>> {
    let result = py.detach(|| {
      self.0.encode_string(text)
    });
    match result {
      Ok(v) => Ok(v.into_pyarray(py)),
      Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
    }
  }

  #[pyo3(name = "encode")]
  /// Encode an arbitrary string into a Python list of token ids.
  pub fn py_encode(&self, py: Python, text: &str) -> PyResult<Vec<Idx>> {
    py.detach(|| {
      self.0.encode_string(text)
    }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
  }

  #[pyo3(name = "encode_file")]
  /// Encode a file into a NumPy `uint32` array of token ids.
  pub fn py_encode_file<'py>(&self, py: Python<'py>, path: PathBuf, num_chunks: usize) -> PyResult<Bound<'py, PyArray1<Idx>>> {
    let result = py.detach(||
      self.0.encode_file(&path, num_chunks)
    );
    match result {
      Ok(v) => Ok(v.into_pyarray(py)),
      Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
    }
  }

  #[pyo3(name = "decode")]
  /// Decode token ids back into a UTF-8 string.
  ///
  /// Accepts either a Python sequence of ints or a NumPy `uint32` array.
  pub fn py_decode(&self, py: Python, idxs: &Bound<PyAny>) -> PyResult<String> {
    let vec: Vec<Idx> = if let Ok(v) = idxs.extract::<Vec<Idx>>() {
      v
    } else if let Ok(arr) = idxs.extract::<PyReadonlyArray1<Idx>>() {
      arr.as_array().iter().copied().collect()
    } else {
      return Err(pyo3::exceptions::PyTypeError::new_err(
        "idxs must be a sequence[int] or a numpy.ndarray[uint32]",
      ));
    };

    py.detach(|| self.0.decode(&vec))
      .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
  }
}

// #[pymodule(gil_used = false)]
// #[pyo3(name="_lib")]
// fn _tiktoken(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
//   m.add_class::<BpeTrainerBase>()?;
//   m.add_class::<BpeTrainer_u8_Idx>()?;
//   m.add_class::<BpeTrainer_Character_CharIdx>()?;
//   Ok(())
// }


}

#[test]
#[ignore = "manual"]
fn generate_py_stubs() {
  println!("test");
  let module = pyo3_introspection::introspect_cdylib(
      "./python/uni_tokenizer/_lib.cpython-313-darwin.so",
      "_lib",
  )
  .expect("introspection to succeed");
  let result = pyo3_introspection::module_stub_files(&module);
  println!("{result:?}");
  let value = result.get(&std::path::PathBuf::from("__init__.pyi")).unwrap();
  std::fs::write("./python/uni_tokenizer/_lib.pyi", value).unwrap();
}
