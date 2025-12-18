#[pyo3::pymodule(gil_used = false)]
mod _lib {
use std::{collections::BTreeMap, path::PathBuf, sync::Arc};

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use ordermap::OrderMap;
use pyo3::{prelude::*, pymethods, types::PyAny};

use crate::{MyError, MyResult, bpe::{BpeEncoder, BpeTrainer, CharIdx, CharSplit, Character, Idx, IdxLike, Word, encoder::BpeBuilder, utils::ToWord}, spec::{Spec, gpt2::Gpt2Spec, uni::UniSpec}, traits::{CanEncode, CanStrToWord, Encoder, Train as _}};

#[pyclass(subclass)]
pub struct BpeTrainerBase;

#[allow(dead_code)]
/// this is just a reference for impl blocks, not directly used
pub trait BpeTrainerBaseImpl: Sized {
  fn new_py(special_tokens: Vec<String>) -> (Self, BpeTrainerBase);

  fn add_words(&mut self, py: Python, words: Vec<(String, i64)>);
  fn vocab_size(&self) -> usize;
  fn init_training(&mut self, py: Python);
  fn step(&mut self, py: Python) -> PyResult<i64>;
  fn get_vocabs(&self) -> Vocabs;
  fn save_vocab(&self, py: Python, path: PathBuf, spec: &str) -> PyResult<()>;
  fn save_merges_txt(&self, py: Python, path: PathBuf, spec: &str) -> PyResult<()>;
}

// #[pyclass(eq, eq_int)]
// #[derive(PartialEq)]
// pub enum SpecEnum {
//   #[pyo3(name = "gpt2")]
//   Gpt2,
//   #[pyo3(name = "uni")]
//   Uni,
// }

// #[pyclass(eq, eq_int)]
// #[derive(PartialEq)]
// pub enum CharLevel {
//   #[pyo3(name = "u8")]
//   U8,
//   #[pyo3(name = "char")]
//   Char,
// }

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
  pub fn new_py(special_tokens: Vec<String>) -> (Self, BpeTrainerBase) {
    (
      Self {
        inner: BpeTrainer::new(vec![], special_tokens),
      },
      BpeTrainerBase {},
    )
  }

  /// Add `(word, frequency)` pairs to the trainer's inventory.
  pub fn add_words(&mut self, py: Python, words: Vec<(String, i64)>) {
    py.detach(||
      self.inner.add_words(&mut words.iter().map(|(w, f)| (w.as_str(), *f)))
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

  /// Perform one training step.
  ///
  /// Returns the updated vocabulary size.
  pub fn step(&mut self, py: Python) -> PyResult<i64> {
    py.detach(|| self.inner.step()).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(self.inner.vocab_size() as i64)
  }

  /// Return a view of the current vocabulary.
  pub fn get_vocabs(&self) -> Vocabs {
    Vocabs {
      inner: Box::new(VocabsInner::new(&self.inner.vocab)),
    }
  }

  /// Save the vocabulary JSON to `path` using the requested spec (`"gpt2"` or `"uni"`).
  pub fn save_vocab(&self, py: Python, path: PathBuf, spec: &str) -> PyResult<()> {
    py.detach(|| {
      let mut file = std::fs::File::create(&path)?;
      let mut writer = std::io::BufWriter::new(&mut file);
      match spec {
        "gpt2" => self.inner.save_vocab_json(&Gpt2Spec, &mut writer),
        "uni" => self.inner.save_vocab_json(&UniSpec, &mut writer),
        _ => Err(MyError::SpecError(format!("Unknown spec: {}", spec))),
      }
    }).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
  }

  /// Save merges to `path` using the requested spec (`"gpt2"` or `"uni"`).
  pub fn save_merges_txt(&self, py: Python, path: PathBuf, spec: &str) -> PyResult<()> {
    py.detach(|| {
      let mut file = std::fs::File::create(&path)?;
      let mut writer = std::io::BufWriter::new(&mut file);
      match spec {
        "gpt2" => self.inner.save_merges_txt(&Gpt2Spec, &mut writer),
        "uni" => self.inner.save_merges_txt(&UniSpec, &mut writer),
        _ => Err(MyError::SpecError(format!("Unknown spec: {}", spec))),
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
  pub fn new_py(special_tokens: Vec<String>) -> (Self, BpeTrainerBase) {
    (
      Self {
        inner: BpeTrainer::new(vec![], special_tokens),
      },
      BpeTrainerBase {},
    )
  }

  /// Add `(word, frequency)` pairs to the trainer's inventory.
  pub fn add_words(&mut self, py: Python, words: Vec<(String, i64)>) {
    py.detach(||
      self.inner.add_words(&mut words.iter().map(|(w, f)| (w.as_str(), *f)))
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

  /// Perform one training step.
  ///
  /// Returns the updated vocabulary size.
  pub fn step(&mut self, py: Python) -> PyResult<i64> {
    py.detach(|| self.inner.step()).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(self.inner.vocab_size() as i64)
  }

  /// Return a view of the current vocabulary.
  pub fn get_vocabs(&self) -> Vocabs {
    Vocabs {
      inner: Box::new(VocabsInner::new(&self.inner.vocab)),
    }
  }

  /// Save the vocabulary JSON to `path`.
  ///
  /// Note: `"gpt2"` is not supported for the character tokenizer.
  pub fn save_vocab(&self, py: Python, path: PathBuf, spec: &str) -> PyResult<()> {
    py.detach(|| {
      let mut file = std::fs::File::create(&path)?;
      let mut writer = std::io::BufWriter::new(&mut file);
      match spec {
        "gpt2" => Err(MyError::SpecError("gpt2 spec not supported for Character tokenizer".to_string())),
        "uni" => self.inner.save_vocab_json(&UniSpec, &mut writer),
        _ => Err(MyError::SpecError(format!("Unknown spec: {}", spec))),
      }
    }).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
  }

  /// Save merges to `path`.
  ///
  /// Note: `"gpt2"` is not supported for the character tokenizer.
  pub fn save_merges_txt(&self, py: Python, path: PathBuf, spec: &str) -> PyResult<()> {
    py.detach(|| {
      let mut file = std::fs::File::create(&path)?;
      let mut writer = std::io::BufWriter::new(&mut file);
      match spec {
        "gpt2" => Err(MyError::SpecError("gpt2 spec not supported for Character tokenizer".to_string())),
        "uni" => self.inner.save_merges_txt(&UniSpec, &mut writer),
        _ => Err(MyError::SpecError(format!("Unknown spec: {}", spec))),
      }
    }).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
  }
}

pub struct VocabsInner<C, I>(OrderMap<Word<C>, I>);

impl<C: std::hash::Hash + Eq, I: IdxLike> VocabsInner<C, I> {
  /// Build a reverse map from token bytes to token id.
  pub fn new(vocab: &BTreeMap<I, Word<C>>) -> Self {
    Self(vocab.iter().map(|(i, c)| (c.clone(), i.clone())).collect())
  }
}

trait VocabsImpl {
  fn len(&self) -> usize;
  fn get(&self, word: &str) -> Option<i64>;
  fn items(&self) -> Vec<(Vec<u8>, i64)>;
}

impl<C: CanStrToWord + CharSplit + std::hash::Hash + Eq, I: IdxLike> VocabsImpl for VocabsInner<C, I> {
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
pub struct Vocabs {
  inner: Box<dyn VocabsImpl + Send + Sync>,
}

#[pymethods]
impl Vocabs {
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

#[pymethods]
impl PreTokenizer {
  #[new]
  /// Create a Python `PreTokenizer`.
  ///
  /// - `special_tokens`: special tokens to treat as indivisible.
  /// - `eot_token`: end-of-text token used for chunk boundary alignment.
  /// - `pat`: optional regex pattern; defaults to the crate's default.
  #[pyo3(signature = (special_tokens, eot_token=None, pat=None))]
  pub fn new_py(special_tokens: Vec<String>, eot_token: Option<String>, pat: Option<String>) -> PyResult<Self> {
    Self::try_new(&special_tokens, eot_token.as_deref(), pat.as_deref())
      .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
  }

  #[pyo3(name = "find_chunk_boundaries", signature = (path, desired_num_chunks = 1024))]
  /// Python wrapper for [`PreTokenizer::find_chunk_boundaries`].
  pub fn py_find_chunk_boundaries(
    &self, py: Python, path: PathBuf, desired_num_chunks: usize,
  ) -> PyResult<Vec<(u64, usize)>> {
    py.detach(||
      self.find_chunk_boundaries(path, desired_num_chunks)
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

  #[pyo3(name = "get_words_from_file", signature = (path, desired_num_chunks = 1024))]
  /// Python wrapper for [`PreTokenizer::get_words_from_file`].
  pub fn py_get_words_from_file(
    &self, py: Python, path: PathBuf, desired_num_chunks: usize,
  ) -> PyResult<BTreeMap<String, i64>> {
    py.detach(||
      self.get_words_from_file(path, desired_num_chunks)
    ).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
  }
}

#[pyclass]
pub struct BpeEncoderBase(Arc<dyn Encoder<Idx> + Send + Sync>);

fn _arc_to_vec<I: Copy>(i: Arc<[I]>) -> Vec<I> {
  i.iter().copied().collect()
}

fn new_bpe<C: Clone>(
  vocabs: Option<BTreeMap<Vec<u8>, Idx>>,
  merges: Option<Vec<(Vec<u8>, Vec<u8>)>>,
  vocab_filename: Option<PathBuf>,
  merges_filename: Option<PathBuf>,
  special_tokens: Option<Vec<String>>,
  spec: &dyn Spec<C, Idx>,
) -> MyResult<BpeEncoderBase>
where
  BpeEncoder<C>: CanEncode<C, Idx>
{
  let mut builder = BpeBuilder::new();
  if let Some(filename) = vocab_filename {
    builder = builder.load_vocab_file(filename, spec)?;
  } else if let Some(vocabs) = vocabs {
    builder = builder.set_vocab(vocabs.into_iter().map(|(k, v)| (v, k)).collect());
  } else {
    return Err(MyError::BpeBuilder("Either vocab_filename or vocabs must be provided".to_string()));
  }
  if let Some(filename) = merges_filename {
    builder = builder.load_merges_file(filename, spec)?;
  } else if let Some(merges) = merges {
    builder = builder.set_merges_raw(merges);
  } else {
    return Err(MyError::BpeBuilder("Either merges_filename or merges must be provided".to_string()));
  }
  builder= builder.set_special_tokens(special_tokens);
  let bpe = builder.build(spec)?;
  Ok(BpeEncoderBase(Arc::new(bpe)))
}

#[pymethods]
impl BpeEncoderBase {
  #[new]
  /// Create a Python BPE encoder.
  ///
  /// The encoder can be created from in-memory `vocabs`/`merges` or from file paths.
  /// `spec` and `char_level` must be compatible (e.g. `("gpt2", "u8")`, `("uni", "char")`).
  pub fn new_py(
    py: Python,
    spec: &str, char_level: &str,
    vocabs: Option<BTreeMap<Vec<u8>, Idx>>,
    merges: Option<Vec<(Vec<u8>, Vec<u8>)>>,
    vocab_filename: Option<PathBuf>,
    merges_filename: Option<PathBuf>,
    special_tokens: Option<Vec<String>>,
  ) -> PyResult<Self> {
    py.detach(||
      match (spec, char_level) {
        ("gpt2", "u8") => new_bpe::<u8>(vocabs, merges, vocab_filename, merges_filename, special_tokens, &Gpt2Spec),
        ("uni", "u8") => new_bpe::<u8>(vocabs, merges, vocab_filename, merges_filename, special_tokens, &UniSpec),
        ("uni", "char") => new_bpe::<Character>(vocabs, merges, vocab_filename, merges_filename, special_tokens, &UniSpec),
        _ => Err(MyError::SpecError(format!("spec {spec} not compatibale with {char_level}"))),
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

  #[pyo3(name = "encode_string")]
  /// Encode an arbitrary string into a NumPy `uint32` array of token ids.
  pub fn py_encode_string<'py>(&self, py: Python<'py>, s: &str) -> PyResult<Bound<'py, PyArray1<Idx>>> {
    let result = py.detach(|| {
      self.0.encode_string(s)
    });
    match result {
      Ok(v) => Ok(v.into_pyarray(py)),
      Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
    }
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
