use fancy_regex::Regex;
use lazy_static::lazy_static;
use memchr::memmem;
use rayon::iter::{IntoParallelIterator, ParallelIterator as _};
use std::{
  collections::{BTreeMap, BTreeSet, HashMap},
  fs::{self, File},
  io::{Read as _, Seek},
  path::Path,
};

use crate::{MyError, MyResult, bpe::Freq};

lazy_static! {
  /// PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
  pub static ref DEFAULT_PAT: Regex = Regex::new(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+").unwrap();
}
pub const DEFAULT_EOT: &'static str = "<|endoftext|>";

#[derive(Clone, Debug)]
#[cfg_attr(feature = "py", pyo3::pyclass(from_py_object))]
pub struct PreTokenizer {
  pub re_pat: Regex,
  pub re_special_tokens: Regex,
  pub end_of_text: String,
  pub metrics: bool
}

impl PreTokenizer {
  /// Create a pre-tokenizer using the default pattern.
  ///
  /// This is an infallible convenience wrapper around [`Self::try_new`].
  ///
  /// - `special_tokens`: Tokens that should be detected as indivisible chunks.
  /// - `end_of_text`: Token used as the document boundary marker when chunking files.
  pub fn new(special_tokens: &[String], end_of_text: Option<&str>) -> Self {
    // Infallible default constructor.
    Self::try_new(special_tokens, end_of_text, None).expect("DEFAULT_PAT must be valid")
  }

  /// Create a pre-tokenizer with an optional custom regex pattern.
  ///
  /// When `pat` is `None`, uses `DEFAULT_PAT`.
  pub fn try_new(
    special_tokens: &[String], end_of_text: Option<&str>, pat: Option<&str>,
  ) -> MyResult<Self> {
    let re_pat = match pat {
      Some(pat) => Regex::new(pat)?,
      None => DEFAULT_PAT.clone(),
    };
    let re_special_tokens = create_special_token_regex(special_tokens);
    Ok(Self {
      re_pat,
      re_special_tokens,
      end_of_text: end_of_text.unwrap_or(DEFAULT_EOT).to_string(),
      metrics: true,
    })
  }

  /// Count pre-tokenized pieces in a string.
  ///
  /// Returns a map from token slice to frequency. The keys borrow from `text`.
  pub fn count_tokens<'a>(&self, text: &'a str) -> MyResult<BTreeMap<&'a str, Freq>> {
    _pretokenizer_counter(text, &self.re_pat)
  }

  /// Compute byte `(offset, len)` pairs that split a file into approximately `desired_num_chunks`.
  ///
  /// Boundaries are adjusted to fall on occurrences of `self.end_of_text` (the EOT marker),
  /// so that chunks do not split across document boundaries.
  pub fn find_chunk_boundaries<P: AsRef<Path>>(
    &self, path: P, desired_num_chunks: usize,
  ) -> MyResult<Vec<(u64, usize)>> {
    let boundaries = _find_chunk_boundaries(&path, desired_num_chunks, &self.end_of_text)?;
    Ok(boundaries.iter().zip(boundaries.iter().skip(1)).map(|(&a, &b)| (a, (b-a) as usize)).collect())
  }

  /// Build token and special-token indexes for a segment.
  ///
  /// The returned maps associate each token (borrowed from `content`) with the list of
  /// positions it appears in the fully-tokenized stream.
  #[hotpath::measure]
  pub fn get_tokens_index_from_segment<'a>(
    &self, content: &'a str,
  ) -> MyResult<(HashMap<&'a str, Vec<usize>>, HashMap<&'a str, Vec<usize>>)> {
    let _span = trace_span!("get_tokens_index_from_segment", len=content.len()).entered();

    if self.metrics {
      metrics::counter!("get_tokens_index_from_segment.calls").increment(1);
    }
    let parts = split_special_tokens(&content, &self.re_special_tokens)?;
    let mut tokens_index: HashMap<&'a str, Vec<usize>> = HashMap::new();
    let mut special_tokens_index: HashMap<&'a str, Vec<usize>> = HashMap::new();
    let mut doc_idx = 0;
    for part in parts.into_iter() {
      match part {
        SplitChunk::Special(token) => {
          special_tokens_index.entry(token).or_default().push(doc_idx);
          doc_idx += 1;
        }
        SplitChunk::Chunk(part) => {
          for token in self.re_pat.find_iter(part) {
            tokens_index.entry(token?.as_str()).or_default().push(doc_idx);
            doc_idx += 1;
          }
        }
      }
    }

    if self.metrics {
      metrics::counter!("get_tokens_index_from_segment.len").increment(content.len() as _);
      metrics::histogram!("get_tokens_index_from_segment.special_tokens_sum").record(special_tokens_index.values().map(Vec::len).sum::<usize>() as f64);
      metrics::histogram!("get_tokens_index_from_segment.tokens_count").record(tokens_index.len() as f64);
      metrics::histogram!("get_tokens_index_from_segment.doc_idx").record(doc_idx as f64);
    }

    trace!(tokens_index_len=?tokens_index.len(), "result");
    Ok((tokens_index, special_tokens_index))
  }

  /// Read a slice of a file and count pre-tokenized word frequencies within it.
  ///
  /// The byte range is described by `(offset, len)`. Special tokens are excluded from counting.
  #[hotpath::measure]
  pub fn get_words_from_segment<P: AsRef<Path>>(
    &self, path: P, offset: u64, len: usize,
  ) -> MyResult<BTreeMap<String, Freq>> {
    let _span = trace_span!("get_words_from_segment", offset = offset, len = len).entered();

    if self.metrics {
      metrics::counter!("get_words_from_segment.calls").increment(1);
    }
    let buffer = _read_file_to_buffer(&path, offset, len)?;

    let content = String::from_utf8_lossy(&buffer);
    let parts = split_special_tokens(&content, &self.re_special_tokens)?;
    let mut words = BTreeMap::new();
    for part in parts.iter().filter(|i| !i.is_special()) {
      for (token, count) in _pretokenizer_counter(part.as_str(), &self.re_pat)? {
        *words.entry(token).or_default() += count;
      }
    }
    if self.metrics {
      metrics::histogram!("get_words_from_segment.words_count").record(words.len() as f64);
      metrics::counter!("get_words_from_segment.len").increment(len as _);
    }

    trace!(words_len=?words.len(), "result");
    Ok(words.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
  }

  /// Count pre-tokenized word frequencies across an entire file.
  ///
  /// The file is split into `num_chunks` using [`Self::find_chunk_boundaries`], processed in
  /// parallel, and merged into a single frequency map.
  pub fn get_words_from_file<P: AsRef<Path>>(
    &self, path: P, num_chunks: usize,
  ) -> MyResult<BTreeMap<String, Freq>> {
    let boundaries = _find_chunk_boundaries(&path, num_chunks, &self.end_of_text)?;
    let path = path.as_ref().to_path_buf();
    let params = boundaries
      .iter()
      .zip(boundaries.iter().skip(1))
      .map(|(start, end)| (*start, (*end - *start) as usize))
      .collect::<Vec<_>>();

    let words = params
      .into_par_iter()
      .map(|(offset, len)| self.get_words_from_segment(&path, offset, len))
      .try_reduce(
        || BTreeMap::new(),
        |a, b| {
          let (mut a, b) = if a.len() < b.len() {
            (b, a)
          } else {
            (a, b)
          };
          for (k, v) in b.into_iter() {
            *a.entry(k).or_default() += v;
          }
          Ok(a)
        },
      )?;
    Ok(words)
  }
}

/// Tokenize a string using `pat` and return token frequencies.
///
/// The returned keys borrow from `s`.
pub fn _pretokenizer_counter<'a>(s: &'a str, pat: &Regex) -> MyResult<BTreeMap<&'a str, Freq>> {
  let mut result = BTreeMap::new();
  for i in pat.find_iter(s) {
    let token = i?.as_str();
    *result.entry(token).or_default() += 1;
  }
  Ok(result)
}

#[hotpath::measure]
/// Find byte offsets that can be used to split a file into `desired_num_chunks`.
///
/// Offsets are aligned to occurrences of `split_special_token` to avoid splitting across
/// boundaries (typically the end-of-text token).
pub fn _find_chunk_boundaries<P: AsRef<Path>>(
  path: P, desired_num_chunks: usize, split_special_token: &str,
) -> MyResult<Vec<u64>> {
  let file_size = fs::metadata(&path)?.len();
  let chunk_size = file_size / desired_num_chunks as u64;
  let mini_chunk_size = 4096;
  let finder = memmem::Finder::new(split_special_token);
  debug!(
    file_size = file_size,
    chunk_size = chunk_size,
    desired_num_chunks = desired_num_chunks,
    "find_chunk_boundaries"
  );

  let mut boundaries = Vec::new();
  for i in 0..(desired_num_chunks) {
    boundaries.push(chunk_size * i as u64);
  }
  boundaries.push(file_size);

  let mut file = File::open(&path)?;
  for bi in 1..boundaries.len() - 1 {
    let mut initial_position = boundaries[bi];
    let _ = file.seek(std::io::SeekFrom::Start(initial_position))?;
    loop {
      let mut buffer = vec![0; mini_chunk_size as usize];
      let bytes_read = file.read(&mut buffer)?;
      if bytes_read < mini_chunk_size as usize {
        boundaries[bi] = file_size;
        break;
      }
      if let Some(pos) = finder.find(buffer[..bytes_read].as_ref()) {
        let boundary = initial_position + pos as u64;
        boundaries[bi] = boundary;
        break;
      }
      initial_position += mini_chunk_size;
    }
  }

  let deduplicated_boundaries = boundaries.into_iter().collect::<BTreeSet<_>>();
  debug!(boundaries.len=?deduplicated_boundaries.len(), "find_chunk_boundaries");
  Ok(deduplicated_boundaries.into_iter().collect())
}

pub enum SplitChunk<'a> {
  Special(&'a str),
  Chunk(&'a str),
}

impl<'a> SplitChunk<'a> {
  /// Return the underlying string slice.
  pub fn as_str(&self) -> &'a str {
    match self {
      SplitChunk::Special(s) => s,
      SplitChunk::Chunk(s) => s,
    }
  }

  /// Whether this chunk is a special token match.
  pub fn is_special(&self) -> bool {
    matches!(self, SplitChunk::Special(_))
  }
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub enum SplitToken {
  Special(String),
  Token(String),
}

impl SplitToken {
  /// Return the underlying string slice.
  pub fn as_str(&self) -> &str {
    match self {
      SplitToken::Special(s) => s.as_str(),
      SplitToken::Token(s) => s.as_str(),
    }
  }

  /// Whether this token is marked as special.
  pub fn is_special(&self) -> bool {
    matches!(self, SplitToken::Special(_))
  }
}

impl std::ops::Deref for SplitToken {
  type Target = str;

  fn deref(&self) -> &Self::Target {
    self.as_str()
  }
}

/// Build a regex that matches any of the provided `special_tokens`.
///
/// If `special_tokens` is empty, returns a regex that matches nothing.
pub fn create_special_token_regex(special_tokens: &[String]) -> Regex {
  if special_tokens.is_empty() {
    return Regex::new("$^").unwrap(); // matches nothing
  }
  let pattern = special_tokens
    .iter()
    .map(|s| fancy_regex::escape(s).into_owned())
    .collect::<Vec<String>>()
    .join("|");
  Regex::new(&pattern).unwrap()
}

/// Split `text` into alternating regular chunks and exact special-token chunks.
///
/// The `special_tokens` regex should match only the special tokens (typically built with
/// [`create_special_token_regex`]). Returned chunks borrow from `text`.
pub fn split_special_tokens<'a>(text: &'a str, special_tokens: &Regex) -> MyResult<Vec<SplitChunk<'a>>> {
  let mut parts = Vec::new();
  let mut last_pos = 0;
  for mat in special_tokens.find_iter(text) {
    match mat {
      Ok(m) => {
        if m.start() > last_pos {
          parts.push(SplitChunk::Chunk(&text[last_pos..m.start()]));
        }
        parts.push(SplitChunk::Special(&text[m.start()..m.end()]));
        last_pos = m.end();
      }
      Err(e) => return Err(MyError::Regex(e)),
    }
  }
  if last_pos < text.len() {
    parts.push(SplitChunk::Chunk(&text[last_pos..]));
  }
  Ok(parts)
}

#[hotpath::measure]
/// Read `len` bytes from `path` starting at `offset`.
///
/// This is a low-level helper used by the pre-tokenizer and encoder.
pub fn _read_file_to_buffer<P: AsRef<Path>>(path: P, offset: u64, len: usize) -> MyResult<Vec<u8>> {
  let mut file = File::open(&path)?;
  file.seek(std::io::SeekFrom::Start(offset))?;
  let mut buffer = vec![0; len];
  file.read_exact(&mut buffer)?;
  Ok(buffer)
}

/// Sort a word-frequency map into a stable, descending-by-frequency order.
///
/// Ties are broken by lexicographic order of the word.
pub fn sort_words(words: &BTreeMap<String, Freq>) -> ordermap::OrderMap<String, Freq> {
  let mut word_freq_vec: Vec<(String, Freq)> = words.iter().map(|(k,v)| (k.clone(), *v)).collect();
  word_freq_vec.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)).reverse());
  word_freq_vec.into_iter().collect()
}

/// Save a sorted word-frequency map as pretty-printed JSON.
pub fn save_words<W: std::io::Write>(w: W, words: &ordermap::OrderMap<String, Freq>) -> Result<(), std::io::Error> {
  serde_json::to_writer_pretty(w, &words)?;
  Ok(())
}

#[cfg(test)]
mod tests {
  use ordermap::OrderMap;
  use super::*;
  #[test]
  fn test_pretokenizer() {
    let s = "Hello, world! It's 2024.";
    let tokens = _pretokenizer_counter(s, &DEFAULT_PAT).unwrap();
    let expected_tokens = vec![
      ("Hello", 1),
      (",", 1),
      (" world", 1),
      ("!", 1),
      (" It", 1),
      ("'s", 1),
      (" 2024", 1),
      (".", 1),
    ]
    .into_iter()
    .collect::<BTreeMap<_, _>>();
    assert_eq!(tokens, expected_tokens);

    let s = "你好，世界！Now是2024年。";
    let tokens = _pretokenizer_counter(s, &DEFAULT_PAT).unwrap();
    let expected_tokens = vec![
      ("你好", 1),
      ("，", 1),
      ("世界", 1),
      ("！", 1),
      ("Now是", 1),
      ("2024", 1),
      ("年", 1),
      ("。", 1),
    ]
    .into_iter()
    .collect::<BTreeMap<_, _>>();
    assert_eq!(tokens, expected_tokens);
  }

  #[test]
  fn test_sample() {
    let input = std::fs::read_to_string("fixtures/tinystories_sample_5M.txt").unwrap();
    let tokens = _pretokenizer_counter(&input, &DEFAULT_PAT).unwrap();
    assert_eq!(tokens.get(" the").cloned().unwrap_or(0), 48886);
  }

  #[test]
  fn test_find_chunk_boundaries() {
    let path = std::path::Path::new("fixtures/tinystories_sample_5M.txt");

    let desired_num_chunks = 4;
    let boundaries = _find_chunk_boundaries(path, desired_num_chunks, DEFAULT_EOT).unwrap();
    let expect = vec![0, 1310951, 2621933, 3932548, 5242880];
    assert!(boundaries == expect, "{:?} != {:?}", boundaries, expect);

    let desired_num_chunks = 10;
    let boundaries = _find_chunk_boundaries(path, desired_num_chunks, DEFAULT_EOT).unwrap();
    let expect = vec![
      0, 525166, 1048920, 1573438, 2097691, 2621933, 3146237, 3670035, 4196392, 4718956, 5242880,
    ];
    assert!(boundaries == expect, "{:?} != {:?}", boundaries, expect);
  }

  #[test]
  fn test_get_words_from_file() {
    const NAME: &str = "tinystories_sample_5M";
    // const NAME: &str = "TinyStoriesV2-GPT4-train";
    let path = format!("fixtures/{NAME}.txt");
    let num_chunks = 16;
    let pre_tokenizer = PreTokenizer::new(&vec![DEFAULT_EOT.to_string()], Some(DEFAULT_EOT));
    let words = pre_tokenizer.get_words_from_file(
      path,
      num_chunks,
    )
    .unwrap();
    let words = sort_words(&words);
    if NAME == "tinystories_sample_5M" {
      assert_eq!(words.get(" the").cloned().unwrap_or(0), 48886);
    }
    std::fs::create_dir_all("out").ok();
    serde_json::to_writer_pretty(std::fs::File::create(format!("out/_words.{NAME}.json")).unwrap(), &words).unwrap();
    let answer = std::fs::read_to_string(format!("fixtures/_words.{NAME}.json")).unwrap();
    let expected: OrderMap<String, Freq> = serde_json::from_str(&answer).unwrap();
    assert_eq!(words, expected);
  }

  #[test]
  fn test_split_special_tokens() {
    const NAME: &str = "tinystories_sample_5M";
    let path = format!("fixtures/{NAME}.txt");
    let text = std::fs::read_to_string(&path).unwrap();
    let parts = split_special_tokens(
      &text,
      &create_special_token_regex(&[DEFAULT_EOT.to_string()]),
    ).unwrap();
    assert!(parts.len() == 12915);
  }

  #[test]
  fn test_get_tokens_index_from_segment() {
    const NAME: &str = "tinystories_sample_5M";
    let path = format!("fixtures/{NAME}.txt");
    let text = std::fs::read_to_string(&path).unwrap();
    let tokenizer = PreTokenizer::new(&vec![DEFAULT_EOT.to_string()], Some(DEFAULT_EOT));
    let (tokens_index, special_tokens_index) = tokenizer.get_tokens_index_from_segment(
      &text,
    ).unwrap();
    let idxs = tokens_index.get(" the").unwrap();
    println!("the idxs length: {:?}", idxs.len());
    assert_ne!(idxs.len(), 0);
    assert_eq!(special_tokens_index.len(), 1)
  }

  #[test]
  fn test_custom_pat_is_used_everywhere() {
    // Split into single characters, ignoring whitespace.
    let pat = r"[^\s]";
    let t = PreTokenizer::try_new(&vec![DEFAULT_EOT.to_string()], Some(DEFAULT_EOT), Some(pat)).unwrap();

    let s = "ab cd";
    let counts = t.count_tokens(s).unwrap();
    assert_eq!(counts.get("a").cloned().unwrap_or(0), 1);
    assert_eq!(counts.get("b").cloned().unwrap_or(0), 1);
    assert_eq!(counts.get("c").cloned().unwrap_or(0), 1);
    assert_eq!(counts.get("d").cloned().unwrap_or(0), 1);
  }
}
