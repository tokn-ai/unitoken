use fancy_regex::Regex;
use lazy_static::lazy_static;
use memchr::memmem;
use rayon::iter::{IntoParallelIterator, ParallelIterator as _};
use std::{
  collections::{BTreeMap, BTreeSet},
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryMode {
  Auto,
  Eot,
  Line,
  Utf8,
}

impl BoundaryMode {
  pub fn parse(value: &str) -> MyResult<Self> {
    match value {
      "auto" => Ok(Self::Auto),
      "eot" => Ok(Self::Eot),
      "line" => Ok(Self::Line),
      "utf8" => Ok(Self::Utf8),
      _ => Err(MyError::SpecError(format!("Unknown boundary mode: {value}"))),
    }
  }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkHint {
  Count(usize),
  Size(u64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkOptions {
  pub hint: ChunkHint,
  pub boundary: BoundaryMode,
}

impl ChunkOptions {
  pub fn count(chunks: usize) -> Self {
    Self {
      hint: ChunkHint::Count(chunks),
      boundary: BoundaryMode::Auto,
    }
  }

  pub fn chunk_count(&self, file_size: u64) -> usize {
    match self.hint {
      ChunkHint::Count(count) => count.max(1),
      ChunkHint::Size(size) => {
        if size == 0 || file_size == 0 {
          1
        } else {
          file_size.div_ceil(size) as usize
        }
      }
    }
  }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "py", pyo3::pyclass(from_py_object))]
pub struct PreTokenizer {
  pub re_pat: Regex,
  pub re_special_tokens: Regex,
  pub end_of_text: String,
  pub unicode_bigrams: Option<BTreeSet<(char, char)>>,
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
      unicode_bigrams: None,
      metrics: true,
    })
  }

  pub fn with_unicode_bigrams(mut self, bigrams: BTreeSet<(char, char)>) -> Self {
    self.unicode_bigrams = Some(bigrams);
    self
  }

  /// Count pre-tokenized pieces in a string.
  ///
  /// Returns a map from token slice to frequency. The keys borrow from `text`.
  pub fn count_tokens<'a>(&self, text: &'a str) -> MyResult<BTreeMap<&'a str, Freq>> {
    _pretokenizer_counter(text, &self.re_pat)
  }

  pub fn count_tokens_owned(&self, text: &str) -> MyResult<BTreeMap<String, Freq>> {
    _pretokenizer_counter_with_unicode_bigrams(text, &self.re_pat, self.unicode_bigrams.as_ref())
  }

  /// Compute byte `(offset, len)` pairs that split a file into approximately `desired_num_chunks`.
  ///
  /// Boundaries are adjusted to fall on occurrences of `self.end_of_text` (the EOT marker),
  /// so that chunks do not split across document boundaries.
  pub fn find_chunk_boundaries<P: AsRef<Path>>(
    &self, path: P, desired_num_chunks: usize,
  ) -> MyResult<Vec<(u64, usize)>> {
    self.find_chunk_boundaries_with_options(path, ChunkOptions::count(desired_num_chunks))
  }

  pub fn find_chunk_boundaries_with_options<P: AsRef<Path>>(
    &self, path: P, options: ChunkOptions,
  ) -> MyResult<Vec<(u64, usize)>> {
    let boundaries = _find_chunk_boundaries_with_options(&path, options, &self.end_of_text)?;
    Ok(boundaries.iter().zip(boundaries.iter().skip(1)).map(|(&a, &b)| (a, (b-a) as usize)).collect())
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
      for (token, count) in _pretokenizer_counter_with_unicode_bigrams(part.as_str(), &self.re_pat, self.unicode_bigrams.as_ref())? {
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
    self.get_words_from_file_with_options(path, ChunkOptions::count(num_chunks))
  }

  pub fn get_words_from_file_with_options<P: AsRef<Path>>(
    &self, path: P, options: ChunkOptions,
  ) -> MyResult<BTreeMap<String, Freq>> {
    let boundaries = _find_chunk_boundaries_with_options(&path, options, &self.end_of_text)?;
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

  pub fn build_unicode_bigram_set_from_file_with_options<P: AsRef<Path>>(
    &self, path: P, options: ChunkOptions, top_k: usize, min_freq: Freq,
  ) -> MyResult<BTreeSet<(char, char)>> {
    let boundaries = _find_chunk_boundaries_with_options(&path, options, &self.end_of_text)?;
    let path = path.as_ref().to_path_buf();
    let params = boundaries
      .iter()
      .zip(boundaries.iter().skip(1))
      .map(|(start, end)| (*start, (*end - *start) as usize))
      .collect::<Vec<_>>();

    let counts = params
      .into_par_iter()
      .map(|(offset, len)| self.count_unicode_bigrams_from_segment(&path, offset, len))
      .try_reduce(
        || BTreeMap::new(),
        |mut a, b| {
          for (k, v) in b {
            *a.entry(k).or_default() += v;
          }
          Ok(a)
        },
      )?;
    Ok(select_unicode_bigrams(counts, top_k, min_freq))
  }

  fn count_unicode_bigrams_from_segment<P: AsRef<Path>>(
    &self, path: P, offset: u64, len: usize,
  ) -> MyResult<BTreeMap<(char, char), Freq>> {
    let buffer = _read_file_to_buffer(&path, offset, len)?;
    let content = String::from_utf8_lossy(&buffer);
    let parts = split_special_tokens(&content, &self.re_special_tokens)?;
    let mut counts = BTreeMap::new();
    for part in parts.iter().filter(|i| !i.is_special()) {
      for i in self.re_pat.find_iter(part.as_str()) {
        let token = i?.as_str();
        if contains_cjk(token) {
          count_unicode_bigrams(token, &mut counts);
        }
      }
    }
    Ok(counts)
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

pub fn _pretokenizer_counter_with_unicode_bigrams(
  s: &str, pat: &Regex, unicode_bigrams: Option<&BTreeSet<(char, char)>>,
) -> MyResult<BTreeMap<String, Freq>> {
  let mut result = BTreeMap::new();
  for i in pat.find_iter(s) {
    let token = i?.as_str();
    if let Some(unicode_bigrams) = unicode_bigrams.filter(|_| contains_cjk(token)) {
      for segment in split_by_unicode_bigrams(token, unicode_bigrams) {
        *result.entry(segment).or_default() += 1;
      }
    } else {
      *result.entry(token.to_string()).or_default() += 1;
    }
  }
  Ok(result)
}

pub fn parse_unicode_bigrams(bigrams: &[String]) -> MyResult<BTreeSet<(char, char)>> {
  let mut parsed = BTreeSet::new();
  for bigram in bigrams {
    let chars = bigram.chars().collect::<Vec<_>>();
    if chars.len() != 2 {
      return Err(MyError::SpecError(format!("Unicode bigram must contain exactly two chars: {bigram:?}")));
    }
    parsed.insert((chars[0], chars[1]));
  }
  Ok(parsed)
}

pub fn unicode_bigram_to_string(bigram: (char, char)) -> String {
  let mut s = String::new();
  s.push(bigram.0);
  s.push(bigram.1);
  s
}

fn select_unicode_bigrams(
  counts: BTreeMap<(char, char), Freq>, top_k: usize, min_freq: Freq,
) -> BTreeSet<(char, char)> {
  let mut sorted = counts
    .into_iter()
    .filter(|(_, freq)| *freq >= min_freq)
    .collect::<Vec<_>>();
  sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
  sorted.into_iter().take(top_k).map(|(bigram, _)| bigram).collect()
}

fn count_unicode_bigrams(token: &str, counts: &mut BTreeMap<(char, char), Freq>) {
  let mut chars = token.chars();
  let Some(mut prev) = chars.next() else {
    return;
  };
  for next in chars {
    *counts.entry((prev, next)).or_default() += 1;
    prev = next;
  }
}

fn split_by_unicode_bigrams(token: &str, unicode_bigrams: &BTreeSet<(char, char)>) -> Vec<String> {
  let chars = token.char_indices().collect::<Vec<_>>();
  if chars.len() <= 1 {
    return vec![token.to_string()];
  }
  let mut segments = Vec::new();
  let mut start = 0;
  for pair in chars.windows(2) {
    let (_, left) = pair[0];
    let (right_byte, right) = pair[1];
    if !unicode_bigrams.contains(&(left, right)) {
      if start < right_byte {
        segments.push(token[start..right_byte].to_string());
      }
      start = right_byte;
    }
  }
  if start < token.len() {
    segments.push(token[start..].to_string());
  }
  segments
}

fn contains_cjk(s: &str) -> bool {
  s.chars().any(is_cjk)
}

fn is_cjk(ch: char) -> bool {
  matches!(
    ch as u32,
    0x3400..=0x4DBF
      | 0x4E00..=0x9FFF
      | 0xF900..=0xFAFF
      | 0x20000..=0x2A6DF
      | 0x2A700..=0x2B73F
      | 0x2B740..=0x2B81F
      | 0x2B820..=0x2CEAF
      | 0x2CEB0..=0x2EBEF
      | 0x30000..=0x3134F
  )
}

#[hotpath::measure]
/// Find byte offsets that can be used to split a file into `desired_num_chunks`.
///
/// Offsets are aligned to occurrences of `split_special_token` to avoid splitting across
/// boundaries (typically the end-of-text token).
pub fn _find_chunk_boundaries<P: AsRef<Path>>(
  path: P, desired_num_chunks: usize, split_special_token: &str,
) -> MyResult<Vec<u64>> {
  _find_chunk_boundaries_with_options(path, ChunkOptions::count(desired_num_chunks), split_special_token)
}

pub fn _find_chunk_boundaries_with_options<P: AsRef<Path>>(
  path: P, options: ChunkOptions, split_special_token: &str,
) -> MyResult<Vec<u64>> {
  let file_size = fs::metadata(&path)?.len();
  let desired_num_chunks = options.chunk_count(file_size);
  if desired_num_chunks <= 1 || file_size == 0 {
    return Ok(vec![0, file_size]);
  }
  let chunk_size = match options.hint {
    ChunkHint::Count(_) => file_size / desired_num_chunks as u64,
    ChunkHint::Size(size) => size,
  };
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
  let has_split_token = if matches!(options.boundary, BoundaryMode::Auto | BoundaryMode::Eot) {
    _file_contains(&mut file, &finder, mini_chunk_size, split_special_token.len())?
  } else {
    false
  };

  match options.boundary {
    BoundaryMode::Eot => {
      if !has_split_token {
        return Ok(vec![0, file_size]);
      }
      _align_eot_boundaries(&mut file, &mut boundaries, &finder, mini_chunk_size, file_size)?;
    }
    BoundaryMode::Auto if has_split_token => {
      _align_eot_boundaries(&mut file, &mut boundaries, &finder, mini_chunk_size, file_size)?;
    }
    BoundaryMode::Auto | BoundaryMode::Line => {
      for boundary in boundaries.iter_mut().skip(1).take(desired_num_chunks - 1) {
        *boundary = _align_line_boundary(&mut file, *boundary, file_size, mini_chunk_size)?;
      }
    }
    BoundaryMode::Utf8 => {
      for boundary in boundaries.iter_mut().skip(1).take(desired_num_chunks - 1) {
        *boundary = _align_utf8_boundary(&mut file, *boundary, file_size)?;
      }
    }
  }

  let deduplicated_boundaries = boundaries.into_iter().collect::<BTreeSet<_>>();
  debug!(boundaries.len=?deduplicated_boundaries.len(), "find_chunk_boundaries");
  Ok(deduplicated_boundaries.into_iter().collect())
}

fn _align_eot_boundaries(
  file: &mut File, boundaries: &mut [u64], finder: &memmem::Finder, window_size: usize, file_size: u64,
) -> MyResult<()> {
  let interior_count = boundaries.len().saturating_sub(2);
  for boundary in boundaries.iter_mut().skip(1).take(interior_count) {
    let mut initial_position = *boundary;
    let _ = file.seek(std::io::SeekFrom::Start(initial_position))?;
    loop {
      let mut buffer = vec![0; window_size];
      let bytes_read = file.read(&mut buffer)?;
      if bytes_read < window_size {
        *boundary = file_size;
        break;
      }
      if let Some(pos) = finder.find(buffer[..bytes_read].as_ref()) {
        *boundary = initial_position + pos as u64;
        break;
      }
      initial_position += window_size as u64;
    }
  }
  Ok(())
}

fn _align_line_boundary(file: &mut File, boundary: u64, file_size: u64, window_size: usize) -> MyResult<u64> {
  if boundary == 0 || boundary >= file_size {
    return Ok(boundary);
  }
  file.seek(std::io::SeekFrom::Start(boundary - 1))?;
  let mut previous = [0; 1];
  file.read_exact(&mut previous)?;
  if previous[0] == b'\n' {
    return Ok(boundary);
  }
  file.seek(std::io::SeekFrom::Start(boundary))?;
  let mut buffer = vec![0; window_size];
  let bytes_read = file.read(&mut buffer)?;
  if let Some(pos) = memchr::memchr(b'\n', &buffer[..bytes_read]) {
    return Ok((boundary + pos as u64 + 1).min(file_size));
  }
  _align_utf8_boundary(file, boundary, file_size)
}

fn _file_contains(file: &mut File, finder: &memmem::Finder, chunk_size: usize, needle_len: usize) -> MyResult<bool> {
  file.seek(std::io::SeekFrom::Start(0))?;
  let overlap_len = needle_len.saturating_sub(1);
  let mut overlap = Vec::new();
  loop {
    let mut buffer = vec![0; chunk_size];
    let bytes_read = file.read(&mut buffer)?;
    if bytes_read == 0 {
      return Ok(false);
    }
    buffer.truncate(bytes_read);

    if overlap.is_empty() {
      if finder.find(&buffer).is_some() {
        return Ok(true);
      }
    } else {
      let mut combined = Vec::with_capacity(overlap.len() + buffer.len());
      combined.extend_from_slice(&overlap);
      combined.extend_from_slice(&buffer);
      if finder.find(&combined).is_some() {
        return Ok(true);
      }
    }

    if overlap_len == 0 {
      overlap.clear();
    } else if buffer.len() >= overlap_len {
      overlap = buffer[buffer.len() - overlap_len..].to_vec();
    } else {
      overlap.extend_from_slice(&buffer);
      if overlap.len() > overlap_len {
        overlap = overlap[overlap.len() - overlap_len..].to_vec();
      }
    }
  }
}

fn _align_utf8_boundary(file: &mut File, mut boundary: u64, file_size: u64) -> MyResult<u64> {
  if boundary == 0 || boundary >= file_size {
    return Ok(boundary);
  }
  let mut buffer = [0; 4];
  file.seek(std::io::SeekFrom::Start(boundary))?;
  let bytes_read = file.read(&mut buffer)?;
  for byte in buffer.iter().take(bytes_read) {
    if byte & 0b1100_0000 != 0b1000_0000 {
      return Ok(boundary);
    }
    boundary += 1;
  }
  Ok(boundary.min(file_size))
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
  fn test_unicode_bigram_split_keeps_non_cjk_regex_tokens() {
    let bigrams = parse_unicode_bigrams(&["世界".to_string()]).unwrap();
    let pretokenizer = PreTokenizer::new(&[], None).with_unicode_bigrams(bigrams);
    let tokens = pretokenizer.count_tokens_owned("Hello 世界你好 world").unwrap();
    let expected_tokens = vec![
      ("Hello".to_string(), 1),
      (" ".to_string(), 1),
      ("世界".to_string(), 1),
      ("你".to_string(), 1),
      ("好".to_string(), 1),
      (" world".to_string(), 1),
    ]
    .into_iter()
    .collect::<BTreeMap<_, _>>();
    assert_eq!(tokens, expected_tokens);
  }

  #[test]
  fn test_unicode_bigram_split_is_bigram_agnostic() {
    let bigrams = parse_unicode_bigrams(&["w是".to_string()]).unwrap();
    let pretokenizer = PreTokenizer::new(&[], None).with_unicode_bigrams(bigrams);
    let tokens = pretokenizer.count_tokens_owned("Now是2024年").unwrap();
    let expected_tokens = vec![
      ("N".to_string(), 1),
      ("o".to_string(), 1),
      ("w是".to_string(), 1),
      ("2024".to_string(), 1),
      ("年".to_string(), 1),
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
  fn test_find_chunk_boundaries_falls_back_without_split_token() {
    std::fs::create_dir_all("out/reports/smoke").ok();
    let path = std::path::Path::new("out/reports/smoke/no_split_token.txt");
    std::fs::write(path, "abcdefgh").unwrap();

    let boundaries = _find_chunk_boundaries(path, 4, DEFAULT_EOT).unwrap();

    assert_eq!(boundaries, vec![0, 2, 4, 6, 8]);
  }

  #[test]
  fn test_find_chunk_boundaries_line_mode() {
    std::fs::create_dir_all("out/reports/smoke").ok();
    let path = std::path::Path::new("out/reports/smoke/line_boundaries.txt");
    std::fs::write(path, "aa\nbb\ncc\ndd\n").unwrap();

    let boundaries = _find_chunk_boundaries_with_options(
      path,
      ChunkOptions {
        hint: ChunkHint::Count(4),
        boundary: BoundaryMode::Line,
      },
      DEFAULT_EOT,
    ).unwrap();

    assert_eq!(boundaries, vec![0, 3, 6, 9, 12]);
  }

  #[test]
  fn test_find_chunk_boundaries_chunk_size_hint() {
    std::fs::create_dir_all("out/reports/smoke").ok();
    let path = std::path::Path::new("out/reports/smoke/chunk_size_hint.txt");
    std::fs::write(path, "abcdefghij").unwrap();

    let boundaries = _find_chunk_boundaries_with_options(
      path,
      ChunkOptions {
        hint: ChunkHint::Size(4),
        boundary: BoundaryMode::Utf8,
      },
      DEFAULT_EOT,
    ).unwrap();

    assert_eq!(boundaries, vec![0, 4, 8, 10]);
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
    std::fs::create_dir_all(format!("out/data/{NAME}")).ok();
    serde_json::to_writer_pretty(std::fs::File::create(format!("out/data/{NAME}/_words.json")).unwrap(), &words).unwrap();
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
