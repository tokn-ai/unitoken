use ahash::{AHashMap, AHashSet};
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

use crate::{
  MyError, MyResult,
  bigram::VocabBigramIndex,
  bpe::Freq,
};

/// Unicode bigrams retained by a frequency selection and its effective boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnicodeBigramSelection {
  /// Retained bigrams, including every tie at `cutoff_freq`.
  pub bigrams: AHashSet<(char, char)>,
  /// Least frequency among retained bigrams, including all ties.
  pub cutoff_freq: Option<Freq>,
  /// Greatest frequency among counted bigrams that were not retained.
  pub max_excluded_freq: Option<Freq>,
}

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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UnicodeBigramMixedBoundary {
  #[default]
  Keep,
  Split,
}

impl UnicodeBigramMixedBoundary {
  pub fn parse(value: &str) -> MyResult<Self> {
    match value {
      "keep" => Ok(Self::Keep),
      "split" => Ok(Self::Split),
      _ => Err(MyError::SpecError(format!("Unknown unicode bigram mixed boundary mode: {value}"))),
    }
  }
}

#[non_exhaustive]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "py", pyo3::pyclass(from_py_object))]
pub struct PreTokenizer {
  pub re_pat: Regex,
  pub re_special_tokens: Regex,
  pub end_of_text: String,
  pub unicode_bigrams: Option<AHashSet<(char, char)>>,
  pub unicode_bigram_mixed_boundary: UnicodeBigramMixedBoundary,
  pub metrics: bool,
  vocab_bigram_index: VocabBigramIndex,
}

/// A borrowed output from the complete pretokenization pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PreTokenPiece<'a> {
  Special(&'a str),
  Word(&'a str),
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
  /// When `pat_str` is `None`, uses `DEFAULT_PAT`.
  pub fn try_new(
    special_tokens: &[String], end_of_text: Option<&str>, pat_str: Option<&str>,
  ) -> MyResult<Self> {
    let re_pat = match pat_str {
      Some(pat_str) => Regex::new(pat_str)?,
      None => DEFAULT_PAT.clone(),
    };
    let re_special_tokens = create_special_token_regex(special_tokens);
    Ok(Self {
      re_pat,
      re_special_tokens,
      end_of_text: end_of_text.unwrap_or(DEFAULT_EOT).to_string(),
      unicode_bigrams: None,
      unicode_bigram_mixed_boundary: UnicodeBigramMixedBoundary::default(),
      metrics: true,
      vocab_bigram_index: VocabBigramIndex::disabled(),
    })
  }

  pub fn with_unicode_bigrams(mut self, bigrams: AHashSet<(char, char)>) -> Self {
    self.unicode_bigrams = Some(bigrams);
    self
  }

  pub fn with_unicode_bigram_mixed_boundary(mut self, boundary: UnicodeBigramMixedBoundary) -> Self {
    self.unicode_bigram_mixed_boundary = boundary;
    self
  }

  pub(crate) fn with_vocab_bigram_index(mut self, index: VocabBigramIndex) -> Self {
    self.vocab_bigram_index = index;
    self
  }

  /// Visit special tokens and ordinary words in input order.
  ///
  /// Vocab-bigram cuts conservatively partition encoding work. Byte words
  /// shorter than the internal profitability threshold remain PAT-sized.
  pub(crate) fn for_each_piece<'a>(
    &self,
    text: &'a str,
    mut emit: impl FnMut(PreTokenPiece<'a>) -> MyResult<()>,
  ) -> MyResult<()> {
    let mut split_points = Vec::new();
    if self.re_special_tokens.as_str() == "$^" {
      return self.for_each_word(text, &mut split_points, &mut emit);
    }

    let mut last_pos = 0;
    for found in self.re_special_tokens.find_iter(text) {
      let special = found?;
      if special.start() > last_pos {
        self.for_each_word(
          &text[last_pos..special.start()],
          &mut split_points,
          &mut emit,
        )?;
      }
      emit(PreTokenPiece::Special(&text[special.start()..special.end()]))?;
      last_pos = special.end();
    }
    if last_pos < text.len() {
      self.for_each_word(&text[last_pos..], &mut split_points, &mut emit)?;
    }
    Ok(())
  }

  fn for_each_word<'a>(
    &self,
    text: &'a str,
    split_points: &mut Vec<usize>,
    emit: &mut impl FnMut(PreTokenPiece<'a>) -> MyResult<()>,
  ) -> MyResult<()> {
    for_each_pretoken(
      text,
      &self.re_pat,
      self.unicode_bigrams.as_ref(),
      self.unicode_bigram_mixed_boundary,
      |word| self.emit_vocab_bigram_segments(word, split_points, emit),
    )
  }

  fn emit_vocab_bigram_segments<'a>(
    &self,
    word: &'a str,
    split_points: &mut Vec<usize>,
    emit: &mut impl FnMut(PreTokenPiece<'a>) -> MyResult<()>,
  ) -> MyResult<()> {
    debug_assert!(split_points.is_empty());
    self.vocab_bigram_index.split_points(word, split_points);
    if split_points.is_empty() {
      return emit(PreTokenPiece::Word(word));
    }

    let mut start = 0;
    for end in split_points.drain(..) {
      emit(PreTokenPiece::Word(&word[start..end]))?;
      start = end;
    }
    emit(PreTokenPiece::Word(&word[start..]))
  }

  /// Pretokenize a string and return borrowed words with their frequencies.
  ///
  /// Special tokens are excluded. The map keys borrow from `text`.
  pub fn get_words<'a>(&self, text: &'a str) -> MyResult<BTreeMap<&'a str, Freq>> {
    let mut words = BTreeMap::new();
    self.for_each_piece(text, |piece| {
      if let PreTokenPiece::Word(word) = piece {
        *words.entry(word).or_default() += 1;
      }
      Ok(())
    })?;
    Ok(words)
  }

  /// Pretokenize a string and return owned words with their frequencies.
  pub fn get_words_owned(&self, text: &str) -> MyResult<BTreeMap<String, Freq>> {
    let mut words = BTreeMap::new();
    self.for_each_piece(text, |piece| {
      if let PreTokenPiece::Word(word) = piece {
        *words.entry(word.to_string()).or_default() += 1;
      }
      Ok(())
    })?;
    Ok(words)
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
    let mut words = BTreeMap::new();
    self.for_each_piece(&content, |piece| {
      if let PreTokenPiece::Word(word) = piece {
        *words.entry(word.to_string()).or_default() += 1;
      }
      Ok(())
    })?;
    if self.metrics {
      metrics::histogram!("get_words_from_segment.words_count").record(words.len() as f64);
      metrics::counter!("get_words_from_segment.len").increment(len as _);
    }

    trace!(words_len=?words.len(), "result");
    Ok(words)
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
  ) -> MyResult<AHashSet<(char, char)>> {
    Ok(
      self
        .build_unicode_bigram_selection_from_file_with_options(
          path,
          options,
          top_k,
          min_freq,
        )?
        .bigrams,
    )
  }

  /// Count and select Unicode bigrams while preserving the effective cutoff.
  pub fn build_unicode_bigram_selection_from_file_with_options<P: AsRef<Path>>(
    &self, path: P, options: ChunkOptions, top_k: usize, min_freq: Freq,
  ) -> MyResult<UnicodeBigramSelection> {
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
        || AHashMap::new(),
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
  ) -> MyResult<AHashMap<(char, char), Freq>> {
    let buffer = _read_file_to_buffer(&path, offset, len)?;
    let content = String::from_utf8_lossy(&buffer);
    let mut counts = AHashMap::new();
    for_each_regular_chunk(&content, &self.re_special_tokens, |chunk| {
      count_unicode_bigrams(chunk, &mut counts, is_unicode_bigram_script)
    })?;
    Ok(counts)
  }
}

/// Pretokenize a string using `pat` and return word frequencies.
///
/// The returned keys borrow from `s`.
fn _pretokenizer_counter<'a>(s: &'a str, pat: &Regex) -> MyResult<BTreeMap<&'a str, Freq>> {
  let mut result = BTreeMap::new();
  for i in pat.find_iter(s) {
    let token = i?.as_str();
    *result.entry(token).or_default() += 1;
  }
  Ok(result)
}

pub(crate) fn for_each_pretoken<'a>(
  s: &'a str,
  pat: &Regex,
  unicode_bigrams: Option<&AHashSet<(char, char)>>,
  unicode_bigram_mixed_boundary: UnicodeBigramMixedBoundary,
  mut emit: impl FnMut(&'a str) -> MyResult<()>,
) -> MyResult<()> {
  for found in pat.find_iter(s) {
    let token = found?.as_str();
    if let Some(unicode_bigrams) = unicode_bigrams {
      for_each_unicode_bigram_segment(
        token,
        unicode_bigrams,
        unicode_bigram_mixed_boundary,
        &mut emit,
      )?;
    } else {
      emit(token)?;
    }
  }
  Ok(())
}

pub fn parse_unicode_bigrams(bigrams: &[String]) -> MyResult<AHashSet<(char, char)>> {
  let mut parsed = AHashSet::new();
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

pub(crate) fn select_unicode_bigrams(
  counts: AHashMap<(char, char), Freq>, top_k: usize, min_freq: Freq,
) -> UnicodeBigramSelection {
  if top_k == 0 {
    return UnicodeBigramSelection {
      bigrams: AHashSet::new(),
      cutoff_freq: None,
      max_excluded_freq: counts.values().copied().max(),
    };
  }
  let mut max_below_min_freq = None;
  let mut sorted = Vec::new();
  for (bigram, freq) in counts {
    if freq >= min_freq {
      sorted.push((bigram, freq));
    } else if max_below_min_freq.is_none_or(|current| freq > current) {
      max_below_min_freq = Some(freq);
    }
  }
  sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
  let Some(cutoff_freq) = sorted
    .get(top_k.min(sorted.len()).saturating_sub(1))
    .map(|(_, freq)| *freq)
  else {
    return UnicodeBigramSelection {
      bigrams: AHashSet::new(),
      cutoff_freq: None,
      max_excluded_freq: max_below_min_freq,
    };
  };
  let retained_len = sorted.partition_point(|(_, freq)| *freq >= cutoff_freq);
  let max_excluded_freq = sorted
    .get(retained_len)
    .map(|(_, freq)| *freq)
    .or(max_below_min_freq);
  UnicodeBigramSelection {
    bigrams: sorted
      .into_iter()
      .take(retained_len)
      .map(|(bigram, _)| bigram)
      .collect(),
    cutoff_freq: Some(cutoff_freq),
    max_excluded_freq,
  }
}

pub(crate) fn count_unicode_bigrams(
  token: &str, counts: &mut AHashMap<(char, char), Freq>, keep_char: impl Fn(char) -> bool,
) -> MyResult<()> {
  let mut chars = token.chars();
  let Some(mut prev) = chars.next() else {
    return Ok(());
  };
  for next in chars {
    if keep_char(prev) && keep_char(next) {
      let count = counts.entry((prev, next)).or_default();
      *count = count.checked_add(1).ok_or(MyError::FrequencyOverflow)?;
    }
    prev = next;
  }
  Ok(())
}

fn for_each_unicode_bigram_segment<'a>(
  token: &'a str,
  unicode_bigrams: &AHashSet<(char, char)>,
  unicode_bigram_mixed_boundary: UnicodeBigramMixedBoundary,
  emit: &mut impl FnMut(&'a str) -> MyResult<()>,
) -> MyResult<()> {
  let mut chars = token.char_indices();
  let Some((_, mut left)) = chars.next() else {
    return emit(token);
  };
  let mut start = 0;
  for (right_byte, right) in chars {
    if should_split_unicode_bigram_mixed_boundary(left, right, unicode_bigrams, unicode_bigram_mixed_boundary) {
      if start < right_byte {
        emit(&token[start..right_byte])?;
      }
      start = right_byte;
    }
    left = right;
  }
  if start < token.len() {
    emit(&token[start..])?;
  }
  Ok(())
}

fn should_split_unicode_bigram_mixed_boundary(
  left: char,
  right: char,
  unicode_bigrams: &AHashSet<(char, char)>,
  unicode_bigram_mixed_boundary: UnicodeBigramMixedBoundary,
) -> bool {
  if unicode_bigrams.contains(&(left, right)) {
    return false;
  }
  match (is_unicode_bigram_script(left), is_unicode_bigram_script(right)) {
    (true, true) => true,
    (true, false) | (false, true) => unicode_bigram_mixed_boundary == UnicodeBigramMixedBoundary::Split,
    (false, false) => false,
  }
}

pub(crate) fn is_unicode_bigram_script(ch: char) -> bool {
  matches!(
    ch as u32,
    // CJK Unified Ideographs Extension A.
    0x3400..=0x4DBF
      // CJK Unified Ideographs.
      | 0x4E00..=0x9FFF
      // CJK Compatibility Ideographs.
      | 0xF900..=0xFAFF
      // Hiragana.
      | 0x3040..=0x309F
      // Katakana.
      | 0x30A0..=0x30FF
      // Hangul Jamo.
      | 0x1100..=0x11FF
      // Hangul Compatibility Jamo.
      | 0x3130..=0x318F
      // Hangul Jamo Extended-A.
      | 0xA960..=0xA97F
      // Hangul Syllables.
      | 0xAC00..=0xD7AF
      // Hangul Jamo Extended-B.
      | 0xD7B0..=0xD7FF
      // Thai.
      | 0x0E00..=0x0E7F
      // Lao.
      | 0x0E80..=0x0EFF
      // Khmer.
      | 0x1780..=0x17FF
      // Myanmar.
      | 0x1000..=0x109F
      // Myanmar Extended-A.
      | 0xAA60..=0xAA7F
      // Myanmar Extended-B.
      | 0xA9E0..=0xA9FF
      // CJK Unified Ideographs Extension B.
      | 0x20000..=0x2A6DF
      // CJK Unified Ideographs Extension C.
      | 0x2A700..=0x2B73F
      // CJK Unified Ideographs Extension D.
      | 0x2B740..=0x2B81F
      // CJK Unified Ideographs Extension E and F.
      | 0x2B820..=0x2CEAF
      // CJK Unified Ideographs Extension I.
      | 0x2CEB0..=0x2EBEF
      // CJK Unified Ideographs Extension G and H.
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

pub(crate) fn for_each_regular_chunk<'a>(
  text: &'a str,
  special_tokens: &Regex,
  mut emit: impl FnMut(&'a str) -> MyResult<()>,
) -> MyResult<()> {
  let mut last_pos = 0;
  for found in special_tokens.find_iter(text) {
    let special = found?;
    if special.start() > last_pos {
      emit(&text[last_pos..special.start()])?;
    }
    last_pos = special.end();
  }
  if last_pos < text.len() {
    emit(&text[last_pos..])?;
  }
  Ok(())
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
  use crate::bigram::Bigram;

  #[test]
  fn test_piece_pipeline_applies_special_pat_unicode_and_vocab_splits() {
    let unicode_bigrams = parse_unicode_bigrams(&["你好".to_string()]).unwrap();
    let vocab_bigrams = [
      Bigram::new('你', '好'),
      Bigram::new('a', 'b'),
      Bigram::new('b', 'c'),
    ]
    .into_iter()
    .collect();
    let pre_tokenizer = PreTokenizer::try_new(
      &["<eot>".to_string()],
      Some("<eot>"),
      Some(r"\p{L}+"),
    )
    .unwrap()
    .with_unicode_bigrams(unicode_bigrams)
    .with_vocab_bigram_index(VocabBigramIndex::unicode(vocab_bigrams));

    let mut pieces = Vec::new();
    pre_tokenizer
      .for_each_piece("你好世界<eot>abcz", |piece| {
        pieces.push(match piece {
          PreTokenPiece::Special(special) => ("special", special),
          PreTokenPiece::Word(word) => ("word", word),
        });
        Ok(())
      })
      .unwrap();

    assert_eq!(
      pieces,
      [
        ("word", "你好"),
        ("word", "世"),
        ("word", "界"),
        ("special", "<eot>"),
        ("word", "abc"),
        ("word", "z"),
      ],
    );
    assert_eq!(
      pre_tokenizer.get_words("你好世界<eot>abcz").unwrap(),
      [
        ("abc", 1),
        ("z", 1),
        ("世", 1),
        ("你好", 1),
        ("界", 1),
      ]
      .into_iter()
      .collect(),
    );
    assert_eq!(
      pre_tokenizer.get_words_owned("你好世界<eot>abcz").unwrap(),
      [
        ("abc".to_string(), 1),
        ("z".to_string(), 1),
        ("世".to_string(), 1),
        ("你好".to_string(), 1),
        ("界".to_string(), 1),
      ]
      .into_iter()
      .collect(),
    );

    std::fs::create_dir_all("out/reports/smoke").ok();
    let path = std::path::Path::new("out/reports/smoke/pretokenizer_piece_pipeline.txt");
    let input = "你好世界<eot>abcz";
    std::fs::write(path, input).unwrap();
    assert_eq!(
      pre_tokenizer.get_words_from_segment(path, 0, input.len()).unwrap(),
      pre_tokenizer.get_words_owned(input).unwrap(),
    );
  }

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
    let tokens = pretokenizer.get_words_owned("Hello 世界你好 world").unwrap();
    let expected_tokens = vec![
      ("Hello".to_string(), 1),
      (" 世界".to_string(), 1),
      ("你".to_string(), 1),
      ("好".to_string(), 1),
      (" world".to_string(), 1),
    ]
    .into_iter()
    .collect::<BTreeMap<_, _>>();
    assert_eq!(tokens, expected_tokens);
  }

  #[test]
  fn test_unicode_bigram_split_keeps_mixed_edges_by_default() {
    let bigrams = parse_unicode_bigrams(&["w是".to_string()]).unwrap();
    let pretokenizer = PreTokenizer::new(&[], None).with_unicode_bigrams(bigrams);
    let tokens = pretokenizer.get_words_owned("Now是2024年").unwrap();
    let expected_tokens = vec![
      ("Now是".to_string(), 1),
      ("2024".to_string(), 1),
      ("年".to_string(), 1),
    ]
    .into_iter()
    .collect::<BTreeMap<_, _>>();
    assert_eq!(tokens, expected_tokens);
  }

  #[test]
  fn test_unicode_bigram_split_can_split_mixed_edges() {
    let bigrams = parse_unicode_bigrams(&["世界".to_string()]).unwrap();
    let pretokenizer = PreTokenizer::new(&[], None)
      .with_unicode_bigrams(bigrams)
      .with_unicode_bigram_mixed_boundary(UnicodeBigramMixedBoundary::Split);
    let tokens = pretokenizer.get_words_owned("er世界").unwrap();
    let expected_tokens = vec![
      ("er".to_string(), 1),
      ("世界".to_string(), 1),
    ]
    .into_iter()
    .collect::<BTreeMap<_, _>>();
    assert_eq!(tokens, expected_tokens);
  }

  #[test]
  fn test_unicode_bigram_split_keeps_mixed_edges() {
    let bigrams = parse_unicode_bigrams(&["世界".to_string()]).unwrap();
    let pretokenizer = PreTokenizer::new(&[], None).with_unicode_bigrams(bigrams);
    let tokens = pretokenizer.get_words_owned("er世界").unwrap();
    let expected_tokens = vec![
      ("er世界".to_string(), 1),
    ]
    .into_iter()
    .collect::<BTreeMap<_, _>>();
    assert_eq!(tokens, expected_tokens);
  }

  #[test]
  fn test_unicode_bigram_split_cuts_unretained_script_edges() {
    let bigrams = parse_unicode_bigrams(&["世界".to_string()]).unwrap();
    let pretokenizer = PreTokenizer::new(&[], None).with_unicode_bigrams(bigrams);
    let tokens = pretokenizer.get_words_owned("你好世界").unwrap();
    let expected_tokens = vec![
      ("你".to_string(), 1),
      ("好".to_string(), 1),
      ("世界".to_string(), 1),
    ]
    .into_iter()
    .collect::<BTreeMap<_, _>>();
    assert_eq!(tokens, expected_tokens);
  }

  #[test]
  fn test_unicode_bigram_count_scans_raw_text_without_crossing_eot() {
    std::fs::create_dir_all("out/reports/smoke").ok();
    let path = std::path::Path::new("out/reports/smoke/unicode_bigram_raw.txt");
    std::fs::write(path, format!("ab你{DEFAULT_EOT}bc你好한글かな")).unwrap();

    let pretokenizer = PreTokenizer::new(&vec![DEFAULT_EOT.to_string()], Some(DEFAULT_EOT));
    let selection = pretokenizer
      .build_unicode_bigram_selection_from_file_with_options(
        path,
        ChunkOptions {
          hint: ChunkHint::Count(1),
          boundary: BoundaryMode::Utf8,
        },
        16,
        1,
      )
      .unwrap();
    let bigrams = &selection.bigrams;

    assert!(bigrams.contains(&('你', '好')));
    assert!(bigrams.contains(&('한', '글')));
    assert!(bigrams.contains(&('か', 'な')));
    assert!(!bigrams.contains(&('a', 'b')));
    assert!(!bigrams.contains(&('b', 'c')));
    assert!(!bigrams.contains(&('b', '你')));
    assert!(!bigrams.contains(&('c', '你')));
    assert!(!bigrams.contains(&('b', '<')));
    assert!(!bigrams.contains(&('>', 'b')));
    assert_eq!(selection.cutoff_freq, Some(1));
    assert_eq!(selection.max_excluded_freq, None);
  }

  #[test]
  fn test_select_unicode_bigrams_includes_cutoff_frequency_ties() {
    let counts = [
      (('你', '好'), 10),
      (('世', '界'), 5),
      (('한', '글'), 5),
      (('か', 'な'), 4),
    ]
    .into_iter()
    .collect::<AHashMap<_, _>>();

    let selection = select_unicode_bigrams(counts, 2, 1);

    assert_eq!(
      selection.bigrams,
      [('你', '好'), ('世', '界'), ('한', '글')]
        .into_iter()
        .collect::<AHashSet<_>>()
    );
    assert_eq!(selection.cutoff_freq, Some(5));
    assert_eq!(selection.max_excluded_freq, Some(4));
  }

  #[test]
  fn test_unicode_bigram_selection_reports_underfilled_and_empty_cutoffs() {
    let counts = [
      (('你', '好'), 10),
      (('世', '界'), 5),
      (('한', '글'), 4),
    ]
    .into_iter()
    .collect::<AHashMap<_, _>>();

    let underfilled = select_unicode_bigrams(counts.clone(), 10, 5);
    assert_eq!(underfilled.cutoff_freq, Some(5));
    assert_eq!(underfilled.max_excluded_freq, Some(4));
    assert_eq!(underfilled.bigrams.len(), 2);

    let filtered = select_unicode_bigrams(counts.clone(), 10, 11);
    assert_eq!(filtered.cutoff_freq, None);
    assert_eq!(filtered.max_excluded_freq, Some(10));
    assert!(filtered.bigrams.is_empty());

    let zero = select_unicode_bigrams(counts, 0, 1);
    assert_eq!(zero.cutoff_freq, None);
    assert_eq!(zero.max_excluded_freq, Some(10));
    assert!(zero.bigrams.is_empty());
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
    let pat_str = r"[^\s]";
    let t = PreTokenizer::try_new(&vec![DEFAULT_EOT.to_string()], Some(DEFAULT_EOT), Some(pat_str)).unwrap();

    let s = "ab cd";
    let counts = t.get_words(s).unwrap();
    assert_eq!(counts.get("a").cloned().unwrap_or(0), 1);
    assert_eq!(counts.get("b").cloned().unwrap_or(0), 1);
    assert_eq!(counts.get("c").cloned().unwrap_or(0), 1);
    assert_eq!(counts.get("d").cloned().unwrap_or(0), 1);
  }
}
