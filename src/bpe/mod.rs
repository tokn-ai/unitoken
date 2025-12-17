use std::{collections::{BTreeMap, BTreeSet}, sync::Arc};

pub mod trainer;
pub mod encoder;
pub mod utils;

pub use trainer::BpeTrainer;
pub use encoder::BpeEncoder;
use utils::*;

use ordermap::OrderMap;

pub type Idx = u32;
pub type Word<C> = Arc<[C]>;
pub type Freq = i64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Character {
  Unicode(char),
  Byte(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CharIdx {
  Char(char),
  Idx(Idx),
}

#[derive(Debug)]
pub struct PreToken<C, I> {
  pub src: Word<C>,
  pub idxs: Vec<I>,
  pub freq: Freq,
}

impl<C, I> PreToken<C, I> {
  /// Render a debug string showing the original token and its frequency.
  pub fn display(&self) -> String where Word<C>: WordDebugExt {
    format!("<{:?} => {}>", self.src.debug_display(), self.freq)
  }

  /// Render a debug string by looking up each index in `vocabs`.
  ///
  /// This is mainly useful for inspecting intermediate training states.
  pub fn display_split(&self, vocabs: &BTreeMap<I, Word<C>>) -> String where I: Ord, C: Clone, Word<C>: WordDebugExt {
    let parts = self
      .idxs
      .iter()
      .map(|i| vocabs.get(i).unwrap().debug_display())
      .collect::<Vec<_>>()
      .join(" ");
    format!("<{} => {}>", parts, self.freq)
  }
}

#[derive(Debug)]
pub struct Merge<C, I> {
  pub tp: (I, I),
  pub content: (Word<C>, Word<C>),
  pub target: Option<I>,
  pub data: MergeData,
}

impl<C, I: Clone> Clone for Merge<C, I> {
  fn clone(&self) -> Self {
    Self { tp: self.tp.clone(), content: self.content.clone(), target: self.target.clone(), data: self.data.clone() }
  }
}

impl<C, I> Merge<C, I> {
  /// Concatenate the left and right content and return the merged token.
  pub fn merged_content(&self) -> Word<C> where C: Clone {
    let mut v = Vec::with_capacity(self.content.0.len() + self.content.1.len());
    v.extend_from_slice(&self.content.0);
    v.extend_from_slice(&self.content.1);
    Arc::<[C]>::from(v.into_boxed_slice())
  }

  /// Set the target (new vocab id) for this merge.
  pub fn with_target(mut self, target: I) -> Self {
    self.target = Some(target);
    self
  }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct MergeData {
  pub occurs_in: BTreeSet<u64>,
  pub freq: Freq,
}

impl MergeData {
  /// Create a new [`MergeData`] with the given frequency.
  pub fn new(freq: Freq) -> Self {
    Self {
      occurs_in: BTreeSet::new(),
      freq,
    }
  }

  #[must_use]
  /// Replace the occurrence set with `iter`.
  pub fn add_occurs_in<I: IntoIterator<Item = u64>>(self, iter: I) -> Self {
    Self {
      occurs_in: iter.into_iter().collect(),
      freq: self.freq,
    }
  }

  /// Return `occurs_in` as a `Vec`.
  pub fn occurs_in_vec(&self) -> Vec<u64> {
    self.occurs_in.iter().copied().collect::<Vec<u64>>()
  }
}

impl<C, I> Merge<C, I> {
  /// Create a merge candidate for a pair `(left, right)`.
  pub fn new(tp: (I, I), content: (Word<C>, Word<C>)) -> Self {
    Self {
      tp,
      content,
      target: None,
      data: MergeData::default(),
    }
  }

  /// Record an occurrence of this merge in a document.
  pub fn add(&mut self, doc_id: u64, freq: Freq) {
    self.data.occurs_in.insert(doc_id);
    self.data.freq += freq;
  }

  /// Remove an occurrence of this merge from a document.
  pub fn remove(&mut self, doc_id: &u64, freq: Freq) {
    self.data.freq -= freq;
    self.data.occurs_in.remove(doc_id);
  }
}

pub trait Cachable: std::hash::Hash + Send + Sync + 'static { }
impl<C: std::hash::Hash + Send + Sync + 'static> Cachable for C { }

pub trait IdxLike: Ord + std::hash::Hash + Eq + Copy + Send + Sync + 'static {
  fn from_u64(v: u64) -> Self;
  fn to_u64(self) -> u64;
  fn decode_from_u64(v: u64, start: u64) -> Option<Self> {
    Some(Self::from_u64(v - start))
  }
  fn encode_to_u64(&self, start: u64) -> u64 {
    self.to_u64() + start
  }
}
impl IdxLike for Idx {
  fn from_u64(v: u64) -> Self {
    v as Self
  }
  fn to_u64(self) -> u64 {
    self as u64
  }
}
impl IdxLike for CharIdx {
  fn from_u64(v: u64) -> Self {
    CharIdx::Idx(v as Idx)
  }
  fn to_u64(self) -> u64 {
    match self {
      CharIdx::Idx(i) => i as u64,
      CharIdx::Char(c) => unimplemented!("Cannot convert CharIdx::Char to u64: {:?} [u{:04x}]", c, c as u32),
    }
  }
}

/// Trait to convert a character or byte to an index.
/// This is only used in training, not in encoding.
/// Since it assuming the idx of byte are contiguous.
pub trait CharToIdx<I: IdxLike> {
  /// Convert a character or byte to an index.
  /// If the character is a byte, it will be converted to an index.
  /// If the character is a unicode character, it will be converted to a `CharIdx::Char`.
  fn char_to_idx(&self, start: u64) -> I;
}

impl CharToIdx<Idx> for u8 {
  fn char_to_idx(&self, start: u64) -> Idx {
    (*self as u64 + start) as Idx
  }
}
impl CharToIdx<CharIdx> for char {
  fn char_to_idx(&self, start: u64) -> CharIdx {
    if self.is_ascii() {
      CharIdx::Idx(*self as u8 as Idx + start as Idx)
    } else {
      CharIdx::Char(*self)
    }
  }
}
impl CharToIdx<CharIdx> for u8 {
  fn char_to_idx(&self, start: u64) -> CharIdx {
    CharIdx::Idx((*self as u64 + start) as Idx)
  }
}
impl CharToIdx<CharIdx> for Character {
  fn char_to_idx(&self, start: u64) -> CharIdx {
    match self {
      Character::Unicode(c) => c.char_to_idx(start),
      Character::Byte(b) => b.char_to_idx(start),
    }
  }
}

/// Trait to extract a character from an index or a character.
/// This is only used in training, since CharIdx is only used in training.
///
/// Only [`CharIdx`] would return a character, while [`Idx`] would return `None`.
pub trait HasChar<C>: Sized {
  fn get_char(self) -> Option<char>;
  fn from_char(_c: char) -> Option<Self> { None }
  fn idx_to_word(self) -> Option<Word<C>> where for<'a> &'a str: ToWord<C>{
    self.get_char().map(|i| i.to_string().to_word())
  }
}
impl<C> HasChar<C> for Idx {
  fn get_char(self) -> Option<char> {
    None
  }
}
impl<C> HasChar<C> for char {
  fn get_char(self) -> Option<char> {
    Some(self)
  }
  fn from_char(c: char) -> Option<Self> {
    Some(c)
  }
}
impl<C> HasChar<C> for CharIdx {
  fn get_char(self) -> Option<char> {
    match self {
      CharIdx::Char(c) => Some(c),
      CharIdx::Idx(_) => None,
    }
  }
  fn from_char(c: char) -> Option<Self> {
    Some(CharIdx::Char(c))
  }
}

pub trait CharSplit: Sized {
  /// Split a character into a vector of characters.
  /// This is used to split a character into its constituent parts.
  fn char_split(&self) -> Option<Vec<Self>> {
    None
  }
  fn char_split_u8(&self, buffer: &mut Vec<u8>);
  fn to_vec_u8(w: &Word<Self>) -> Vec<u8> {
    let mut v = Vec::new();
    for c in w.iter() {
      c.char_split_u8(&mut v);
    }
    v
  }
  fn from_vec_u8(v: &[u8]) -> Word<Self>;
}
impl CharSplit for u8 {
  fn char_split_u8(&self, buffer: &mut Vec<u8>) {
    buffer.push(*self);
  }
  fn from_vec_u8(v: &[u8]) -> Word<Self> {
    v.to_word()
  }
}
impl CharSplit for Character {
  fn char_split(&self) -> Option<Vec<Self>> {
    match self {
      Self::Unicode(c) => Some(c.to_string().bytes().into_iter().map(Self::Byte).collect()),
      Self::Byte(_) => None,
    }
  }
  fn char_split_u8(&self, buffer: &mut Vec<u8>) {
    match self {
      Self::Unicode(c) => {
        // TODO: memory allocate
        buffer.extend_from_slice(c.to_string().as_bytes());
      }
      Self::Byte(b) => {
        buffer.push(*b);
      }
    }
  }
  fn from_vec_u8(v: &[u8]) -> Word<Self> {
    _try_combine(v).to_word()
  }
}

fn _try_combine(word: &[u8]) -> Vec<Character> {
  let mut chars = Vec::with_capacity(word.len());
  let mut c = vec![];
  fn convert_str(v: &[u8]) -> Vec<Character> {
    match std::str::from_utf8(v) {
      Ok(s) => s.chars().map(|ch| Character::Unicode(ch)).collect(),
      Err(_) => v.iter().map(|b| Character::Byte(*b)).collect(),
    }
  }
  for &b in word.iter() {
    if b.is_ascii() {
      if !c.is_empty() {
        chars.extend(convert_str(&c));
        c.clear();
      }
      chars.push(Character::Unicode(b as char));
    } else if b < 0b_1100_0000 {
      // 0b_10xx_xxxx means middle byte
      if !c.is_empty() {
        c.push(b);
      } else {
        chars.push(Character::Byte(b));
      }
      continue;
    } else {
      // 0b_110x_xxxx or above means start of a multi-byte character
      if !c.is_empty() {
        chars.extend(convert_str(&c));
        c.clear();
      }
      c.push(b);
    }
  }
  if !c.is_empty() {
    chars.extend(convert_str(&c));
  }
  chars
}
