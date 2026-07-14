//! Ordered source-unit bigrams used by encoding.

use std::{fmt, sync::Arc};

use ahash::AHashSet;

const BYTE_BIGRAM_WORDS: usize = (u16::MAX as usize + 1) / u64::BITS as usize;
// English regex pretokens are usually short enough that scanning costs more
// than the merge work it avoids. Long byte tokens still benefit from cutting.
const MIN_BYTE_BIGRAM_SPLIT_BYTES: usize = 32;

/// An ordered pair of adjacent units.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Bigram<T> {
  /// Left unit in the pair.
  pub left: T,
  /// Right unit in the pair.
  pub right: T,
}

impl<T> Bigram<T> {
  /// Create an ordered bigram from its left and right units.
  #[must_use]
  pub const fn new(left: T, right: T) -> Self {
    Self { left, right }
  }
}

impl<T> From<(T, T)> for Bigram<T> {
  fn from((left, right): (T, T)) -> Self {
    Self::new(left, right)
  }
}

impl<T> From<Bigram<T>> for (T, T) {
  fn from(bigram: Bigram<T>) -> Self {
    (bigram.left, bigram.right)
  }
}

#[derive(Clone, Debug)]
struct ByteBigramIndex {
  // The complete u8 × u8 domain fits in an 8 KiB bitset.
  bits: Box<[u64; BYTE_BIGRAM_WORDS]>,
}

impl ByteBigramIndex {
  fn new() -> Self {
    Self {
      bits: Box::new([0; BYTE_BIGRAM_WORDS]),
    }
  }

  fn position(bigram: Bigram<u8>) -> (usize, u64) {
    let key = (usize::from(bigram.left) << u8::BITS) | usize::from(bigram.right);
    (key / u64::BITS as usize, 1 << (key % u64::BITS as usize))
  }

  fn insert(&mut self, bigram: Bigram<u8>) {
    let (word, mask) = Self::position(bigram);
    self.bits[word] |= mask;
  }

  fn contains(&self, bigram: Bigram<u8>) -> bool {
    let (word, mask) = Self::position(bigram);
    self.bits[word] & mask != 0
  }
}

#[derive(Clone)]
enum VocabBigramIndexInner {
  Byte(ByteBigramIndex),
  Unicode(AHashSet<Bigram<char>>),
  Disabled,
}

#[doc(hidden)]
#[derive(Clone)]
pub struct VocabBigramIndex(Arc<VocabBigramIndexInner>);

impl fmt::Debug for VocabBigramIndex {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self.0.as_ref() {
      VocabBigramIndexInner::Byte(_) => f.write_str("VocabBigramIndex::Byte"),
      VocabBigramIndexInner::Unicode(bigrams) => f
        .debug_tuple("VocabBigramIndex::Unicode")
        .field(&bigrams.len())
        .finish(),
      VocabBigramIndexInner::Disabled => f.write_str("VocabBigramIndex::Disabled"),
    }
  }
}

impl VocabBigramIndex {
  pub(crate) fn byte() -> Self {
    Self(Arc::new(VocabBigramIndexInner::Byte(ByteBigramIndex::new())))
  }

  pub(crate) fn insert_byte(&mut self, bigram: Bigram<u8>) {
    let VocabBigramIndexInner::Byte(index) = Arc::make_mut(&mut self.0) else {
      unreachable!("only byte indexes accept byte bigrams");
    };
    index.insert(bigram);
  }

  pub(crate) fn unicode(bigrams: AHashSet<Bigram<char>>) -> Self {
    Self(Arc::new(VocabBigramIndexInner::Unicode(bigrams)))
  }

  pub(crate) fn disabled() -> Self {
    Self(Arc::new(VocabBigramIndexInner::Disabled))
  }

  #[cfg(test)]
  pub(crate) fn contains_byte(&self, bigram: Bigram<u8>) -> bool {
    let VocabBigramIndexInner::Byte(index) = self.0.as_ref() else {
      return false;
    };
    index.contains(bigram)
  }

  #[cfg(test)]
  pub(crate) fn unicode_bigrams(&self) -> Option<&AHashSet<Bigram<char>>> {
    let VocabBigramIndexInner::Unicode(bigrams) = self.0.as_ref() else {
      return None;
    };
    Some(bigrams)
  }

  pub(crate) fn should_split(&self, input: &str) -> bool {
    match self.0.as_ref() {
      VocabBigramIndexInner::Byte(_) => input.len() >= MIN_BYTE_BIGRAM_SPLIT_BYTES,
      VocabBigramIndexInner::Unicode(_) => input.char_indices().nth(1).is_some(),
      VocabBigramIndexInner::Disabled => false,
    }
  }

  pub(crate) fn split_points(&self, input: &str, output: &mut Vec<usize>) {
    debug_assert!(output.is_empty());
    if !self.should_split(input) {
      return;
    }
    match self.0.as_ref() {
      VocabBigramIndexInner::Byte(bigrams) => {
        let bytes = input.as_bytes();
        for right in 1..bytes.len() {
          if input.is_char_boundary(right)
            && !bigrams.contains(Bigram::new(bytes[right - 1], bytes[right]))
          {
            output.push(right);
          }
        }
      }
      VocabBigramIndexInner::Unicode(bigrams) => {
        let mut chars = input.char_indices();
        let Some((_, mut left)) = chars.next() else {
          return;
        };
        for (right_byte, right) in chars {
          if !bigrams.contains(&Bigram::new(left, right)) {
            output.push(right_byte);
          }
          left = right;
        }
      }
      VocabBigramIndexInner::Disabled => unreachable!("disabled indexes cannot split input"),
    }
  }
}
