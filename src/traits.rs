use std::path::Path;

use crate::{MyResult, bpe::{BpeEncoder, BpeTrainer, CharSplit, CharToIdx, Freq, HasChar, Idx, IdxLike, Word, utils::{ToWord, WordDebugExt}}, pretokenizer::PreTokenizer};

/// Training interface for tokenizers that learn from a word-frequency inventory.
pub trait Train {
  /// Create a new trainer with the given special tokens.
  fn new(special_tokens: Vec<String>) -> Self;
  /// Add or replace the word inventory used for training.
  fn add_words(&mut self, words: &mut dyn Iterator<Item = (&str, Freq)>);
  /// Current vocabulary size.
  fn vocab_size(&self) -> usize;
  /// Initialize internal training state (e.g., merge candidate table).
  fn init_training(&mut self);
  /// Perform one training step (e.g., apply the next best merge).
  fn step(&mut self) -> MyResult<()>;

  /// Train until `vocab_size()` reaches `vocab_size`.
  ///
  /// This calls [`Self::init_training`] and then repeatedly calls [`Self::step`].
  fn train(&mut self, vocab_size: usize) -> MyResult<()> {
    self.init_training();
    loop {
      self.step()?;
      if self.vocab_size() >= vocab_size {
        break;
      }
    }
    Ok(())
  }
}

/// Encoding interface for trained tokenizers.
pub trait Encode<I> {
  /// Return the pre-tokenizer used by this encoder.
  fn pre_tokenizer(&self) -> &PreTokenizer;
  /// Encode a single word into token ids.
  fn encode_word(&self, word: &str) -> MyResult<Word<I>>;

  /// Encode multiple words.
  ///
  /// The default implementation calls [`Self::encode_word`] for each word.
  fn encode_words(&self, words: &[&str]) -> MyResult<Vec<Word<I>>> {
    words.iter().map(|w| self.encode_word(w)).collect()
  }
  /// Encode a string (including whitespace and punctuation) into token ids.
  fn encode_string(&self, s: &str) -> MyResult<Vec<I>>;
  /// Encode an entire file into token ids, using `chunks` to parallelize.
  fn encode_file(&self, file: &Path, chunks: usize) -> MyResult<Vec<I>>;
}

/// Decoding interface for trained tokenizers.
pub trait Decode<I> {
  /// Decode token ids back into a UTF-8 string.
  fn decode(&self, idxs: &[I]) -> MyResult<String>;
}

pub trait Encoder<I>: Encode<I> + Decode<I> {}
impl<I, T> Encoder<I> for T where T: Encode<I> + Decode<I> {}

/// Marker trait indicating that `T` can be converted into a `Word<Self>`.
pub trait CanToWord<T>: Sized
where
  Self: Imply<T, Is: ToWord<Self>>,
{}

impl<C, T> CanToWord<T> for C
where
  T: ToWord<C>,
{}

/// Marker trait indicating that `&str` can be converted into a `Word<Self>`.
pub trait CanStrToWord: for<'a> CanToWord<&'a str> {}

impl<C> CanStrToWord for C
where
  for<'a> &'a str: ToWord<C>,
{}

/// Marker trait for types that satisfy the requirements to train a [`BpeTrainer`].
pub trait CanTrain<C, I>
where
  Self: Imply<Word<C>, Is: WordDebugExt>,
  Self: Imply<C, Is: Clone + Ord + Send + Sync + 'static>,
  Self: Imply<C, Is: CharToIdx<I> + CanToWord<u8> + CanStrToWord>,
  Self: Imply<I, Is: IdxLike + HasChar<C>>,
{}

impl<C, I> CanTrain<C, I> for BpeTrainer<C, I>
where
  Word<C>: WordDebugExt,
  for<'a> &'a str: ToWord<C>,
  C: Clone + Ord + Send + Sync + CharToIdx<I> + 'static,
  I: IdxLike + HasChar<C>,
  u8: ToWord<C>,
{}

/// Marker trait for types that satisfy the requirements to encode with a [`BpeEncoder`].
pub trait CanEncode<C, I>
where
  Self: Imply<Word<C>, Is: WordDebugExt>,
  Self: Imply<C, Is: Ord + std::hash::Hash + Clone + Send + Sync + 'static>,
  Self: Imply<C, Is: CharSplit + CanStrToWord>,
  Self: Imply<I, Is: IdxLike>,
{}

impl<C> CanEncode<C, Idx> for BpeEncoder<C>
where
  C: Ord + std::hash::Hash + CharSplit + CanStrToWord + Clone + Send + Sync + 'static,
  Word<C>: WordDebugExt,
{}

// https://docs.rs/imply-hack/latest/imply_hack/
// https://github.com/rust-lang/rust/issues/44491#issuecomment-2496196742
/// Helper trait used to emulate trait aliases on stable Rust.
pub trait Imply<T>: ImplyHack<T, Is = T> {}

impl<T, U> Imply<T> for U {}

/// Implementation detail for [`Imply`].
pub trait ImplyHack<T> {
  type Is;
}

impl<T, U> ImplyHack<T> for U {
  type Is = T;
}
