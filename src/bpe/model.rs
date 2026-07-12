use std::collections::BTreeMap;

use crate::{MyResult, spec::Spec};

use super::{Freq, Merge, Word};

/// An immutable BPE model produced by validating a trainer snapshot.
#[derive(Debug)]
pub struct BpeModel<C, I> {
  special_tokens: Vec<String>,
  vocab: BTreeMap<I, Word<C>>,
  merges: Vec<Merge<C, I>>,
}

impl<C, I> BpeModel<C, I> {
  pub(crate) fn new(
    special_tokens: Vec<String>,
    vocab: BTreeMap<I, Word<C>>,
    merges: Vec<Merge<C, I>>,
  ) -> Self {
    Self {
      special_tokens,
      vocab,
      merges,
    }
  }

  /// Reserved special tokens in vocabulary order.
  pub fn special_tokens(&self) -> &[String] {
    &self.special_tokens
  }

  /// Validated token-id vocabulary.
  pub fn vocab(&self) -> &BTreeMap<I, Word<C>> {
    &self.vocab
  }

  /// Validated merge rules in rank order.
  pub fn merges(&self) -> &[Merge<C, I>] {
    &self.merges
  }

  /// Frequency of the final pair merge, if the model contains one.
  pub fn last_merge_freq(&self) -> Option<Freq> {
    self.merges.last().map(|merge| merge.data.freq)
  }

  /// Serialize the vocabulary to JSON using `spec`.
  pub fn save_vocab_json<W: std::io::Write>(&self, spec: &dyn Spec<C, I>, mut writer: W) -> MyResult<()> {
    spec.encode_vocab(&mut writer, &self.vocab)
  }

  /// Serialize the merge list to text using `spec`.
  pub fn save_merges_txt<W: std::io::Write>(&self, spec: &dyn Spec<C, I>, mut writer: W) -> MyResult<()> {
    spec.encode_merges(&mut writer, &self.merges)
  }
}
