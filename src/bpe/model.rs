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

  pub(crate) fn into_parts(
    self,
  ) -> (Vec<String>, BTreeMap<I, Word<C>>, Vec<Merge<C, I>>) {
    (self.special_tokens, self.vocab, self.merges)
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

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{bpe::{BpeEncoder, Character, Idx, utils::ToWord}, spec::unitoken::UnitokenSpec, traits::Encode};

  #[test]
  fn unicode_fallback_merge_serialization_preserves_dependency_order() {
    let vocab = BTreeMap::from([
      (0, vec![Character::Byte(0xe4)].to_word()),
      (1, vec![Character::Byte(0xbd)].to_word()),
      (2, vec![Character::Byte(0xa0)].to_word()),
      (3, vec![Character::Byte(0xe4), Character::Byte(0xbd)].to_word()),
      (4, "你".to_word()),
    ]);
    let mut prefix = Merge::new(
      (0 as Idx, 1 as Idx),
      (
        vec![Character::Byte(0xe4)].to_word(),
        vec![Character::Byte(0xbd)].to_word(),
      ),
    ).with_target(3);
    prefix.data.freq = 11;
    let mut scalar = Merge::new(
      (3 as Idx, 2 as Idx),
      (
        vec![Character::Byte(0xe4), Character::Byte(0xbd)].to_word(),
        vec![Character::Byte(0xa0)].to_word(),
      ),
    ).with_target(4);
    scalar.data.freq = 7;
    let model = BpeModel::new(vec![], vocab, vec![prefix, scalar]);

    assert_eq!(model.merges()[0].target, Some(3));
    assert_eq!(model.merges()[1].target, Some(4));
    assert_eq!(
      model.merges()[1].canonical_merged_content().as_ref(),
      [Character::Unicode('你')],
    );

    let mut serialized = Vec::new();
    model.save_merges_txt(&UnitokenSpec, &mut serialized).unwrap();
    assert_eq!(
      String::from_utf8(serialized).unwrap(),
      "{xe4} {xbd} => 11\n{xe4}{xbd} {xa0} => 7\n",
    );

    let mut serialized_vocab = Vec::new();
    let mut serialized_merges = Vec::new();
    model.save_vocab_json(&UnitokenSpec, &mut serialized_vocab).unwrap();
    model.save_merges_txt(&UnitokenSpec, &mut serialized_merges).unwrap();
    let decoded_vocab = <UnitokenSpec as Spec<Character, Idx>>::decode_vocab(
      &UnitokenSpec,
      &mut serialized_vocab.as_slice(),
    ).unwrap();
    let decoded_merges = <UnitokenSpec as Spec<Character, Idx>>::decode_merges(
      &UnitokenSpec,
      &mut serialized_merges.as_slice(),
      &decoded_vocab,
    ).unwrap();
    assert_eq!(
      decoded_merges
        .iter()
        .map(|merge| merge.target)
        .collect::<Vec<_>>(),
      [Some(3), Some(4)],
    );

    let encoder = BpeEncoder::new(
      decoded_vocab,
      decoded_merges.iter().map(|merge| {
        (merge.tp, merge.target.unwrap())
      }).collect(),
      vec![],
    ).unwrap();
    assert_eq!(encoder.encode_word("你").unwrap().as_ref(), [4]);
  }
}
