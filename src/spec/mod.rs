use std::collections::BTreeMap;

use crate::{MyResult, bpe::{Merge, Word}};

pub mod gpt2;
pub mod unitoken;

pub trait Spec<Char, Idx> {
  fn suffix(&self) -> Option<&str> {
    None
  }

  fn encode_vocab(&self, w: &mut dyn std::io::Write, vocab: &BTreeMap<Idx, Word<Char>>) -> MyResult<()>;
  fn decode_vocab(&self, r: &mut dyn std::io::Read) -> MyResult<BTreeMap<Idx, Word<Char>>>;

  fn encode_merges(&self, w: &mut dyn std::io::Write, merges: &Vec<Merge<Char, Idx>>) -> MyResult<()>;
  fn decode_merges_raw(&self, r: &mut dyn std::io::Read) -> MyResult<Vec<Merge<Char, Word<Char>>>>;
  fn decode_merges(&self, r: &mut dyn std::io::Read, vocab: &BTreeMap<Idx, Word<Char>>) -> MyResult<Vec<Merge<Char, Idx>>>;
}

pub trait WordDisplay<C> {
  fn word_display(&self, word: &Word<C>) -> String;
  fn word_parse(&self, s: &str) -> MyResult<Word<C>>;
}
