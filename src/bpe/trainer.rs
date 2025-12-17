use std::{collections::{BTreeMap, HashMap}, sync::atomic::AtomicU64};

use crate::{MyError, MyResult, spec::Spec, traits::{CanStrToWord, CanToWord, CanTrain, Train}};

use super::*;

#[derive(Debug, Default)]
pub struct BpeTrainer<C, I> {
  pub start_vocab_idx: AtomicU64,
  pub _byte_vocab_start_idx: Option<u64>,
  pub special_tokens: Vec<String>,
  pub vocab: BTreeMap<I, Word<C>>,
  pub merges: Vec<Merge<C, I>>,
  pub pre_merges: HashMap<(I, I), Merge<C, I>>,
  pub words: Vec<PreToken<C, I>>,
}

impl<C, I: IdxLike> BpeTrainer<C, I>
where
  Word<C>: WordDebugExt,
  C: CanStrToWord + CanToWord<u8>,
{
  /// Build a trainer from `(word, frequency)` pairs.
  ///
  /// - `special_tokens` are reserved at the start of the vocabulary.
  /// - Words equal to any special token are skipped.
  pub fn from_words<Iter: IntoIterator<Item = (S, Freq)>, S: AsRef<str>>(words: Iter, special_tokens: &[String]) -> Self
  where
    C: CharToIdx<I>,
    I: HasChar<C>,
  {
    let vocab_start_idx = special_tokens.len() as u64;
    let sp_set = special_tokens.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let tokens = Self::_words_to_tokens(words, vocab_start_idx, &sp_set);
    Self::new(tokens, special_tokens.to_vec())
  }

  /// Create a trainer from already pre-tokenized words.
  ///
  /// This initializes vocab with `special_tokens` and a 256-entry byte vocabulary.
  pub fn new(words: Vec<PreToken<C, I>>, special_tokens: Vec<String>) -> Self {
    let mut bpe = Self::empty();
    bpe._vocab_insert_special_tokens(special_tokens);
    bpe._vocab_insert_all_single_byte();
    bpe.words = words;
    bpe
  }

  /// Insert the full single-byte vocabulary (0..=255) into `self.vocab`.
  ///
  /// Returns the next available vocab index.
  pub fn _vocab_insert_all_single_byte(&mut self) -> I {
    let start_idx = self.start_vocab_idx.fetch_add(256, std::sync::atomic::Ordering::AcqRel);
    let vocab = &mut self.vocab;
    for i in 0u8..128 {
      vocab.insert(I::from_u64(i as u64 + start_idx), (i as char).to_string().to_word());
    }
    for i in 128u8..=255 {
      vocab.insert(I::from_u64(i as u64 + start_idx), i.to_word());
    }
    self._byte_vocab_start_idx = Some(start_idx);
    I::from_u64(start_idx + 256)
  }

  /// Convert `(word, frequency)` input into [`PreToken`]s.
  ///
  /// Words that match `special_tokens` are skipped.
  pub fn _words_to_tokens<Iter: IntoIterator<Item = (S, Freq)>, S: AsRef<str>>(words: Iter, vocab_start_idx: u64, special_tokens: &BTreeSet<&str>) -> Vec<PreToken<C, I>>
  where
    C: CharToIdx<I>,
  {
    let mut tokens = Vec::new();
    for (w, freq) in words.into_iter() {
      let w = w.as_ref();
      if special_tokens.contains(w) {
        continue;
      }
      let src = w.to_word();
      let idxs = src.iter().map(|b| b.char_to_idx(vocab_start_idx)).collect::<Vec<_>>();
      let pre_token = PreToken {
        src: src.clone(),
        idxs,
        freq: freq as Freq,
      };
      tokens.push(pre_token);
    }
    tokens
  }
}

impl<C: CanStrToWord, I: IdxLike> BpeTrainer<C, I>
where
  Word<C>: WordDebugExt,
{
  /// Insert special tokens at the start of the vocabulary.
  ///
  /// Returns the next available vocab index.
  pub fn _vocab_insert_special_tokens(&mut self, special_tokens: Vec<String>) -> I {
    let length = special_tokens.len();
    let start_idx = self.start_vocab_idx.fetch_add(length as u64, std::sync::atomic::Ordering::AcqRel);
    let vocab = &mut self.vocab;
    for (i, token) in special_tokens.iter().enumerate() {
      vocab.insert(I::from_u64(i as u64 + start_idx), token.as_str().to_word());
    }
    self.special_tokens.extend(special_tokens);
    I::from_u64(start_idx + length as u64)
  }

  /// Serialize the current vocabulary to JSON using `spec`.
  pub fn save_vocab_json<W: std::io::Write>(&self, spec: &dyn Spec<C, I>, mut w: W) -> MyResult<()> {
    spec.encode_vocab(&mut w, &self.vocab)
  }

  /// Serialize the current merge list to a text format using `spec`.
  pub fn save_merges_txt<W: std::io::Write>(&self, spec: &dyn Spec<C, I>, mut w: W) -> MyResult<()> {
    spec.encode_merges(&mut w, &self.merges)
  }
}

impl<C, I> BpeTrainer<C, I> {
  /// Construct an empty trainer with no vocab, merges, or words.
  pub fn empty() -> Self {
    Self {
      start_vocab_idx: AtomicU64::new(0),
      _byte_vocab_start_idx: None,
      vocab: BTreeMap::new(),
      merges: Vec::new(),
      pre_merges: HashMap::new(),
      special_tokens: Vec::new(),
      words: Vec::new(),
    }
  }
}

impl<C, I: IdxLike> BpeTrainer<C, I>
where
  Word<C>: WordDebugExt,
  I: HasChar<C>,
  C: CanStrToWord,
{
  /// Initialize the merge candidate map from `self.words`.
  ///
  /// This computes merge frequencies and document-occurrence sets used by [`Train::step`].
  pub fn _build_pre_merges(&mut self) {
    debug!("Initializing BPE training with {} words", self.words.len());
    self.pre_merges.clear();
    let vocab_get = |i: I| {
      self.vocab.get(&i).cloned().or_else(|| i.idx_to_word()).ok_or_else(|| MyError::OovIdx(i.to_u64()))
    };
    let i_none = I::from_u64(u64::MAX);
    let w_none = char::from_u32(0x10FFFF).unwrap().to_string().to_word();
    for (i, word) in self.words.iter().enumerate() {
      // for single char tokens
      // this for loop should takes no effects then <C=u8>.
      // note: all idx in vocab should be CharIdx::Idx
      for j in word.idxs.iter() {
        // ascii chars, j should be CharIdx::Idx
        if self.vocab.contains_key(j) {
          continue;
        }
        // for unicode, j should be CharIdx::Char
        let tp = (i_none, *j);
        let merge = self.pre_merges.entry(tp).or_insert_with(|| {
          let content = (
            // w_none goes 0x10FFFF, which should have precedence over any valid pair when sort. (lexicographically largest)
            w_none.clone(),
            vocab_get(*j).unwrap(),
          );
          // set merge.target = Some(j) to indicate this is a single char token.
          // see also [`Self::step`]
          Merge::new(tp, content).with_target(*j)
        });
        merge.data.freq += word.freq;
      }
      for (j1, j2) in word.idxs.iter().copied().zip(word.idxs.iter().skip(1).copied()) {
        let tp = (j1, j2);
        let merge = self.pre_merges.entry(tp).or_insert_with(|| {
          let content = (
            vocab_get(j1).unwrap(),
            vocab_get(j2).unwrap(),
          );
          Merge::new(tp, content)
        });
        merge.add(i as u64, word.freq);
      }
    }
  }

  fn _set_vocab_idx(&mut self, start_idx: I) {
    self.start_vocab_idx.store(start_idx.to_u64(), std::sync::atomic::Ordering::Release);
  }

  fn _add_vocab_idx(&self) -> I {
    I::from_u64(self.start_vocab_idx.fetch_add(1, std::sync::atomic::Ordering::AcqRel))
  }

  fn update_pre_merges(&mut self, merge: &Merge<C, I>, changes: BTreeMap<(I, I), MergeData>) {
    _update_merge_map(&mut self.pre_merges, merge, changes, Some(&self.vocab));
  }

  fn merge(&mut self, merge: &Merge<C, I>, target_idx: I) -> BTreeMap<(I, I), MergeData> {
    _merge(&mut self.words, merge, target_idx)
  }

  fn _get_largest_merge(&self) -> Option<Merge<C, I>> where C: Ord {
    self
      .pre_merges
      .values()
      .max_by_key(|m| (m.data.freq, &m.content))
      .cloned()
  }

  fn _get_largest_merge2(&self) -> Option<Merge<C, I>> where C: Ord + Send + Sync + 'static {
    use rayon::prelude::*;
    self
      .pre_merges
      .par_iter()
      .map(|(_, m)| m)
      .max_by_key(|m| (m.data.freq, &m.content))
      .cloned()
  }

  /// Apply one merge operation and return the newly assigned vocab index.
  ///
  /// This is the core training step once a merge candidate has been selected.
  pub fn _step(&mut self, merge: Merge<C, I>) -> I where C: Clone {
    let target_idx = self._add_vocab_idx();
    // if target = Some(j), this is a single char token, no need to merge.
    // but we have to add it to vocab.
    if merge.target.is_some() {
      self.vocab.insert(target_idx, merge.content.1.clone());
      self.pre_merges.remove(&merge.tp);
      return target_idx;
    }
    let changes = self.merge(&merge, target_idx);
    // println!("Merge {:?} (freq={}) into idx {}", merge.tp, merge.data.freq, target_idx);
    let merge = merge.with_target(target_idx);
    let merged_word = merge.merged_content();
    // self.vocab.entry(merge.tp.0).or_insert_with(|| merge.content.0.clone());
    // self.vocab.entry(merge.tp.1).or_insert_with(|| merge.content.1.clone());
    self.vocab.insert(target_idx, merged_word);
    assert_eq!(-changes.get(&merge.tp).map(|i| i.freq).unwrap_or(0), merge.data.freq);
    metrics::histogram!("bpe_trainer.changes").record(changes.len() as f64);
    self.update_pre_merges(&merge, changes);
    self.pre_merges.remove(&merge.tp);
    metrics::histogram!("bpe_trainer.occurs_in").record(merge.data.occurs_in.len() as f64);
    metrics::histogram!("bpe_trainer.freq").record(merge.data.freq as f64);
    self.merges.push(merge);
    target_idx
  }

  /// Convert a trained [`BpeTrainer`] into a [`BpeEncoder`].
  ///
  /// This re-encodes indices into the concrete `Idx` type used by encoders.
  pub fn finish(self) -> MyResult<BpeEncoder<C>>
  where
    C: Ord + Clone + Cachable,
  {
    let merges = self.merges
      .into_iter()
      .map(|m| {
        let tp = (m.tp.0.to_u64() as Idx, m.tp.1.to_u64() as Idx);
        let target = m.target.unwrap().to_u64() as Idx;
        (tp, target)
      })
      .collect();
    let vocab = self.vocab.into_iter().map(|(i, w)| (i.to_u64() as Idx, w)).collect();
    BpeEncoder::new(vocab, merges, self.special_tokens)
  }

  /// Emit internal metrics about the trainer state.
  pub fn _metrics(&self) {
    metrics::counter!("bpe_trainer.vocab_size").absolute(self.vocab.len() as u64);
    metrics::gauge!("bpe_trainer.pre_merges_count").set(self.pre_merges.len() as f64);
    metrics::gauge!("bpe_trainer.words_count").set(self.words.len() as f64);
  }
}

impl<C, I> Train for BpeTrainer<C, I>
where
  Self: CanTrain<C, I>,
{
  fn new(special_tokens: Vec<String>) -> Self {
    Self::new(vec![], special_tokens)
  }

  fn add_words(&mut self, words: &mut dyn Iterator<Item = (&str, Freq)>) {
    let special_tokens = self.special_tokens.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let vocab_start_idx = self._byte_vocab_start_idx.unwrap();
    self.words = Self::_words_to_tokens(words, vocab_start_idx, &special_tokens);
  }

  fn vocab_size(&self) -> usize {
    self.vocab.len()
  }

  fn init_training(&mut self) {
    self._build_pre_merges();
    self._metrics();
  }

  fn step(&mut self) -> MyResult<()> {
    // find the most frequent merge,
    // if the frequency is the same, choose the lexicographically largest one.
    let merge = if self.pre_merges.len() < 100_000 {
      self._get_largest_merge()
    } else {
      self._get_largest_merge2()
    };
    if let Some(merge) = merge {
      self._step(merge);
      if self.vocab_size() % 100 == 0 {
        self._metrics();
      }
      Ok(())
    } else {
      Err(MyError::TrainStep)
    }

  }
}


#[cfg(test)]
mod tests {
  use crate::{pretokenizer::DEFAULT_EOT, spec::gpt2::Gpt2Spec};

  use super::*;

  fn _test_bpe_merge(pretokens: &[(&str, Freq)], merges: &[((&str, &str), Vec<(&str, &str, MergeData)>)]) {
    fn pretoken(s: &str, freq: Freq) -> PreToken<u8, Idx> {
      let idxs = s.bytes().map(|b| b as Idx - 'a' as Idx).collect::<Vec<_>>();
      PreToken {
        src: s.to_word(),
        idxs,
        freq,
      }
    }
    fn lookup(bpe: &BpeTrainer<u8, Idx>, s: &str) -> Option<Idx> {
      bpe.vocab.iter().find_map(|(i, w)| {
        if w.as_ref() == s.as_bytes() {
          Some(*i)
        } else {
          None
        }
      })
    }
    fn display(bpe: &BpeTrainer<u8, Idx>, changes: &BTreeMap<(Idx, Idx), MergeData>) -> String {
      let mut parts = Vec::new();
      let target = ("__target__").to_word();
      for (tp, data) in changes.iter() {
        let left = bpe.vocab.get(&tp.0).unwrap_or(&target).debug_display();
        let right = bpe.vocab.get(&tp.1).unwrap_or(&target).debug_display();
        parts.push(format!("({:?}, {:?}, MergeData::new({}).occurs_in({:?}))", left, right, data.freq, data.occurs_in_vec()));
      }
      format!("{{\n  {}\n}}", parts.join(",\n  "))
    }

    let mut bpe = BpeTrainer::default();
    bpe.vocab.extend(
      ('a' ..= 'z').enumerate().map(|(i, c)| (i as Idx, c.to_string().to_word()))
    );
    bpe._set_vocab_idx(100);
    bpe.words.extend(
      pretokens.iter().map(|(s, f)| pretoken(s, *f))
    );
    bpe.init_training();
    for (m, expected) in merges {
      let merge_tp = (
        lookup(&bpe, m.0).unwrap(), lookup(&bpe, m.1).unwrap()
      );
      let merge = bpe.pre_merges.get(&merge_tp).unwrap().clone();
      let target = bpe._add_vocab_idx();
      let changes = bpe.merge(&merge, target);
      assert_eq!(merge.data.freq, -changes.get(&merge_tp).cloned().unwrap().freq);
      if expected.is_empty() {
        continue;
      }
      let expected = expected.into_iter().map(|(a, b, data)| {
        let tp_idx = (lookup(&bpe, a).unwrap_or(target), lookup(&bpe, b).unwrap_or(target));
        (tp_idx, data.clone())
      }).collect::<BTreeMap<_, _>>();
      assert_eq!(changes, expected, "\nExpected changes:\n{}\nActual changes:\n{}", display(&bpe, &expected), display(&bpe, &changes));
    }
  }

  #[test]
  fn test_bpe_merge() {
    _test_bpe_merge(&[("abcd", 5), ("abcdbcd", 30), ("abcbcdab", 200)], &[(("b", "c"), vec![
      ("a", "b", MergeData::new(-235).add_occurs_in([0, 1])),
      ("a", "bc", MergeData::new(235).add_occurs_in([0, 1, 2])),
      ("b", "c", MergeData::new(-465).add_occurs_in([0, 1, 2])),
      ("c", "b", MergeData::new(-200).add_occurs_in([2])),
      ("c", "d", MergeData::new(-265).add_occurs_in([0, 1, 2])),
      ("d", "b", MergeData::new(-30).add_occurs_in([1])),
      ("d", "bc", MergeData::new(30).add_occurs_in([1])),
      ("bc", "b", MergeData::new(0).add_occurs_in([2])),
      ("bc", "d", MergeData::new(265).add_occurs_in([0, 1, 2])),
      ("bc", "bc", MergeData::new(200).add_occurs_in([2])),
    ])]);

    _test_bpe_merge(&[("wherever", 10)],
    &[(("h", "e"), vec![
      ("e", "r", MergeData::new(-10).add_occurs_in([])),
      ("h", "e", MergeData::new(-10).add_occurs_in([0])),
      ("w", "h", MergeData::new(-10).add_occurs_in([0])),
      ("w", "he", MergeData::new(10).add_occurs_in([0])),
      ("he", "r", MergeData::new(10).add_occurs_in([0])),
    ])]);

    _test_bpe_merge(&[("aaa", 10), ("aaaa", 1)],
    &[(("a", "a"), vec![
      ("a", "a", MergeData::new(-23).add_occurs_in([0, 1])),
      ("aa", "a", MergeData::new(10).add_occurs_in([0, 1])),
      ("aa", "aa", MergeData::new(1).add_occurs_in([1])),
    ])]);
  }

  #[test]
  fn test_bpe_step() {
    let mut bpe = BpeTrainer::<u8, Idx>::from_words(vec![
      ("ababc", 5),
      ("ababcbabc", 30),
      ("abcbabcab", 200),
    ], &vec![]);
    assert!(bpe.words.len() > 0);
    bpe.init_training();
    assert!(bpe.pre_merges.len() > 0);
    for _ in 0..3 {
      bpe.step().unwrap();
    }
    let result_vocab = bpe.vocab.into_iter().map(|(i, w)| (i, w.debug_display())).skip(256).collect::<Vec<_>>();
    assert_eq!(
      result_vocab,
      vec![
        (256, "ab".to_string()),
        (257, "abc".to_string()),
        (258, "babc".to_string()),
      ]
    );
    let result_merges = bpe.merges.into_iter().map(|m| {
      let left = m.content.0.debug_display();
      let right = m.content.1.debug_display();
      (left, right, m.data.freq)
    }).collect::<Vec<_>>();
    assert_eq!(
      result_merges,
      vec![
        ("a".to_string(), "b".to_string(), 700),
        ("ab".to_string(), "c".to_string(), 465),
        ("b".to_string(), "abc".to_string(), 230),
      ]
    );
  }

  #[test]
  fn test_bpe_from_words() {
    const NAME: &str = "tinystories_sample_5M";
    // const NAME: &str = "TinyStoriesV2-GPT4-train";
    let input = std::fs::read_to_string(format!("fixtures/_words.{NAME}.json")).unwrap();
    let words: BTreeMap<String, Freq> = serde_json::from_str(&input).unwrap();
    let mut bpe = BpeTrainer::from_words(words, &vec![DEFAULT_EOT.to_string()]);
    bpe.init_training();
    let vocab_size = match NAME {
      "tinystories_sample_5M" => 2000,
      _ => 10000,
    };
    while bpe.vocab.len() < vocab_size {
      bpe.step().unwrap();
      // let m = &bpe.merges.last().unwrap();
      // println!("{} {} => {}", _printable(&m.content.0), _printable(&m.content.1), m.data.freq);
    }
    std::fs::create_dir_all("out").ok();
    bpe.save_vocab_json(&Gpt2Spec, std::fs::File::create(format!("out/vocab.{NAME}.json")).unwrap()).unwrap();
    bpe.save_merges_txt(&Gpt2Spec, std::fs::File::create(format!("out/merges.{NAME}.txt")).unwrap()).unwrap();

    let merges_txt = std::fs::read_to_string(format!("out/merges.{NAME}.txt")).unwrap();
    let merges_expect_txt = std::fs::read_to_string(format!("fixtures/merges.{NAME}.txt")).unwrap();
    assert_eq!(merges_txt, merges_expect_txt);
  }

  #[test]
  fn test_bpe_from_words_uni() {
    // const NAME: &str = "tinystories_sample_5M";
    // const NAME: &str = "TinyStoriesV2-GPT4-train";
    const NAME: &str = "TinyStories_all_data_zh_1M-sample";
    let spec = crate::spec::uni::UniSpec;
    let input = std::fs::read_to_string(format!("fixtures/_words.{NAME}.json")).unwrap();
    let words: BTreeMap<String, Freq> = serde_json::from_str(&input).unwrap();
    let mut bpe = BpeTrainer::<Character, CharIdx>::from_words(words, &vec![DEFAULT_EOT.to_string()]);
    bpe.init_training();
    let vocab_size = match NAME {
      "tinystories_sample_5M" | "TinyStories_all_data_zh_1M-sample" => 2000,
      _ => 10000,
    };
    while bpe.vocab.len() < vocab_size {
      bpe.step().unwrap();
      // let m = &bpe.merges.last().unwrap();
      // println!("{} {} => {}", _printable(&m.content.0), _printable(&m.content.1), m.data.freq);
    }
    std::fs::create_dir_all("out").ok();
    bpe.save_vocab_json(&spec, std::fs::File::create(format!("out/vocab.{NAME}.uni.json")).unwrap()).unwrap();
    bpe.save_merges_txt(&spec, std::fs::File::create(format!("out/merges.{NAME}.uni.txt")).unwrap()).unwrap();

    let merges_txt = std::fs::read_to_string(format!("out/merges.{NAME}.uni.txt")).unwrap();
    let merges_expect_txt = std::fs::read_to_string(format!("fixtures/merges.{NAME}.uni.txt")).unwrap();
    let merges = merges_txt.trim_end().lines().collect::<Vec<_>>();
    assert_eq!(merges, merges_expect_txt.lines().take(merges.len()).collect::<Vec<_>>());
  }
}
