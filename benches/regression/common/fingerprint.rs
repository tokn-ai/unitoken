use std::{
  collections::BTreeMap,
  fs,
  io::{BufReader, Read},
  path::Path,
};

use ahash::AHashSet;
use sha2::{Digest, Sha256};
use unitoken::bpe::{BpeModel, CharIdx, Character, Freq, Idx, Merge, PreToken, Word};

const FINGERPRINT_VERSION: u64 = 1;

#[derive(Clone, Debug, serde::Deserialize, Eq, PartialEq, serde::Serialize)]
pub struct ModelFingerprints {
  pub vocab_sha256: String,
  pub merges_sha256: String,
  pub model_sha256: String,
  pub word_state_sha256: String,
}

pub trait CanonicalId {
  fn update_fingerprint(&self, fingerprint: &mut CanonicalSha256);
}

impl CanonicalId for u32 {
  fn update_fingerprint(&self, fingerprint: &mut CanonicalSha256) {
    fingerprint.update_tag(b'n');
    fingerprint.update_u64(*self as u64);
  }
}

impl CanonicalId for CharIdx {
  fn update_fingerprint(&self, fingerprint: &mut CanonicalSha256) {
    match self {
      Self::Idx(idx) => {
        fingerprint.update_tag(b'n');
        fingerprint.update_u64(*idx as u64);
      }
      Self::Char(ch) => {
        fingerprint.update_tag(b'c');
        fingerprint.update_u64(*ch as u32 as u64);
      }
    }
  }
}

pub trait CanonicalUnit {
  const UNIT_NAME: &'static [u8];

  fn update_fingerprint(&self, fingerprint: &mut CanonicalSha256);
}

impl CanonicalUnit for u8 {
  const UNIT_NAME: &'static [u8] = b"byte";

  fn update_fingerprint(&self, fingerprint: &mut CanonicalSha256) {
    fingerprint.update_tag(b'b');
    fingerprint.update_tag(*self);
  }
}

impl CanonicalUnit for Character {
  const UNIT_NAME: &'static [u8] = b"unicode";

  fn update_fingerprint(&self, fingerprint: &mut CanonicalSha256) {
    match self {
      Self::Unicode(ch) => {
        fingerprint.update_tag(b'u');
        fingerprint.update_u64(*ch as u32 as u64);
      }
      Self::Byte(byte) => {
        fingerprint.update_tag(b'b');
        fingerprint.update_tag(*byte);
      }
    }
  }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
  let mut digest = Sha256::new();
  digest.update(bytes);
  to_hex(&digest.finalize())
}

pub fn sha256_file(path: &Path) -> Result<String, String> {
  let file = fs::File::open(path)
    .map_err(|error| format!("cannot open {}: {error}", path.display()))?;
  let mut reader = BufReader::new(file);
  let mut digest = Sha256::new();
  let mut buffer = vec![0u8; 1024 * 1024];
  loop {
    let read = reader
      .read(&mut buffer)
      .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    if read == 0 {
      break;
    }
    digest.update(&buffer[..read]);
  }
  Ok(to_hex(&digest.finalize()))
}

pub fn fingerprint_unicode_bigrams(
  bigrams: &AHashSet<(char, char)>,
  cutoff_freq: Option<Freq>,
  max_excluded_freq: Option<Freq>,
) -> String {
  let mut sorted = bigrams.iter().copied().collect::<Vec<_>>();
  sorted.sort_unstable();
  let mut fingerprint = CanonicalSha256::new(b"unitoken:unicode_bigrams");
  fingerprint.update_len(sorted.len());
  for (left, right) in sorted {
    fingerprint.update_u64(left as u32 as u64);
    fingerprint.update_u64(right as u32 as u64);
  }
  fingerprint.update_optional_i64(cutoff_freq);
  fingerprint.update_optional_i64(max_excluded_freq);
  fingerprint.finish_hex()
}

pub fn fingerprint_word_counts(words: &BTreeMap<String, Freq>) -> String {
  let mut fingerprint = CanonicalSha256::new(b"unitoken:word_counts");
  fingerprint.update_len(words.len());
  for (word, frequency) in words {
    fingerprint.update_bytes(word.as_bytes());
    fingerprint.update_i64(*frequency);
  }
  fingerprint.finish_hex()
}

pub fn fingerprint_token_ids(ids: &[Idx]) -> String {
  let mut digest = Sha256::new();
  for id in ids {
    digest.update(id.to_le_bytes());
  }
  to_hex(&digest.finalize())
}

pub fn fingerprint_model<C, I>(model: &BpeModel<C, I>, words: &[PreToken<C, I>]) -> Result<ModelFingerprints, String>
where
  C: CanonicalUnit,
  I: CanonicalId,
{
  let vocab_digest = fingerprint_vocab(model);
  let merges_digest = fingerprint_merges(model.merges())?;
  let mut model_fingerprint = CanonicalSha256::new(b"unitoken:model");
  model_fingerprint.update_bytes(&vocab_digest);
  model_fingerprint.update_bytes(&merges_digest);

  Ok(ModelFingerprints {
    vocab_sha256: to_hex(&vocab_digest),
    merges_sha256: to_hex(&merges_digest),
    model_sha256: model_fingerprint.finish_hex(),
    word_state_sha256: fingerprint_words(words).finish_hex(),
  })
}

fn fingerprint_vocab<C, I>(model: &BpeModel<C, I>) -> [u8; 32]
where
  C: CanonicalUnit,
  I: CanonicalId,
{
  let mut fingerprint = CanonicalSha256::new(b"unitoken:vocab");
  fingerprint.update_bytes(C::UNIT_NAME);
  fingerprint.update_len(model.special_tokens().len());
  for token in model.special_tokens() {
    fingerprint.update_bytes(token.as_bytes());
  }
  fingerprint.update_len(model.vocab().len());
  for (idx, token) in model.vocab() {
    idx.update_fingerprint(&mut fingerprint);
    fingerprint.update_word(token);
  }
  fingerprint.finish()
}

fn fingerprint_merges<C, I>(merges: &[Merge<C, I>]) -> Result<[u8; 32], String>
where
  I: CanonicalId,
{
  let mut fingerprint = CanonicalSha256::new(b"unitoken:merges");
  fingerprint.update_len(merges.len());
  for (rank, merge) in merges.iter().enumerate() {
    fingerprint.update_u64(rank as u64);
    merge.tp.0.update_fingerprint(&mut fingerprint);
    merge.tp.1.update_fingerprint(&mut fingerprint);
    merge
      .target
      .as_ref()
      .ok_or_else(|| format!("validated merge {rank} has no target"))?
      .update_fingerprint(&mut fingerprint);
    fingerprint.update_i64(merge.data.freq);
  }
  Ok(fingerprint.finish())
}

fn fingerprint_words<C, I>(words: &[PreToken<C, I>]) -> CanonicalSha256
where
  C: CanonicalUnit,
  I: CanonicalId,
{
  let mut fingerprint = CanonicalSha256::new(b"unitoken:word_state");
  fingerprint.update_bytes(C::UNIT_NAME);
  fingerprint.update_len(words.len());
  for word in words {
    fingerprint.update_word(&word.src);
    fingerprint.update_i64(word.freq);
    fingerprint.update_len(word.idxs.len());
    for idx in &word.idxs {
      idx.update_fingerprint(&mut fingerprint);
    }
  }
  fingerprint
}

pub struct CanonicalSha256 {
  digest: Sha256,
}

impl CanonicalSha256 {
  fn new(domain: &[u8]) -> Self {
    let mut result = Self { digest: Sha256::new() };
    result.update_bytes(domain);
    result.update_u64(FINGERPRINT_VERSION);
    result
  }

  fn update_tag(&mut self, value: u8) {
    self.digest.update([value]);
  }

  fn update_u64(&mut self, value: u64) {
    self.digest.update(value.to_le_bytes());
  }

  fn update_i64(&mut self, value: i64) {
    self.digest.update(value.to_le_bytes());
  }

  fn update_optional_i64(&mut self, value: Option<i64>) {
    match value {
      None => self.update_tag(0),
      Some(value) => {
        self.update_tag(1);
        self.update_i64(value);
      }
    }
  }

  fn update_len(&mut self, value: usize) {
    self.update_u64(value as u64);
  }

  fn update_bytes(&mut self, value: &[u8]) {
    self.update_len(value.len());
    self.digest.update(value);
  }

  fn update_word<C: CanonicalUnit>(&mut self, word: &Word<C>) {
    self.update_len(word.len());
    for unit in word.iter() {
      unit.update_fingerprint(self);
    }
  }

  fn finish(self) -> [u8; 32] {
    self.digest.finalize().into()
  }

  fn finish_hex(self) -> String {
    to_hex(&self.finish())
  }
}

pub fn to_hex(bytes: &[u8]) -> String {
  const HEX: &[u8; 16] = b"0123456789abcdef";
  let mut result = String::with_capacity(bytes.len() * 2);
  for byte in bytes {
    result.push(HEX[(byte >> 4) as usize] as char);
    result.push(HEX[(byte & 0x0f) as usize] as char);
  }
  result
}
