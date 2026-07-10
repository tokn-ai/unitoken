use std::{collections::BTreeMap, io::BufReader, sync::Arc};

use fancy_regex::Regex;
use lazy_static::lazy_static;
use ordermap::OrderMap;

use crate::{MyError, MyResult, bpe::{CharSplit, Character, HasChar, IdxLike, Merge, Word}, spec::{Spec, WordDisplay}};

pub struct UnitokenSpec;

impl<C: Ord, I: IdxLike> Spec<C, I> for UnitokenSpec
where
  Self: WordDisplay<C>,
  I: HasChar<C>,
  C: CharSplit,
{
  fn suffix(&self) -> Option<&str> {
    Some("unitoken")
  }

  fn encode_vocab(&self, w: &mut dyn std::io::Write, vocab: &BTreeMap<I, Word<C>>) -> MyResult<()> {
    let mut map = OrderMap::new();
    for (idx, word) in vocab.iter() {
      let s = self.word_display(word);
      let k = if let Some(char) = idx.get_char() {
        -1-(char as i64)
      } else {
        idx.to_u64() as i64
      };
      map.insert(s, k);
    }
    let json = serde_json::to_string_pretty(&map).unwrap();
    write!(w, "{}", json)?;
    Ok(())
  }

  fn decode_vocab(&self, r: &mut dyn std::io::Read) -> MyResult<BTreeMap<I, Word<C>>> {
    let map: OrderMap<String, i64> = serde_json::from_reader(BufReader::new(r))?;
    map.into_iter().map(|(s, idx)| {
      let word = self.word_parse(&s)?;
      let i = if idx < 0 {
        let c = (-1 - idx) as u32;
        I::from_char(char::from_u32(c).unwrap()).unwrap()
      } else {
        I::from_u64(idx as u64)
      };
      Ok((i, word))
    }).collect()
  }

  fn encode_merges(&self, w: &mut dyn std::io::Write, merges: &Vec<Merge<C, I>>) -> MyResult<()> {
    for merge in merges.iter() {
      let left = self.word_display(&merge.content.0);
      let right = self.word_display(&merge.content.1);
      writeln!(w, "{} {} => {}", left, right, merge.data.freq)?;
    }
    Ok(())
  }

  fn decode_merges_raw(&self, reader: &mut dyn std::io::Read) -> MyResult<Vec<Merge<C, Word<C>>>> {
    let mut result = Vec::new();
    let mut input = String::new();
    reader.read_to_string(&mut input)?;
    for (i, line) in input.lines().enumerate() {
      if line.trim().is_empty() {
        continue;
      }
      let mut main = line;
      let mut freq = 0;
      if line.contains(" => ") {
        let split = line.rsplitn(2, " => ").collect::<Vec<_>>();
        main = split.last().unwrap();
        if split.len() > 1 {
          freq = split[0].trim().parse().unwrap_or_default();
        }
      }
      let parts = main.trim().split_whitespace().collect::<Vec<_>>();
      if parts.len() != 2 {
        return Err(MyError::MergeTxt("main parts is not 2", i))
      }
      let a = self.word_parse(parts[0])?;
      let b = self.word_parse(parts[1])?;
      let mut merge = Merge::new((a.clone(), b.clone()), (a, b));
      merge.data.freq = freq;
      result.push(merge);
    }
    Ok(result)
  }

  fn decode_merges(&self, r: &mut dyn std::io::Read, vocab: &BTreeMap<I, Word<C>>) -> MyResult<Vec<Merge<C, I>>> {
    let merges_raw = <Self as Spec<C, I>>::decode_merges_raw(self, r)?;
    let vocab = vocab.iter().map(|(k, v)| (v.clone(), *k)).collect::<BTreeMap<_, _>>();
    let get_kv = |vocab: &BTreeMap<Word<C>, I>, w: &Word<C>| -> MyResult<I> {
      Ok(*vocab.get(w).ok_or_else(|| MyError::Oov(self.word_display(w)))?)
    };
    let result = merges_raw.into_iter().map(|merge| {
      let (a, b) = &merge.content;
      let a_idx = get_kv(&vocab, &a)?;
      let b_idx = get_kv(&vocab, &b)?;
      let mut vec_merged = CharSplit::to_vec_u8(a);
      vec_merged.extend(CharSplit::to_vec_u8(b));
      let merged = CharSplit::from_vec_u8(&vec_merged);
      let m_idx = get_kv(&vocab, &merged)?;
      let mut merge_new = Merge::new((a_idx, b_idx), merge.content).with_target(m_idx);
      merge_new.data.freq = merge.data.freq;
      Ok(merge_new)
    }).collect::<MyResult<_>>()?;
    Ok(result)
  }
}

impl WordDisplay<Character> for UnitokenSpec {
  fn word_display(&self, word: &Word<Character>) -> String {
    _printable(word)
  }

  fn word_parse(&self, s: &str) -> MyResult<Word<Character>> {
    let w = _parse_str(s)?;
    Ok(Arc::from(w.into_boxed_slice()))
  }
}

impl WordDisplay<u8> for UnitokenSpec {
  fn word_display(&self, word: &Word<u8>) -> String {
    _printable(&CharSplit::from_vec_u8(word.as_ref()))
  }

  fn word_parse(&self, s: &str) -> MyResult<Word<u8>> {
    let w = _parse_str(s)?;
    let mut bytes = Vec::new();
    for ch in w.iter() {
      match ch {
        Character::Unicode(c) => {
          let mut buf = [0; 4];
          let encoded = c.encode_utf8(&mut buf);
          bytes.extend_from_slice(encoded.as_bytes());
        }
        Character::Byte(b) => {
          bytes.push(*b);
        }
      }
    }
    Ok(Arc::from(bytes.into_boxed_slice()))
  }
}

fn _should_escape(c: char) -> bool {
  c < '!' || c == '\x7f' // 33..=126
}

fn _display_char(ch: &Character) -> String {
  match ch {
    Character::Unicode(' ') => '␣'.to_string(),
    Character::Unicode(c @ ('␣' | '{' | '}')) => format!("{{u{:04x}}}", *c as u32),
    Character::Unicode(c) if _should_escape(*c) => format!("{{u{:04x}}}", *c as u32),
    Character::Unicode(c) => {
      c.to_string()
    }
    Character::Byte(b) => format!("{{x{:02x}}}", *b),
  }
}

fn _printable(word: &Word<Character>) -> String {
  word.iter().map(|c| _display_char(c)).collect()
}

lazy_static! {
  static ref PRINTABLE_REGEX: Regex = Regex::new(r"\{([ux][0-9a-fA-F]{2,})\}").unwrap();
}
fn _parse_str(s: &str) -> MyResult<Vec<Character>> {
  let mut result = Vec::new();
  let mut last_i = 0;
  fn _decode_char(ch: char) -> Character {
    match ch {
      '␣' => Character::Unicode(' '),
      _ => Character::Unicode(ch),
    }
  }
  for m in PRINTABLE_REGEX.find_iter(s) {
    let m = m?;
    for c in s[last_i..m.start()].chars() {
      result.push(_decode_char(c));
    }
    last_i = m.end();
    let token = m.as_str();
    let token = &token[1..token.len() - 1]; // strip {}
    if token.starts_with('u') {
      let codepoint = u32::from_str_radix(&token[1..], 16).map_err(|_| MyError::InvalidPrintableChar('?'))?;
      if let Some(ch) = std::char::from_u32(codepoint) {
        result.push(Character::Unicode(ch));
      } else {
        return Err(MyError::InvalidPrintableEscape(token.to_string()));
      }
    } else if token.starts_with('x') {
      let byte = u8::from_str_radix(&token[1..], 16).map_err(|_| MyError::InvalidPrintableEscape(token.to_string()))?;
      result.push(Character::Byte(byte));
    } else {
      return Err(MyError::InvalidPrintableEscape(token.to_string()));
    }
  }
  for c in s[last_i..].chars() {
    result.push(_decode_char(c));
  }
  Ok(result)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_display_char() {
    assert_eq!(_display_char(&Character::Unicode('a')), "a".to_string());
    assert_eq!(_display_char(&Character::Unicode(' ')), "␣".to_string());
    assert_eq!(_display_char(&Character::Unicode('␣')), "{u2423}".to_string());
    assert_eq!(_display_char(&Character::Unicode('{')), "{u007b}".to_string());
    assert_eq!(_display_char(&Character::Unicode('}')), "{u007d}".to_string());
    assert_eq!(_display_char(&Character::Byte(0x41)), "{x41}".to_string());
  }

  #[test]
  fn test_parse_str() {
    let s = "a{u0041} {x42}{x43}{u0044}'␣'你{xe5}{xa5}{xbd}zz";
    let chars = _parse_str(s).unwrap();
    let expected = vec![
      Character::Unicode('a'),
      Character::Unicode('A'),
      Character::Unicode(' '),
      Character::Byte(0x42),
      Character::Byte(0x43),
      Character::Unicode('D'),
      Character::Unicode('\''),
      Character::Unicode(' '),
      Character::Unicode('\''),
      Character::Unicode('你'),
      Character::Byte(229),
      Character::Byte(165),
      Character::Byte(189),
      Character::Unicode('z'),
      Character::Unicode('z'),
    ];
    assert_eq!(chars, expected);
  }
}
