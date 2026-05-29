use std::{
  alloc::{GlobalAlloc, Layout, System},
  hint::black_box,
  path::PathBuf,
  sync::atomic::{AtomicU64, Ordering},
  time::{Duration, Instant},
};

use unitoken::{
  bpe::{encoder::BpeBuilder, Idx},
  pretokenizer::{split_special_tokens, SplitChunk},
  spec::gpt2::Gpt2Spec,
  traits::Encode as _,
};

const R50K_PAT: &str =
  r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";
const DEFAULT_CHARS: usize = 200_000;
const DEFAULT_REPEATS: usize = 100;
const FIXTURE: &str = "tinystories_sample_5M";

struct CountingAllocator;

static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
  unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
    ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
    ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
    unsafe { System.alloc(layout) }
  }

  unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
    unsafe { System.dealloc(ptr, layout) }
  }
}

#[derive(Clone, Copy)]
struct Sample {
  duration: Duration,
  alloc_calls: u64,
  alloc_bytes: u64,
}

fn reset_allocs() {
  ALLOC_CALLS.store(0, Ordering::Relaxed);
  ALLOC_BYTES.store(0, Ordering::Relaxed);
}

fn sample<F>(mut f: F) -> Sample
where
  F: FnMut() -> usize,
{
  reset_allocs();
  let started = Instant::now();
  let value = f();
  let duration = started.elapsed();
  black_box(value);
  Sample {
    duration,
    alloc_calls: ALLOC_CALLS.load(Ordering::Relaxed),
    alloc_bytes: ALLOC_BYTES.load(Ordering::Relaxed),
  }
}

fn measure<F>(label: &str, repeats: usize, mut f: F)
where
  F: FnMut() -> usize,
{
  let mut samples = Vec::with_capacity(repeats);
  for _ in 0..repeats {
    samples.push(sample(&mut f));
  }
  samples.sort_by_key(|s| s.duration);
  let median = samples[samples.len() / 2];
  let min = samples[0];
  println!(
    "{label:32} min_ms={:.4} median_ms={:.4} alloc_calls={} alloc_bytes={}",
    min.duration.as_secs_f64() * 1000.0,
    median.duration.as_secs_f64() * 1000.0,
    median.alloc_calls,
    median.alloc_bytes,
  );
}

fn prefix_by_chars(text: String, chars: usize) -> String {
  text.chars().take(chars).collect()
}

fn parse_arg(args: &mut impl Iterator<Item = String>, default: usize, name: &str) -> usize {
  args
    .next()
    .as_deref()
    .map_or(Ok(default), str::parse::<usize>)
    .unwrap_or_else(|_| panic!("{name} must be an integer"))
}

fn main() {
  let mut args = std::env::args().skip(1);
  let chars = parse_arg(&mut args, DEFAULT_CHARS, "chars");
  let repeats = parse_arg(&mut args, DEFAULT_REPEATS, "repeats");
  assert!(repeats > 0, "repeats must be greater than zero");

  let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
  let text = prefix_by_chars(
    std::fs::read_to_string(root.join("fixtures").join(format!("{FIXTURE}.txt"))).unwrap(),
    chars,
  );
  let bpe = BpeBuilder::new()
    .load_merges_file(root.join("fixtures").join(format!("merges.{FIXTURE}.txt")), &Gpt2Spec)
    .unwrap()
    .load_vocab_file(root.join("fixtures").join(format!("vocab.{FIXTURE}.json")), &Gpt2Spec)
    .unwrap()
    .set_pat_str(Some(R50K_PAT.to_string()))
    .special_tokens(vec!["<|endoftext|>".to_string()])
    .build(&Gpt2Spec)
    .unwrap();

  let encoded: Vec<Idx> = bpe.encode_string(&text).unwrap();
  println!("chars={} bytes={} repeats={} tokens={}", text.chars().count(), text.len(), repeats, encoded.len());

  measure("split_special_tokens", repeats, || {
    let parts = split_special_tokens(&text, &bpe.pre_tokenizer.re_special_tokens).unwrap();
    parts.len()
  });
  measure("regex_tokens_after_split", repeats, || {
    let parts = split_special_tokens(&text, &bpe.pre_tokenizer.re_special_tokens).unwrap();
    let mut count = 0;
    for part in parts {
      match part {
        SplitChunk::Special(_) => count += 1,
        SplitChunk::Chunk(chunk) => {
          for token in bpe.pre_tokenizer.re_pat.find_iter(chunk) {
            black_box(token.unwrap().as_str());
            count += 1;
          }
        }
      }
    }
    count
  });
  measure("get_tokens_index", repeats, || {
    let (tokens, specials) = bpe.pre_tokenizer.get_tokens_index_from_segment(&text).unwrap();
    tokens.len() + specials.len()
  });
  measure("encode_string", repeats, || bpe.encode_string(&text).unwrap().len());
  measure("decode", repeats, || bpe.decode(&encoded).unwrap().len());
}
