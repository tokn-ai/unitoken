use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use unitoken::{
  bpe::{encoder::BpeBuilder, Idx},
  pretokenizer::PreTokenizer,
  spec::gpt2::Gpt2Spec,
  traits::Encode as _,
};

fn build_gpt2_encoder_from_fixtures(name: &str) -> unitoken::bpe::BpeEncoder<u8> {
  BpeBuilder::new()
    .load_merges_file(format!("fixtures/merges.{name}.txt"), &Gpt2Spec)
    .unwrap()
    .load_vocab_file(format!("fixtures/vocab.{name}.json"), &Gpt2Spec)
    .unwrap()
    .build(&Gpt2Spec)
    .unwrap()
}

fn bench_pretokenizer(c: &mut Criterion) {
  let special_tokens = vec![unitoken::pretokenizer::DEFAULT_EOT.to_string()];
  let pre = PreTokenizer::new(&special_tokens, Some(unitoken::pretokenizer::DEFAULT_EOT));

  let base = "Once upon a time, in a small village, there lived a cat named Mango.";
  let input = base.repeat(200);

  c.bench_function("pretokenizer/get_tokens_index_from_segment", |b| {
    b.iter(|| {
      let (tokens_index, special_tokens_index) = pre
        .get_tokens_index_from_segment(black_box(&input))
        .unwrap();
      black_box((tokens_index.len(), special_tokens_index.len()))
    })
  });
}

fn bench_bpe_encode_decode(c: &mut Criterion) {
  const FIXTURE: &str = "tinystories_sample_5M";
  let bpe = build_gpt2_encoder_from_fixtures(FIXTURE);

  let base = "Once upon a time, there was a little robot who loved to read books.";
  let input = base.repeat(200);

  let mut group = c.benchmark_group("bpe");

  group.bench_with_input(BenchmarkId::new("encode_string", FIXTURE), &input, |b, s| {
    b.iter(|| {
      let out = bpe.encode_string(black_box(s)).unwrap();
      black_box(out)
    })
  });

  let encoded: Vec<Idx> = bpe.encode_string(&input).unwrap();
  group.bench_with_input(BenchmarkId::new("decode", FIXTURE), &encoded, |b, ids| {
    b.iter(|| {
      let out = bpe.decode(black_box(ids)).unwrap();
      black_box(out)
    })
  });

  group.finish();
}

criterion_group!(benches, bench_pretokenizer, bench_bpe_encode_decode);
criterion_main!(benches);
