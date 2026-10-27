use criterion::{criterion_group, criterion_main, Criterion};
use mrl_index::block::source;
use mrl_index::config::{EngineConfig, StrataDepth};
use mrl_index::quantize::encode_block;
use mrl_index::scan::{scan_blocks, QueryPlan};
use mrl_index::PaxBlock64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn make_blocks(n_blocks: usize, dims: usize) -> (Vec<PaxBlock64>, Vec<f32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let mut data = Vec::with_capacity(n_blocks);
    for _ in 0..n_blocks {
        let mut vecs = Vec::with_capacity(64);
        for _ in 0..64 {
            let mut v: Vec<f32> = (0..dims).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            for x in &mut v {
                *x /= n;
            }
            vecs.push(v);
        }
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        data.push(encode_block(&refs, &[1u32; 64], &[source::DOC; 64], dims));
    }
    let mut q: Vec<f32> = (0..dims).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let n = q.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for x in &mut q {
        *x /= n;
    }
    (data, q)
}

fn bench_scan(c: &mut Criterion) {
    let dims = 384;
    let (data, q) = make_blocks(256, dims); // 16,384 vectors
    let plan = QueryPlan::from_query(&q, dims);
    let mut cfg = EngineConfig::default();
    cfg.embed_dims = dims;
    cfg.strata_depth = StrataDepth::Full64b;

    c.bench_function("scan_16k_full64b", |b| {
        b.iter(|| scan_blocks(&data, &plan, &cfg, None, None))
    });
}

criterion_group!(benches, bench_scan);
criterion_main!(benches);
