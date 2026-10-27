//! Diagnose scan vs exact IP recall on an MRL corpus.

use mrl_index::block::VECTORS_PER_BLOCK;
use mrl_index::config::{EngineConfig, StrataDepth};
use mrl_index::corpus::MrlCorpus;
use mrl_index::embed::dot;
use mrl_index::quantize::encode_block;
use mrl_index::scan::{scan_blocks, QueryPlan};
use mrl_index::block::source;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;
use std::env;

fn main() {
    let dir = env::args().nth(1).unwrap_or_else(|| "data/sec10k_mrl".into());
    let corpus = MrlCorpus::load(&dir).expect("load");
    let vectors = corpus.all_vectors();
    let dims = corpus.dims();
    let n = vectors.len();
    println!("loaded n={n} dims={dims} model={}", corpus.manifest.model);

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let n_q = 64usize;
    let mut qis = Vec::new();
    while qis.len() < n_q {
        let i = rng.gen_range(0..n);
        if !qis.contains(&i) {
            qis.push(i);
        }
    }

    // Build blocks once
    let mut blocks = Vec::new();
    for chunk in vectors.chunks(VECTORS_PER_BLOCK) {
        let refs: Vec<&[f32]> = chunk.iter().map(|v| v.as_slice()).collect();
        let ts: Vec<u32> = (0..chunk.len()).map(|i| i as u32).collect();
        let src = vec![source::DOC; chunk.len()];
        blocks.push(encode_block(&refs, &ts, &src, dims));
    }

    for (label, strata, shortlist, cs) in [
        ("L1 cs=1.0 sl=64", StrataDepth::L1Rabitq, 64, 1.0f32),
        ("L1 cs=1.2 sl=64", StrataDepth::L1Rabitq, 64, 1.2),
        ("FULL cs=1.0 sl=64", StrataDepth::Full64b, 64, 1.0),
        ("FULL cs=1.2 sl=64", StrataDepth::Full64b, 64, 1.2),
        ("FULL cs=1.0 sl=256", StrataDepth::Full64b, 256, 1.0),
        ("L0 cs=1.2 sl=256", StrataDepth::L0Only, 256, 1.2),
    ] {
        let mut cfg = EngineConfig::default();
        cfg.embed_dims = dims;
        cfg.strata_depth = strata;
        cfg.shortlist_cand = shortlist;
        cfg.cs_tolerance = cs;
        cfg.temporal_filter_enable = false;

        let mut r10 = 0.0;
        let mut r_sl = 0.0;
        let mut spearman_acc = 0.0;
        let mut in_sl_of_exact10 = 0.0;

        for &qi in &qis {
            let q = &vectors[qi];
            // exact top
            let mut exact: Vec<(usize, f32)> = vectors
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != qi)
                .map(|(i, v)| (i, dot(q, v)))
                .collect();
            exact.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let truth10: HashSet<usize> = exact.iter().take(10).map(|x| x.0).collect();
            let truth_sl: HashSet<usize> = exact.iter().take(shortlist).map(|x| x.0).collect();

            let plan = QueryPlan::from_query(q, dims);
            let hits = scan_blocks(&blocks, &plan, &cfg, None, None);
            let ids: Vec<usize> = hits
                .iter()
                .map(|h| h.id.block as usize * VECTORS_PER_BLOCK + h.id.slot as usize)
                .filter(|i| *i != qi)
                .collect();

            let got10 = ids.iter().take(10).filter(|i| truth10.contains(i)).count() as f64 / 10.0;
            let denom = shortlist.min(ids.len()).max(1);
            let got_sl = ids.iter().filter(|i| truth_sl.contains(i)).count() as f64 / denom as f64;
            // fraction of exact top-10 that appear anywhere in shortlist
            let covered = truth10.iter().filter(|i| ids.contains(i)).count() as f64 / 10.0;

            r10 += got10;
            r_sl += got_sl;
            in_sl_of_exact10 += covered;

            // score correlation on shortlist: compare scan score vs exact for survivors
            if hits.len() >= 2 {
                let mut xs = Vec::new();
                let mut ys = Vec::new();
                for h in &hits {
                    let i = h.id.block as usize * VECTORS_PER_BLOCK + h.id.slot as usize;
                    if i == qi { continue; }
                    xs.push(h.score);
                    ys.push(dot(q, &vectors[i]));
                }
                spearman_acc += pearson(&xs, &ys);
            }
        }
        let nq = qis.len() as f64;
        println!(
            "{label:24}  R@10={:.3}  R@SL={:.3}  exact10_in_SL={:.3}  score_pearson={:.3}",
            r10 / nq,
            r_sl / nq,
            in_sl_of_exact10 / nq,
            spearman_acc / nq
        );
    }
}

fn pearson(xs: &[f32], ys: &[f32]) -> f64 {
    let n = xs.len() as f64;
    if n < 2.0 { return 0.0; }
    let mx = xs.iter().map(|x| *x as f64).sum::<f64>() / n;
    let my = ys.iter().map(|x| *x as f64).sum::<f64>() / n;
    let mut num = 0.0;
    let mut dx = 0.0;
    let mut dy = 0.0;
    for i in 0..xs.len() {
        let a = xs[i] as f64 - mx;
        let b = ys[i] as f64 - my;
        num += a * b;
        dx += a * a;
        dy += b * b;
    }
    if dx == 0.0 || dy == 0.0 { return 0.0; }
    num / (dx.sqrt() * dy.sqrt())
}
