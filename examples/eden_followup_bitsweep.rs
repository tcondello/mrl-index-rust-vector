//! Follow-up: EDEN-prod prefix bitwidth sweep (b=2,3,4) vs min–max 4-bit (offline).
//!
//! Usage:
//!   cargo run --release --example eden_followup_bitsweep -- data/msmarco_gemma2/smoke_bin [max_q]

use std::collections::HashSet;
use std::env;
use std::path::Path;

use mrl_index::block::MRL_PREFIX_DIMS;
use mrl_index::eden::{
    encode_minmax4_prefix, encode_slice, rotate_query, score_ip, score_minmax4_prefix, EdenFit,
};
use mrl_index::retrieval_bench::RetrievalBenchData;

fn pearson(xs: &[f32], ys: &[f32]) -> f32 {
    let n = xs.len() as f32;
    let mx = xs.iter().sum::<f32>() / n;
    let my = ys.iter().sum::<f32>() / n;
    let mut num = 0.0f32;
    let mut dx = 0.0f32;
    let mut dy = 0.0f32;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let a = x - mx;
        let b = y - my;
        num += a * b;
        dx += a * a;
        dy += b * b;
    }
    num / (dx.sqrt() * dy.sqrt()).max(1e-12)
}

fn topk_ids(scores: &[(usize, f32)], k: usize) -> Vec<usize> {
    let mut v = scores.to_vec();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    v.truncate(k);
    v.into_iter().map(|(i, _)| i).collect()
}

fn recall_at_k(truth: &[usize], pred: &[usize]) -> f64 {
    if truth.is_empty() {
        return 1.0;
    }
    let set: HashSet<usize> = pred.iter().copied().collect();
    truth.iter().filter(|id| set.contains(id)).count() as f64 / truth.len() as f64
}

fn main() -> anyhow::Result<()> {
    let data_dir = env::args()
        .nth(1)
        .unwrap_or_else(|| "data/msmarco_gemma2/smoke_bin".into());
    let max_q: usize = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let seed = 1u64;

    let data = RetrievalBenchData::load(Path::new(&data_dir))?;
    let n = data.n_corpus();
    let n_q = data.n_queries().min(max_q);

    let mm_codes: Vec<_> = (0..n)
        .map(|i| encode_minmax4_prefix(data.corpus_vector(i)))
        .collect();

    let mut by_bits = serde_json::Map::new();
    for bits in [2u8, 3, 4] {
        let fit = EdenFit::fit_prefix((0..n).map(|i| data.corpus_vector(i)), bits, seed);
        let codes: Vec<_> = (0..n)
            .map(|i| encode_slice(&fit, data.corpus_vector(i)))
            .collect();

        let mut corr = 0.0f64;
        let mut r10 = 0.0f64;
        let mut mm_corr = 0.0f64;
        let mut mm_r10 = 0.0f64;

        for qi in 0..n_q {
            let q = data.query_vector(qi);
            let q_rot = rotate_query(&fit, q);
            let mut exact = Vec::with_capacity(n);
            let mut est = Vec::with_capacity(n);
            let mut mm = Vec::with_capacity(n);
            let mut ev = Vec::with_capacity(n);
            let mut xv = Vec::with_capacity(n);
            let mut mv = Vec::with_capacity(n);
            for i in 0..n {
                let row = data.corpus_vector(i);
                let mut e = 0.0f32;
                for d in 0..MRL_PREFIX_DIMS {
                    e += q[d] * row[d];
                }
                let s = score_ip(&fit, q, &q_rot, &codes[i]);
                let m = score_minmax4_prefix(q, &mm_codes[i]);
                exact.push((i, e));
                est.push((i, s));
                mm.push((i, m));
                ev.push(s);
                xv.push(e);
                mv.push(m);
            }
            corr += pearson(&ev, &xv) as f64;
            mm_corr += pearson(&mv, &xv) as f64;
            let truth = topk_ids(&exact, 10);
            r10 += recall_at_k(&truth, &topk_ids(&est, 10));
            mm_r10 += recall_at_k(&truth, &topk_ids(&mm, 10));
        }
        let nqf = n_q as f64;
        by_bits.insert(
            format!("b{bits}"),
            serde_json::json!({
                "prefix_ip_pearson": corr / nqf,
                "prefix_rank_R@10": r10 / nqf,
                "minmax4_pearson": mm_corr / nqf,
                "minmax4_R@10": mm_r10 / nqf,
                "beats_minmax_R10": (r10 / nqf) > (mm_r10 / nqf),
            }),
        );
    }

    let report = serde_json::json!({
        "followup": "prefix_bitsweep",
        "data": data_dir,
        "n": n,
        "n_queries": n_q,
        "seed": seed,
        "by_bits": by_bits,
    });
    let out_dir = Path::new("bench_results/eden");
    std::fs::create_dir_all(out_dir)?;
    let out = out_dir.join("followup_bitsweep_smoke.json");
    std::fs::write(&out, serde_json::to_string_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    eprintln!("wrote {}", out.display());
    Ok(())
}
