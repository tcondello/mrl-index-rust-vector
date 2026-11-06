//! Phase 0 offline probe: EDEN-prod b=4 vs min–max 4-bit on the 64-d MRL prefix.
//!
//! Usage:
//!   cargo run --release --example eden_phase0 -- data/msmarco_gemma2/smoke_bin [max_queries]

use std::collections::HashSet;
use std::env;
use std::path::Path;

use mrl_index::block::MRL_PREFIX_DIMS;
use mrl_index::eden::{
    encode_minmax4_prefix, encode_slice, reconstruct, reconstruct_minmax4_prefix,
    rotate_query_centered, score_ip, score_minmax4_prefix, EdenFit,
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
    let data = env::args()
        .nth(1)
        .unwrap_or_else(|| "data/msmarco_gemma2/smoke_bin".into());
    let max_queries: usize = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let bits: u8 = 4;
    let seed: u64 = 1;

    let suite = RetrievalBenchData::load(Path::new(&data))?;
    let n = suite.n_corpus();
    let dims = suite.dims();
    let n_q = suite.n_queries().min(max_queries);
    eprintln!(
        "phase0 EDEN-prod vs minmax4  n={n} q={n_q}/{} dims={dims} bits={bits} seed={seed}",
        suite.n_queries()
    );

    let fit = EdenFit::fit_prefix((0..n).map(|i| suite.corpus_vector(i)), bits, seed);

    let mut eden_codes = Vec::with_capacity(n);
    let mut mm_codes = Vec::with_capacity(n);
    let mut eden_mse = 0.0f64;
    let mut mm_mse = 0.0f64;
    for i in 0..n {
        let row = suite.corpus_vector(i);
        let ec = encode_slice(&fit, row);
        let mc = encode_minmax4_prefix(row);
        let er = reconstruct(&fit, &ec);
        let mr = reconstruct_minmax4_prefix(&mc);
        for d in 0..MRL_PREFIX_DIMS {
            let e = row[d] as f64 - er[d] as f64;
            let m = row[d] as f64 - mr[d] as f64;
            eden_mse += e * e;
            mm_mse += m * m;
        }
        eden_codes.push(ec);
        mm_codes.push(mc);
    }
    let denom = (n * MRL_PREFIX_DIMS) as f64;
    eden_mse /= denom;
    mm_mse /= denom;

    let mut eden_corr = 0.0f64;
    let mut mm_corr = 0.0f64;
    let mut eden_prefix_r10 = 0.0f64;
    let mut mm_prefix_r10 = 0.0f64;
    let mut eden_route256 = 0.0f64;
    let mut mm_route256 = 0.0f64;
    let mut eden_route64 = 0.0f64;
    let mut mm_route64 = 0.0f64;

    for qi in 0..n_q {
        let q = suite.query_vector(qi);
        let q_rot = rotate_query_centered(&fit, q);

        let mut exact_prefix = Vec::with_capacity(n);
        let mut eden_est = Vec::with_capacity(n);
        let mut mm_est = Vec::with_capacity(n);
        let mut exact_vals = Vec::with_capacity(n);
        let mut eden_vals = Vec::with_capacity(n);
        let mut mm_vals = Vec::with_capacity(n);

        for i in 0..n {
            let row = suite.corpus_vector(i);
            let mut exact = 0.0f32;
            for d in 0..MRL_PREFIX_DIMS {
                exact += q[d] * row[d];
            }
            let e = score_ip(&fit, q, &q_rot, &eden_codes[i]);
            let m = score_minmax4_prefix(q, &mm_codes[i]);
            exact_prefix.push((i, exact));
            eden_est.push((i, e));
            mm_est.push((i, m));
            exact_vals.push(exact);
            eden_vals.push(e);
            mm_vals.push(m);
        }

        eden_corr += pearson(&eden_vals, &exact_vals) as f64;
        mm_corr += pearson(&mm_vals, &exact_vals) as f64;

        let truth10 = topk_ids(&exact_prefix, 10);
        eden_prefix_r10 += recall_at_k(&truth10, &topk_ids(&eden_est, 10));
        mm_prefix_r10 += recall_at_k(&truth10, &topk_ids(&mm_est, 10));

        let full_truth: Vec<usize> = suite
            .neighbors
            .get(qi)
            .map(|v| v.iter().copied().take(10).collect())
            .unwrap_or_default();
        if !full_truth.is_empty() {
            eden_route64 += recall_at_k(&full_truth, &topk_ids(&eden_est, 64));
            mm_route64 += recall_at_k(&full_truth, &topk_ids(&mm_est, 64));
            eden_route256 += recall_at_k(&full_truth, &topk_ids(&eden_est, 256));
            mm_route256 += recall_at_k(&full_truth, &topk_ids(&mm_est, 256));
        }
    }

    let nqf = n_q as f64;
    let report = serde_json::json!({
        "phase": 0,
        "data": data,
        "n": n,
        "n_queries": n_q,
        "bits": bits,
        "seed": seed,
        "prefix_recon_mse": { "eden": eden_mse, "minmax4": mm_mse },
        "prefix_ip_pearson": { "eden": eden_corr / nqf, "minmax4": mm_corr / nqf },
        "prefix_rank_R@10": { "eden": eden_prefix_r10 / nqf, "minmax4": mm_prefix_r10 / nqf },
        "full_top10_in_prefix_top64": { "eden": eden_route64 / nqf, "minmax4": mm_route64 / nqf },
        "full_top10_in_prefix_top256": { "eden": eden_route256 / nqf, "minmax4": mm_route256 / nqf },
        "eden_wins_mse": eden_mse < mm_mse,
        "eden_wins_pearson": (eden_corr / nqf) > (mm_corr / nqf),
        "eden_wins_prefix_R10": (eden_prefix_r10 / nqf) > (mm_prefix_r10 / nqf),
    });

    let out_dir = Path::new("bench_results/eden");
    std::fs::create_dir_all(out_dir)?;
    let out_path = out_dir.join("phase0_smoke.json");
    std::fs::write(&out_path, serde_json::to_string_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    eprintln!("wrote {}", out_path.display());
    Ok(())
}
