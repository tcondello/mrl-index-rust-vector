//! Follow-up: EDEN-prod b=1 on the 128-d mid-band vs current RaBitQ signs (offline).
//!
//! Usage:
//!   cargo run --release --example eden_followup_midband -- data/msmarco_gemma2/smoke_bin [max_q]

use std::collections::HashSet;
use std::env;
use std::path::Path;

use mrl_index::block::{MRL_PREFIX_DIMS, RABITQ_DIMS};
use mrl_index::eden::{
    encode_slice, rotate_query, score_ip, EdenFit,
};
use mrl_index::retrieval_bench::RetrievalBenchData;
use mrl_index::rotate::{rabitq_bit_scale, rotate_mid_band};

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

fn mid_slice(row: &[f32]) -> Vec<f32> {
    row[MRL_PREFIX_DIMS..MRL_PREFIX_DIMS + RABITQ_DIMS].to_vec()
}

fn score_rabitq_mid(q_mid: &[f32], x_mid: &[f32], bits: &[u8]) -> f32 {
    // Match scan: (rabitq_base + Σ bit * 2pos) * ‖x_mid‖ with rotated query.
    let q_rot = rotate_mid_band(q_mid);
    let scale = rabitq_bit_scale();
    let mut base = 0.0f32;
    let mut acc = 0.0f32;
    for d in 0..RABITQ_DIMS {
        let pos = q_rot[d] * scale;
        base -= pos;
        if bits[d] != 0 {
            acc += 2.0 * pos;
        }
    }
    let x_norm: f32 = x_mid.iter().map(|t| t * t).sum::<f32>().sqrt().max(1e-6);
    (base + acc) * x_norm
}

fn encode_rabitq_bits(x_mid: &[f32]) -> [u8; RABITQ_DIMS] {
    let rot = rotate_mid_band(x_mid);
    let mut bits = [0u8; RABITQ_DIMS];
    for d in 0..RABITQ_DIMS {
        bits[d] = if rot[d] >= 0.0 { 1 } else { 0 };
    }
    bits
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

    // Fit EDEN on mid-band slices.
    let mids: Vec<Vec<f32>> = (0..n).map(|i| mid_slice(data.corpus_vector(i))).collect();
    let fit = EdenFit::fit_slice(mids.iter().map(|m| m.as_slice()), RABITQ_DIMS, 1, seed);
    let eden_codes: Vec<_> = mids.iter().map(|m| encode_slice(&fit, m)).collect();
    let rabitq_bits: Vec<_> = mids.iter().map(|m| encode_rabitq_bits(m)).collect();

    let mut eden_corr = 0.0f64;
    let mut rq_corr = 0.0f64;
    let mut eden_r10 = 0.0f64;
    let mut rq_r10 = 0.0f64;

    for qi in 0..n_q {
        let q = data.query_vector(qi);
        let q_mid = mid_slice(q);
        let q_rot = rotate_query(&fit, &q_mid);

        let mut exact = Vec::with_capacity(n);
        let mut eden_est = Vec::with_capacity(n);
        let mut rq_est = Vec::with_capacity(n);
        let mut exact_v = Vec::with_capacity(n);
        let mut eden_v = Vec::with_capacity(n);
        let mut rq_v = Vec::with_capacity(n);

        for i in 0..n {
            let x_mid = &mids[i];
            let mut e = 0.0f32;
            for d in 0..RABITQ_DIMS {
                e += q_mid[d] * x_mid[d];
            }
            let es = score_ip(&fit, &q_mid, &q_rot, &eden_codes[i]);
            let rs = score_rabitq_mid(&q_mid, x_mid, &rabitq_bits[i]);
            exact.push((i, e));
            eden_est.push((i, es));
            rq_est.push((i, rs));
            exact_v.push(e);
            eden_v.push(es);
            rq_v.push(rs);
        }

        eden_corr += pearson(&eden_v, &exact_v) as f64;
        rq_corr += pearson(&rq_v, &exact_v) as f64;
        let truth = topk_ids(&exact, 10);
        eden_r10 += recall_at_k(&truth, &topk_ids(&eden_est, 10));
        rq_r10 += recall_at_k(&truth, &topk_ids(&rq_est, 10));
    }

    let nqf = n_q as f64;
    let report = serde_json::json!({
        "followup": "midband_eden_b1_vs_rabitq",
        "data": data_dir,
        "n": n,
        "n_queries": n_q,
        "seed": seed,
        "mid_ip_pearson": {
            "eden_b1": eden_corr / nqf,
            "rabitq": rq_corr / nqf,
        },
        "mid_rank_R@10": {
            "eden_b1": eden_r10 / nqf,
            "rabitq": rq_r10 / nqf,
        },
        "eden_wins_pearson": (eden_corr / nqf) > (rq_corr / nqf),
        "eden_wins_R10": (eden_r10 / nqf) > (rq_r10 / nqf),
    });

    let out_dir = Path::new("bench_results/eden");
    std::fs::create_dir_all(out_dir)?;
    let out = out_dir.join("followup_midband_smoke.json");
    std::fs::write(&out, serde_json::to_string_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    eprintln!("wrote {}", out.display());
    Ok(())
}
