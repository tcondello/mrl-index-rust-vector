//! Follow-up: EDEN prefix b=4 + EDEN mid b=1 vs baseline / prefix-only on smoke.
//!
//! Usage:
//!   cargo run --release --example eden_followup_dual -- data/msmarco_gemma2/smoke_bin [shortlist]

use std::collections::HashSet;
use std::env;
use std::path::Path;
use std::time::Instant;

use mrl_index::block::{source, MRL_PREFIX_DIMS, RABITQ_DIMS};
use mrl_index::config::{EngineConfig, MidCodec, PrefixCodec, StrataDepth};
use mrl_index::eden::EdenFit;
use mrl_index::encoder::{
    encode_f32_corpus_par, EdenDualEncoder, EdenPrefixEncoder, QuantizeEncoder,
};
use mrl_index::pipeline::{scored_hits_to_row_ids, QueryPipeline};
use mrl_index::retrieval_bench::RetrievalBenchData;

fn recall_at_k(truth: &HashSet<usize>, pred: &[usize], k: usize) -> f64 {
    if truth.is_empty() {
        return 1.0;
    }
    let take = k.min(pred.len());
    pred[..take].iter().filter(|i| truth.contains(i)).count() as f64 / truth.len() as f64
}

#[derive(Clone, Copy, Debug)]
enum Variant {
    MinMax,
    EdenPrefix,
    EdenDual,
}

fn main() -> anyhow::Result<()> {
    let data_dir = env::args()
        .nth(1)
        .unwrap_or_else(|| "data/msmarco_gemma2/smoke_bin".into());
    let shortlist: usize = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let seed = 1u64;
    let skip_prefix_only = env::var("EDEN_DUAL_SKIP_PREFIX").ok().as_deref() == Some("1");

    let data = RetrievalBenchData::load(Path::new(&data_dir))?;
    let n = data.n_corpus();
    let n_queries = env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| data.n_queries())
        .min(data.n_queries());
    eprintln!("fitting prefix+mid EDEN on n={n} …");
    let prefix_fit = EdenFit::fit_prefix((0..n).map(|i| data.corpus_vector(i)), 4, seed);
    let mid_fit = EdenFit::fit_band(
        (0..n).map(|i| data.corpus_vector(i)),
        MRL_PREFIX_DIMS,
        RABITQ_DIMS,
        1,
        seed,
    );

    // Patch n_queries into eval by temporarily cloning isn't available; pass via
    // rewriting eval call sites to use n_queries from outer by limiting loops —
    // simplest: shadow with a thin wrapper using only first n_queries in eval.
    let baseline = eval_n(
        "minmax",
        &data,
        Variant::MinMax,
        &prefix_fit,
        &mid_fit,
        shortlist,
        n_queries,
    );
    eprintln!("baseline: {baseline}");
    let eden_p = if skip_prefix_only {
        serde_json::Value::Null
    } else {
        let v = eval_n(
            "eden-prefix",
            &data,
            Variant::EdenPrefix,
            &prefix_fit,
            &mid_fit,
            shortlist,
            n_queries,
        );
        eprintln!("eden-prefix: {v}");
        v
    };
    let eden_d = eval_n(
        "eden-dual",
        &data,
        Variant::EdenDual,
        &prefix_fit,
        &mid_fit,
        shortlist,
        n_queries,
    );
    eprintln!("eden-dual: {eden_d}");

    let out_name = if n >= 100_000 {
        "followup_dual_1m.json"
    } else {
        "followup_dual_smoke.json"
    };
    let report = serde_json::json!({
        "followup": "eden_dual",
        "data": data_dir,
        "shortlist": shortlist,
        "n_queries": n_queries,
        "seed": seed,
        "baseline": baseline,
        "eden_prefix": eden_p,
        "eden_dual": eden_d,
        "dual_vs_baseline_R10": eden_d["reranked_recall_at_10"].as_f64().unwrap()
            - baseline["reranked_recall_at_10"].as_f64().unwrap(),
        "dual_scan_ms_ratio_vs_baseline": eden_d["mean_scan_ms"].as_f64().unwrap()
            / baseline["mean_scan_ms"].as_f64().unwrap().max(1e-9),
    });
    let out_dir = Path::new("bench_results/eden");
    std::fs::create_dir_all(out_dir)?;
    let out = out_dir.join(out_name);
    std::fs::write(&out, serde_json::to_string_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    eprintln!("wrote {}", out.display());
    Ok(())
}

fn eval_n(
    label: &str,
    data: &RetrievalBenchData,
    variant: Variant,
    prefix_fit: &EdenFit,
    mid_fit: &EdenFit,
    shortlist: usize,
    n_queries: usize,
) -> serde_json::Value {
    let dims = data.dims();
    let n = data.n_corpus();

    let t_enc = Instant::now();
    let blocks = match variant {
        Variant::MinMax => {
            let enc = QuantizeEncoder::new(dims);
            encode_f32_corpus_par(&enc, data.corpus_embeddings.as_slice(), n, source::DOC)
        }
        Variant::EdenPrefix => {
            let enc = EdenPrefixEncoder::new(dims, prefix_fit.clone());
            encode_f32_corpus_par(&enc, data.corpus_embeddings.as_slice(), n, source::DOC)
        }
        Variant::EdenDual => {
            let enc = EdenDualEncoder::new(dims, prefix_fit.clone(), mid_fit.clone());
            encode_f32_corpus_par(&enc, data.corpus_embeddings.as_slice(), n, source::DOC)
        }
    };
    let encode_ms = t_enc.elapsed().as_secs_f64() * 1000.0;

    let mut cfg = EngineConfig::default();
    cfg.embed_dims = dims;
    cfg.shortlist_cand = shortlist;
    cfg.strata_depth = StrataDepth::Full64b;
    cfg.temporal_filter_enable = false;
    match variant {
        Variant::MinMax => {
            cfg.prefix_codec = PrefixCodec::MinMax4;
            cfg.mid_codec = MidCodec::Rabitq;
        }
        Variant::EdenPrefix => {
            cfg.prefix_codec = PrefixCodec::EdenProd;
            cfg.mid_codec = MidCodec::Rabitq;
        }
        Variant::EdenDual => {
            cfg.prefix_codec = PrefixCodec::EdenProd;
            cfg.mid_codec = MidCodec::EdenB1;
        }
    }

    let pipe = match variant {
        Variant::MinMax => QueryPipeline::new(&cfg),
        Variant::EdenPrefix => QueryPipeline::with_eden(&cfg, prefix_fit),
        Variant::EdenDual => QueryPipeline::with_eden_dual(&cfg, prefix_fit, mid_fit),
    };

    let mut coverage10 = 0.0;
    let mut rerank10 = 0.0;
    let mut scan_ms_sum = 0.0;
    let progress_every = (n_queries / 10).max(1);
    for qi in 0..n_queries {
        let q = data.query_vector(qi);
        let truth10: HashSet<usize> = data.neighbors[qi].iter().take(10).copied().collect();
        let phase = pipe.scan_slab(q, &blocks, None, None, None);
        scan_ms_sum += phase.scan_ms;
        let ids = scored_hits_to_row_ids(&phase.hits);
        coverage10 += truth10.iter().filter(|i| ids.contains(i)).count() as f64 / 10.0;
        let reranked = QueryPipeline::rerank_exact_rows(q, &ids, |i| data.corpus_vector(i));
        let rerank_ids: Vec<usize> = reranked.into_iter().map(|(i, _)| i).collect();
        rerank10 += recall_at_k(&truth10, &rerank_ids, 10);
        if (qi + 1) % progress_every == 0 || qi + 1 == n_queries {
            eprintln!(
                "  [{label}] {}/{n_queries}  mean_scan={:.1} ms/q",
                qi + 1,
                scan_ms_sum / (qi + 1) as f64
            );
        }
    }

    let mean_scan_ms = scan_ms_sum / n_queries as f64;
    serde_json::json!({
        "label": label,
        "variant": format!("{variant:?}"),
        "n": n,
        "n_queries": n_queries,
        "shortlist": shortlist,
        "encode_ms": encode_ms,
        "mean_scan_ms": mean_scan_ms,
        "shortlist_coverage_at_10": coverage10 / n_queries as f64,
        "reranked_recall_at_10": rerank10 / n_queries as f64,
    })
}
