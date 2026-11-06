//! Phase 1: Pax-integrated EDEN-prod prefix vs min–max on smoke retrieval.
//!
//! Usage:
//!   cargo run --release --example eden_phase1 -- data/msmarco_gemma2/smoke_bin [shortlist]

use std::collections::HashSet;
use std::env;
use std::path::Path;
use std::time::Instant;

use mrl_index::block::source;
use mrl_index::config::{EngineConfig, PrefixCodec, StrataDepth};
use mrl_index::eden::EdenFit;
use mrl_index::encoder::{encode_f32_corpus_par, EdenPrefixEncoder, QuantizeEncoder};
use mrl_index::pipeline::{scored_hits_to_row_ids, QueryPipeline};
use mrl_index::retrieval_bench::RetrievalBenchData;
use mrl_index::scan::ScanStats;

fn recall_at_k(truth: &HashSet<usize>, pred: &[usize], k: usize) -> f64 {
    if truth.is_empty() {
        return 1.0;
    }
    let take = k.min(pred.len());
    pred[..take].iter().filter(|i| truth.contains(i)).count() as f64 / truth.len() as f64
}

fn eval_variant(
    label: &str,
    data: &RetrievalBenchData,
    prefix_codec: PrefixCodec,
    fit: Option<&EdenFit>,
    shortlist: usize,
) -> serde_json::Value {
    let dims = data.dims();
    let n = data.n_corpus();
    let n_queries = data.n_queries();

    let t_enc = Instant::now();
    let blocks = match (prefix_codec, fit) {
        (PrefixCodec::EdenProd, Some(f)) => {
            let enc = EdenPrefixEncoder::new(dims, f.clone());
            encode_f32_corpus_par(
                &enc,
                data.corpus_embeddings.as_slice(),
                n,
                source::DOC,
            )
        }
        _ => {
            let enc = QuantizeEncoder::new(dims);
            encode_f32_corpus_par(
                &enc,
                data.corpus_embeddings.as_slice(),
                n,
                source::DOC,
            )
        }
    };
    let encode_ms = t_enc.elapsed().as_secs_f64() * 1000.0;

    let mut cfg = EngineConfig::default();
    cfg.embed_dims = dims;
    cfg.shortlist_cand = shortlist;
    cfg.strata_depth = StrataDepth::Full64b;
    cfg.temporal_filter_enable = false;
    cfg.prefix_codec = prefix_codec;

    let pipe = match fit {
        Some(f) if prefix_codec == PrefixCodec::EdenProd => QueryPipeline::with_eden(&cfg, f),
        _ => QueryPipeline::new(&cfg),
    };

    let mut coverage10 = 0.0;
    let mut rerank10 = 0.0;
    let mut scan_ms_sum = 0.0;
    let mut scan_stats = ScanStats::default();

    for qi in 0..n_queries {
        let q = data.query_vector(qi);
        let truth10: HashSet<usize> = data.neighbors[qi].iter().take(10).copied().collect();
        let phase = pipe.scan_slab(q, &blocks, None, None, Some(&mut scan_stats));
        scan_ms_sum += phase.scan_ms;
        let ids = scored_hits_to_row_ids(&phase.hits);
        coverage10 += truth10.iter().filter(|i| ids.contains(i)).count() as f64 / 10.0;
        let reranked = QueryPipeline::rerank_exact_rows(q, &ids, |i| data.corpus_vector(i));
        let rerank_ids: Vec<usize> = reranked.into_iter().map(|(i, _)| i).collect();
        rerank10 += recall_at_k(&truth10, &rerank_ids, 10);
    }

    let mean_scan_ms = scan_ms_sum / n_queries as f64;
    let mvecs = (n as f64 / (mean_scan_ms / 1000.0)) / 1e6;
    let keep_prefix = if scan_stats.active_vecs > 0 {
        100.0 * scan_stats.after_prefix as f64 / scan_stats.active_vecs as f64
    } else {
        0.0
    };

    serde_json::json!({
        "label": label,
        "prefix_codec": format!("{prefix_codec:?}"),
        "n": n,
        "n_queries": n_queries,
        "shortlist": shortlist,
        "encode_ms": encode_ms,
        "mean_scan_ms": mean_scan_ms,
        "mvecs_per_sec": mvecs,
        "shortlist_coverage_at_10": coverage10 / n_queries as f64,
        "reranked_recall_at_10": rerank10 / n_queries as f64,
        "keep_after_prefix_pct": keep_prefix,
    })
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
    let bits = 4u8;

    let data = RetrievalBenchData::load(Path::new(&data_dir))?;
    let n = data.n_corpus();
    eprintln!("fitting EDEN mean on n={n} prefixes …");
    let fit = EdenFit::fit_prefix((0..n).map(|i| data.corpus_vector(i)), bits, seed);

    eprintln!("=== baseline min–max ===");
    let baseline = eval_variant(
        "phase1-minmax",
        &data,
        PrefixCodec::MinMax4,
        None,
        shortlist,
    );
    eprintln!("{baseline}");

    eprintln!("=== EDEN-prod prefix ===");
    let eden = eval_variant(
        "phase1-eden",
        &data,
        PrefixCodec::EdenProd,
        Some(&fit),
        shortlist,
    );
    eprintln!("{eden}");

    let base_r = baseline["reranked_recall_at_10"].as_f64().unwrap();
    let eden_r = eden["reranked_recall_at_10"].as_f64().unwrap();
    let base_c = baseline["shortlist_coverage_at_10"].as_f64().unwrap();
    let eden_c = eden["shortlist_coverage_at_10"].as_f64().unwrap();
    let base_ms = baseline["mean_scan_ms"].as_f64().unwrap();
    let eden_ms = eden["mean_scan_ms"].as_f64().unwrap();

    let report = serde_json::json!({
        "phase": 1,
        "data": data_dir,
        "shortlist": shortlist,
        "bits": bits,
        "seed": seed,
        "baseline": baseline,
        "eden": eden,
        "delta_reranked_R10": eden_r - base_r,
        "delta_coverage10": eden_c - base_c,
        "scan_ms_ratio": eden_ms / base_ms.max(1e-9),
        "eden_wins_recall": eden_r > base_r + 0.005,
        "eden_ok_latency": eden_ms <= base_ms * 1.15,
        "go_phase2": (eden_r >= base_r - 0.002) && (eden_ms <= base_ms * 1.20),
    });

    let out_dir = Path::new("bench_results/eden");
    std::fs::create_dir_all(out_dir)?;
    let out = out_dir.join("phase1_smoke.json");
    std::fs::write(&out, serde_json::to_string_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    eprintln!("wrote {}", out.display());
    Ok(())
}
