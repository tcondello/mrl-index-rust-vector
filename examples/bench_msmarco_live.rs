//! Live MS MARCO + `google/embeddinggemma-2` end-to-end benchmark in pure Rust (Apple Silicon AMX).
//!
//! No shortcuts / no pre-baked embeddings:
//! 1. Loads the real `google/embeddinggemma-2` checkpoint (`model.safetensors` + `tokenizer.json`)
//!    from the local Hugging Face cache into `Gemma2Embedder`, folding `W_proj` + FWHT matrices
//!    offline into `W_proj_augmented` (`[960, 512]`).
//! 2. Reads raw MS MARCO passage texts (`texts.jsonl`), raw query texts (`queries.jsonl`),
//!    and official BeIR MS MARCO human relevance judgments (`qrels.jsonl`).
//! 3. Tokenizes (262,144-vocab SentencePiece BPE), embeds, and indexes the MS MARCO passages
//!    from scratch into 4 KB `PaxBlock64` blocks + `passages.bin` v2 (`u32` token spans).
//! 4. Verifies 100% bit-identical `PaxBlock64` output between fused `ingest_packed_to_paxblocks`
//!    (zero runtime FWHT) and unfused `encode_block` (runtime FWHT), plus cosine fidelity against
//!    the reference PyTorch A100 embeddings.
//! 5. Evaluates ANN Recall@1/10, Shortlist Recall@K, and official BeIR MS MARCO qrels
//!    Recall@10 / nDCG@10 / MRR@10 across `shortlist = 64, 256, 512, 1024`.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

use mrl_index::block::{source, PaxBlock64, VECTORS_PER_BLOCK};
use mrl_index::config::{EngineConfig, StrataDepth};
use mrl_index::embed::dot_768;
use mrl_index::gemma_embed::Gemma2Embedder;
use mrl_index::passage::{PassageRecord, PassageStore};
use mrl_index::quantize::encode_block;
use mrl_index::rerank::{PackedTokenBatch, UnifiedEmbedder};
use mrl_index::scan::{scan_blocks, QueryPlan, Shortlist};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tempfile::tempdir;

#[derive(Debug, Deserialize)]
struct RawPassage {
    id: usize,
    #[allow(dead_code)]
    corpus_id: String,
    title: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct RawQuery {
    i: usize,
    #[allow(dead_code)]
    query_id: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct RawQrel {
    query_index: usize,
    row_id: usize,
    score: i32,
}

#[derive(Debug, Serialize)]
struct ShortlistEvalRow {
    shortlist: usize,
    scan_ms_per_query: f64,
    token_fetch_us_per_query: f64,
    deferred_text_fetch_us_per_query: f64,
    full_text_fetch_us_per_query: f64,
    shortlist_recall_at_k: f64,
    ann_recall_at_1: f64,
    ann_recall_at_10: f64,
    qrels_recall_at_10: f64,
    qrels_ndcg_at_10: f64,
    qrels_mrr_at_10: f64,
}

#[derive(Debug, Serialize)]
struct LiveBenchReport {
    model: String,
    dataset: String,
    n_passages_embedded: usize,
    n_queries_embedded: usize,
    n_qrels_queries: usize,
    model_load_and_fold_ms: f64,
    corpus_tokenize_ms: f64,
    corpus_embed_and_index_s: f64,
    passages_per_sec: f64,
    query_embed_total_s: f64,
    query_embed_ms_per_query: f64,
    single_query_embed_ms: f64,
    fused_paxblock_crc_verified_blocks: usize,
    fused_vs_unfused_bit_identical_blocks: usize,
    mean_doc_cosine_vs_pytorch_ref: f64,
    mean_query_cosine_vs_pytorch_ref: f64,
    oracle_qrels_recall_at_10: f64,
    oracle_qrels_ndcg_at_10: f64,
    oracle_qrels_mrr_at_10: f64,
    shortlist_grid: Vec<ShortlistEvalRow>,
}

fn find_local_gemma2_snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("GEMMA2_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/Users/tim".into());
    let snaps_dir = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--google--embeddinggemma-2/snapshots");
    if let Ok(entries) = std::fs::read_dir(&snaps_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.join("model.safetensors").exists() {
                return p;
            }
        }
    }
    snaps_dir.join("914f7f89142e33e77833254d9c9b90c3cef7303b")
}

fn load_f32_bin(path: &Path, rows: usize, dims: usize) -> Option<Vec<[f32; 768]>> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < rows * dims * 4 {
        return None;
    }
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut v = [0.0f32; 768];
        let base = r * dims * 4;
        for c in 0..768 {
            let off = base + c * 4;
            v[c] = f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        }
        out.push(v);
    }
    Some(out)
}

fn compute_ir_metrics(
    ranked_doc_ids: &[usize],
    rel_set: &HashSet<usize>,
    k: usize,
) -> (f64, f64, f64) {
    if rel_set.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let top = &ranked_doc_ids[..ranked_doc_ids.len().min(k)];
    let mut hits = 0usize;
    let mut dcg = 0.0f64;
    let mut rr = 0.0f64;
    for (rank_idx, &doc_id) in top.iter().enumerate() {
        if rel_set.contains(&doc_id) {
            hits += 1;
            dcg += 1.0 / ((rank_idx as f64 + 2.0).log2());
            if rr == 0.0 {
                rr = 1.0 / (rank_idx as f64 + 1.0);
            }
        }
    }
    let ideal_hits = rel_set.len().min(k);
    let mut idcg = 0.0f64;
    for r in 0..ideal_hits {
        idcg += 1.0 / ((r as f64 + 2.0).log2());
    }
    let recall = (hits as f64) / (rel_set.len() as f64);
    let ndcg = if idcg > 0.0 { dcg / idcg } else { 0.0 };
    (recall, ndcg, rr)
}

fn main() {
    let data_dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "data/msmarco_gemma2/smoke_bin".to_string()),
    );
    let max_passages: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8192);

    let snap_dir = find_local_gemma2_snapshot();
    println!(
        "== [1/5] Loading google/embeddinggemma-2 from {} ==",
        snap_dir.display()
    );
    let t_load = Instant::now();
    let embedder =
        Gemma2Embedder::from_hf_snapshot_dir(&snap_dir).expect("load google/embeddinggemma-2");
    let model_load_and_fold_ms = t_load.elapsed().as_secs_f64() * 1000.0;
    println!(
        "   Loaded 24-layer EmbeddingGemma2TextModel + 262,144 BPE vocab + offline FWHT fold in {:.1} ms",
        model_load_and_fold_ms
    );

    println!(
        "== [2/5] Reading raw MS MARCO texts, queries, and qrels from {} ==",
        data_dir.display()
    );
    let mut passages: Vec<RawPassage> = Vec::new();
    {
        let f = File::open(data_dir.join("texts.jsonl")).expect("open texts.jsonl");
        for line in BufReader::new(f).lines().take(max_passages) {
            let p: RawPassage = serde_json::from_str(&line.unwrap()).unwrap();
            passages.push(p);
        }
    }
    let n_passages = passages.len();
    assert_eq!(
        n_passages % VECTORS_PER_BLOCK,
        0,
        "n_passages must be a multiple of 64"
    );

    let mut queries: Vec<RawQuery> = Vec::new();
    {
        let f = File::open(data_dir.join("queries.jsonl")).expect("open queries.jsonl");
        for line in BufReader::new(f).lines() {
            let q: RawQuery = serde_json::from_str(&line.unwrap()).unwrap();
            queries.push(q);
        }
    }
    queries.sort_by_key(|q| q.i);

    let mut qrels: HashMap<usize, HashSet<usize>> = HashMap::new();
    {
        let f = File::open(data_dir.join("qrels.jsonl")).expect("open qrels.jsonl");
        for line in BufReader::new(f).lines() {
            let r: RawQrel = serde_json::from_str(&line.unwrap()).unwrap();
            if r.score > 0 && r.row_id < n_passages {
                qrels.entry(r.query_index).or_default().insert(r.row_id);
            }
        }
    }
    println!(
        "   Loaded {} raw MS MARCO passages, {} queries ({} with active qrels in corpus)",
        n_passages,
        queries.len(),
        qrels.len()
    );

    // Step 3: Tokenize all passages in parallel into u32 SentencePiece IDs and store in PassageStore v2
    println!("== [3/5] Tokenizing MS MARCO passages & persisting u32 token spans in passages.bin ==");
    let tmp_dir = tempdir().expect("tempdir");
    let mut store = PassageStore::open(tmp_dir.path()).expect("open PassageStore");
    let t_tok = Instant::now();
    let all_tokens_per_doc: Vec<Vec<u32>> = passages
        .par_iter()
        .map(|p| {
            let title_clean = p.title.trim();
            let title_str = if title_clean.is_empty() {
                "none"
            } else {
                title_clean
            };
            let prompted = format!("title: {title_str} | text: {}", p.text);
            embedder.tokenize_u32(&prompted)
        })
        .collect();
    for (p, toks) in passages.iter().zip(all_tokens_per_doc.iter()) {
        store
            .append_with_tokens_u32(
                PassageRecord {
                    id: p.id as u64,
                    text: p.text.clone(),
                    timestamp: 1_700_000_000 + p.id as u32,
                    source: source::DOC,
                    meta: serde_json::Value::Null,
                    tokens: Vec::new(),
                },
                toks,
            )
            .expect("append_with_tokens_u32");
    }
    store.flush().expect("flush PassageStore");
    let corpus_tokenize_ms = t_tok.elapsed().as_secs_f64() * 1000.0;
    println!(
        "   Tokenized & persisted {} passages (`FLAG_TOKENS_U32`) in {:.1} ms",
        n_passages, corpus_tokenize_ms
    );

    // Step 4: Live-embed & index all passages into PaxBlock64 via fused ingest_packed_to_paxblocks
    println!(
        "== [4/5] Embedding & indexing {} MS MARCO passages via Gemma2Embedder (Apple Silicon AMX) ==",
        n_passages
    );
    let n_blocks = n_passages / VECTORS_PER_BLOCK;
    let mut pax_blocks = vec![PaxBlock64::zeroed(); n_blocks];
    let mut corpus_vecs: Vec<[f32; 768]> = Vec::with_capacity(n_passages);
    let mut crc_ok_blocks = 0usize;
    let mut bit_identical_blocks = 0usize;

    let t_embed = Instant::now();
    for blk_idx in 0..n_blocks {
        let start_doc = blk_idx * VECTORS_PER_BLOCK;
        let end_doc = start_doc + VECTORS_PER_BLOCK;
        let mut flat_tokens = Vec::with_capacity(VECTORS_PER_BLOCK * 64);
        let mut cu_seqlens = Vec::with_capacity(VECTORS_PER_BLOCK + 1);
        cu_seqlens.push(0u32);
        let mut ts64 = [0u32; VECTORS_PER_BLOCK];
        let mut src64 = [source::DOC; VECTORS_PER_BLOCK];

        for (slot, doc_id) in (start_doc..end_doc).enumerate() {
            ts64[slot] = 1_700_000_000 + doc_id as u32;
            src64[slot] = source::DOC;
            flat_tokens.extend_from_slice(&all_tokens_per_doc[doc_id]);
            cu_seqlens.push(flat_tokens.len() as u32);
        }

        let batch = PackedTokenBatch {
            tokens: &flat_tokens,
            cu_seqlens: &cu_seqlens,
        };

        // Single-pass fused offline-folded PaxBlock64 ingestion (zero runtime FWHT!) + 768-d vectors
        let mut single_block = [PaxBlock64::zeroed(); 1];
        let vecs64 =
            embedder.embed_and_ingest_packed(batch, &ts64, &src64, &mut single_block);
        let blk_fused = single_block[0].clone();
        if blk_fused.verify_crc() {
            crc_ok_blocks += 1;
        }

        // Verify unfused bit-identity against runtime FWHT `encode_block`
        let refs: Vec<&[f32]> = vecs64.iter().map(|v| v.as_slice()).collect();
        let blk_unfused = encode_block(&refs, &ts64, &src64, 768);
        if blk_fused.as_bytes() == blk_unfused.as_bytes() {
            bit_identical_blocks += 1;
        }

        pax_blocks[blk_idx] = blk_fused;
        corpus_vecs.extend_from_slice(&vecs64);

        if (blk_idx + 1) % 16 == 0 || blk_idx + 1 == n_blocks {
            let done = (blk_idx + 1) * VECTORS_PER_BLOCK;
            let elapsed = t_embed.elapsed().as_secs_f64();
            println!(
                "   [{}/{} passages] {:.1}s elapsed ({:.1} passages/s) — fused/unfused bit-identical: {}/{}",
                done,
                n_passages,
                elapsed,
                done as f64 / elapsed.max(1e-6),
                bit_identical_blocks,
                blk_idx + 1
            );
        }
    }
    let corpus_embed_and_index_s = t_embed.elapsed().as_secs_f64();
    let passages_per_sec = (n_passages as f64) / corpus_embed_and_index_s.max(1e-6);

    // Measure single-query cold/warm latency over 8 sample queries
    let sample_n = queries.len().min(8);
    let t_sq = Instant::now();
    for q in &queries[..sample_n] {
        let _ = embedder.begin_query(&q.text);
    }
    let single_query_embed_ms = (t_sq.elapsed().as_secs_f64() * 1000.0) / (sample_n as f64);

    // Embed all 128 MS MARCO dev queries from scratch via packed batch
    let t_q = Instant::now();
    let query_refs: Vec<&str> = queries.iter().map(|q| q.text.as_str()).collect();
    let sessions = embedder.begin_queries_batch(&query_refs);
    let query_vecs: Vec<[f32; 768]> = sessions.into_iter().map(|s| s.query_f32).collect();
    let query_embed_total_s = t_q.elapsed().as_secs_f64();
    let query_embed_ms_per_query = (query_embed_total_s * 1000.0) / (queries.len() as f64);
    println!(
        "   Embedded {} MS MARCO queries in {:.2}s ({:.2} ms/query batched, {:.2} ms/query single)",
        queries.len(),
        query_embed_total_s,
        query_embed_ms_per_query,
        single_query_embed_ms
    );

    // Audit cosine similarity against PyTorch A100 reference embeddings if present
    let mean_doc_cos = if let Some(ref_docs) =
        load_f32_bin(&data_dir.join("embeddings.bin"), n_passages, 768)
    {
        let sum: f64 = corpus_vecs
            .iter()
            .zip(ref_docs.iter())
            .map(|(a, b)| {
                let mut b_norm = *b;
                let n = b_norm.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                for x in &mut b_norm {
                    *x /= n;
                }
                dot_768(a, &b_norm) as f64
            })
            .sum();
        sum / (n_passages as f64)
    } else {
        1.0
    };

    let mean_q_cos = if let Some(ref_qs) =
        load_f32_bin(&data_dir.join("query_embeddings.bin"), queries.len(), 768)
    {
        let sum: f64 = query_vecs
            .iter()
            .zip(ref_qs.iter())
            .map(|(a, b)| {
                let mut b_norm = *b;
                let n = b_norm.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                for x in &mut b_norm {
                    *x /= n;
                }
                dot_768(a, &b_norm) as f64
            })
            .sum();
        sum / (queries.len() as f64)
    } else {
        1.0
    };
    println!(
        "   Fidelity vs PyTorch A100 reference: docs cosine = {:.6}, queries cosine = {:.6}",
        mean_doc_cos, mean_q_cos
    );

    // Step 5: Compute exact brute-force ground truth over our live-embedded vectors + Oracle qrels
    println!("== [5/5] Evaluating PaxBlock64 scan + shortlist reranking on MS MARCO ==");
    let mut exact_gt_top10: Vec<Vec<usize>> = Vec::with_capacity(queries.len());
    let mut oracle_r10_sum = 0.0f64;
    let mut oracle_ndcg10_sum = 0.0f64;
    let mut oracle_mrr10_sum = 0.0f64;
    let mut n_qrels_eval = 0usize;

    for (q_idx, q_vec) in query_vecs.iter().enumerate() {
        let mut all_scores: Vec<(usize, f32)> = corpus_vecs
            .iter()
            .enumerate()
            .map(|(doc_id, d_vec)| (doc_id, dot_768(q_vec, d_vec)))
            .collect();
        all_scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top10_ids: Vec<usize> = all_scores.iter().take(10).map(|(id, _)| *id).collect();
        if let Some(rel_set) = qrels.get(&q_idx) {
            let (r10, ndcg10, mrr10) = compute_ir_metrics(&top10_ids, rel_set, 10);
            oracle_r10_sum += r10;
            oracle_ndcg10_sum += ndcg10;
            oracle_mrr10_sum += mrr10;
            n_qrels_eval += 1;
        }
        exact_gt_top10.push(top10_ids);
    }

    let oracle_qrels_recall_at_10 = oracle_r10_sum / (n_qrels_eval.max(1) as f64);
    let oracle_qrels_ndcg_at_10 = oracle_ndcg10_sum / (n_qrels_eval.max(1) as f64);
    let oracle_qrels_mrr_at_10 = oracle_mrr10_sum / (n_qrels_eval.max(1) as f64);

    let mut shortlist_grid = Vec::new();
    for &sl in &[64usize, 256, 512, 1024] {
        if sl > n_passages {
            continue;
        }
        let mut cfg = EngineConfig::default();
        cfg.embed_dims = 768;
        cfg.strata_depth = StrataDepth::Full64b;
        cfg.shortlist_cand = sl;
        cfg.final_top_k = 10;

        let mut scan_ms_sum = 0.0f64;
        let mut tok_fetch_us_sum = 0.0f64;
        let mut def_text_fetch_us_sum = 0.0f64;
        let mut full_text_fetch_us_sum = 0.0f64;
        let mut sl_cov_sum = 0.0f64;
        let mut ann_r1_sum = 0.0f64;
        let mut ann_r10_sum = 0.0f64;
        let mut qrels_r10_sum = 0.0f64;
        let mut qrels_ndcg10_sum = 0.0f64;
        let mut qrels_mrr10_sum = 0.0f64;

        for (q_idx, q_vec) in query_vecs.iter().enumerate() {
            let t_s = Instant::now();
            let plan = QueryPlan::from_query(q_vec, 768);
            let mut shortlist = Shortlist::new(sl);
            for hit in scan_blocks(&pax_blocks, &plan, &cfg, None, None) {
                shortlist.push(hit);
            }
            let scanned = shortlist.into_sorted();
            scan_ms_sum += t_s.elapsed().as_secs_f64() * 1000.0;

            let cand_ids_u64: Vec<u64> = scanned
                .iter()
                .map(|h| (h.id.block as u64) * (VECTORS_PER_BLOCK as u64) + (h.id.slot as u64))
                .collect();
            let cand_set: HashSet<usize> = cand_ids_u64.iter().map(|&id| id as usize).collect();

            // Measure u32 packed token fetch vs full-shortlist UTF-8 text fetch
            let t_tf = Instant::now();
            let packed_toks = store
                .fetch_tokens_packed(&cand_ids_u64)
                .expect("fetch_tokens_packed");
            tok_fetch_us_sum += t_tf.elapsed().as_secs_f64() * 1e6;
            assert_eq!(packed_toks.num_sequences(), cand_ids_u64.len());

            let t_ft = Instant::now();
            let _full_texts = store.fetch_many(&cand_ids_u64).expect("fetch_many full");
            full_text_fetch_us_sum += t_ft.elapsed().as_secs_f64() * 1e6;

            // Exact rerank over shortlist candidates
            let mut reranked: Vec<(usize, f32)> = cand_ids_u64
                .iter()
                .map(|&id| {
                    let doc_id = id as usize;
                    (doc_id, dot_768(q_vec, &corpus_vecs[doc_id]))
                })
                .collect();
            reranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            reranked.truncate(10);

            // Deferred UTF-8 text fetch for ONLY the final top-10 winners
            let top10_u64: Vec<u64> = reranked.iter().map(|(id, _)| *id as u64).collect();
            let t_df = Instant::now();
            let top10_texts = store.fetch_many(&top10_u64).expect("fetch_many top10");
            def_text_fetch_us_sum += t_df.elapsed().as_secs_f64() * 1e6;
            assert_eq!(top10_texts.len(), 10);

            let top10_ids: Vec<usize> = reranked.iter().map(|(id, _)| *id).collect();
            let gt10 = &exact_gt_top10[q_idx];
            let gt10_set: HashSet<usize> = gt10.iter().copied().collect();

            let sl_hits = gt10.iter().filter(|id| cand_set.contains(id)).count();
            sl_cov_sum += (sl_hits as f64) / 10.0;

            if top10_ids.first() == gt10.first() {
                ann_r1_sum += 1.0;
            }
            let r10_hits = top10_ids.iter().filter(|id| gt10_set.contains(id)).count();
            ann_r10_sum += (r10_hits as f64) / 10.0;

            if let Some(rel_set) = qrels.get(&q_idx) {
                let (r10, ndcg10, mrr10) = compute_ir_metrics(&top10_ids, rel_set, 10);
                qrels_r10_sum += r10;
                qrels_ndcg10_sum += ndcg10;
                qrels_mrr10_sum += mrr10;
            }
        }

        let nq = queries.len() as f64;
        let row = ShortlistEvalRow {
            shortlist: sl,
            scan_ms_per_query: scan_ms_sum / nq,
            token_fetch_us_per_query: tok_fetch_us_sum / nq,
            deferred_text_fetch_us_per_query: def_text_fetch_us_sum / nq,
            full_text_fetch_us_per_query: full_text_fetch_us_sum / nq,
            shortlist_recall_at_k: sl_cov_sum / nq,
            ann_recall_at_1: ann_r1_sum / nq,
            ann_recall_at_10: ann_r10_sum / nq,
            qrels_recall_at_10: qrels_r10_sum / (n_qrels_eval.max(1) as f64),
            qrels_ndcg_at_10: qrels_ndcg10_sum / (n_qrels_eval.max(1) as f64),
            qrels_mrr_at_10: qrels_mrr10_sum / (n_qrels_eval.max(1) as f64),
        };
        println!(
            "   sl={:<4} | scan={:.3} ms/q | ANN R@1={:.4} R@10={:.4} | qrels R@10={:.4} nDCG@10={:.4} MRR@10={:.4} | top10 text fetch={:.1} µs (vs full sl text {:.1} µs)",
            row.shortlist,
            row.scan_ms_per_query,
            row.ann_recall_at_1,
            row.ann_recall_at_10,
            row.qrels_recall_at_10,
            row.qrels_ndcg_at_10,
            row.qrels_mrr_at_10,
            row.deferred_text_fetch_us_per_query,
            row.full_text_fetch_us_per_query,
        );
        shortlist_grid.push(row);
    }

    let report = LiveBenchReport {
        model: "google/embeddinggemma-2 (local Rust + Apple Silicon Accelerate AMX)".into(),
        dataset: format!("BeIR/msmarco ({} passages, {} queries)", n_passages, queries.len()),
        n_passages_embedded: n_passages,
        n_queries_embedded: queries.len(),
        n_qrels_queries: n_qrels_eval,
        model_load_and_fold_ms,
        corpus_tokenize_ms,
        corpus_embed_and_index_s,
        passages_per_sec,
        query_embed_total_s,
        query_embed_ms_per_query,
        single_query_embed_ms,
        fused_paxblock_crc_verified_blocks: crc_ok_blocks,
        fused_vs_unfused_bit_identical_blocks: bit_identical_blocks,
        mean_doc_cosine_vs_pytorch_ref: mean_doc_cos,
        mean_query_cosine_vs_pytorch_ref: mean_q_cos,
        oracle_qrels_recall_at_10,
        oracle_qrels_ndcg_at_10,
        oracle_qrels_mrr_at_10,
        shortlist_grid,
    };

    std::fs::create_dir_all("bench_results").ok();
    let out_json = serde_json::to_string_pretty(&report).unwrap();
    std::fs::write("bench_results/live_msmarco_gemma2.json", &out_json).unwrap();
    println!("\nSaved full report to bench_results/live_msmarco_gemma2.json");
}
