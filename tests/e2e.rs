use mrl_index::block::source;
use mrl_index::config::{EngineConfig, StrataDepth};
use mrl_index::IndexEngine;
use tempfile::tempdir;

#[test]
fn block_layout_is_4kib() {
    assert_eq!(std::mem::size_of::<mrl_index::PaxBlock64>(), 4096);
    assert_eq!(std::mem::align_of::<mrl_index::PaxBlock64>(), 4096);
}

#[test]
fn ingest_flush_query_roundtrip() {
    let dir = tempdir().unwrap();
    let mut cfg = EngineConfig::default();
    cfg.embed_dims = 256;
    cfg.strata_depth = StrataDepth::Full64b;
    cfg.shortlist_cand = 32;
    cfg.final_top_k = 3;
    cfg.ring_buffer_size = 256;

    let mut engine = IndexEngine::create(dir.path(), cfg).unwrap();

    let docs = [
        "bitplane transposed scan uses popcnt across macro blocks",
        "cooking pasta with tomato sauce and basil",
        "matryoshka representation learning for compact prefixes",
        "weather forecast rain tomorrow morning",
        "cauchy schwarz bound prunes cold vector blocks",
        "slack message about lunch plans downtown",
    ];
    for (i, d) in docs.iter().enumerate() {
        engine
            .ingest((*d).to_string(), 1000 + i as u32, source::DOC)
            .unwrap();
    }
    engine.flush().unwrap();
    assert_eq!(engine.passage_count(), docs.len() as u64);
    assert!(engine.block_count() >= 1);

    let result = engine.query("bitplane popcnt matryoshka retrieval").unwrap();
    assert!(!result.hits.is_empty());
    // Top hit should be retrieval-related, not pasta/weather.
    let top = &result.hits[0].text;
    assert!(
        top.contains("bitplane")
            || top.contains("matryoshka")
            || top.contains("cauchy"),
        "unexpected top hit: {top}"
    );
}

#[test]
fn ring_is_queryable_before_flush() {
    let dir = tempdir().unwrap();
    let mut cfg = EngineConfig::default();
    cfg.embed_dims = 256;
    let mut engine = IndexEngine::create(dir.path(), cfg).unwrap();
    engine
        .ingest("unique zucchini spectrometer calibration", 1, source::DOC)
        .unwrap();
    // No flush — still in ring.
    assert_eq!(engine.block_count(), 0);
    let result = engine.query("zucchini spectrometer").unwrap();
    assert_eq!(result.hits[0].passage_id, 0);
}

#[test]
fn reopen_preserves_search() {
    let dir = tempdir().unwrap();
    let mut cfg = EngineConfig::default();
    cfg.embed_dims = 256;
    cfg.strata_depth = StrataDepth::L1Rabitq;

    {
        let mut engine = IndexEngine::create(dir.path(), cfg).unwrap();
        for i in 0..70 {
            engine
                .ingest(
                    format!("document number {i} about orbital mechanics"),
                    i as u32,
                    source::DOC,
                )
                .unwrap();
        }
        engine.flush().unwrap();
    }

    let engine = IndexEngine::open(dir.path()).unwrap();
    assert!(engine.block_count() >= 1);
    let result = engine.query("orbital mechanics").unwrap();
    assert!(!result.hits.is_empty());
}

#[test]
fn binary_passage_store_roundtrip_and_batch_order() {
    use mrl_index::{PassageRecord, PassageStore};

    let dir = tempdir().unwrap();
    let mut store = PassageStore::open(dir.path()).unwrap();
    for i in 0..10u64 {
        let text = format!("passage-{i} with quotes \"and\"\nnewlines");
        let tokens = mrl_index::embed::tokenize_u16(&text);
        store
            .append_with_tokens(
                PassageRecord {
                    id: 0,
                    text,
                    timestamp: 100 + i as u32,
                    source: source::CODE,
                    meta: serde_json::json!({"ord": i}),
                    tokens: Vec::new(),
                },
                &tokens,
            )
            .unwrap();
    }
    store.flush().unwrap();

    let rec = store.fetch_record(3).unwrap();
    assert_eq!(rec.id, 3);
    assert_eq!(rec.timestamp, 103);
    assert_eq!(rec.text, "passage-3 with quotes \"and\"\nnewlines");
    assert_eq!(rec.meta["ord"], 3);
    assert!(!rec.tokens.is_empty());

    // Reverse ID order ensures LBA-sorted fetch_many restores caller order.
    let batch = store.fetch_many(&[8, 2, 5, 0]).unwrap();
    let ids: Vec<u64> = batch.iter().map(|(id, _)| *id).collect();
    assert_eq!(ids, vec![8, 2, 5, 0]);
    assert!(batch[1].1.starts_with("passage-2 "));

    // Token spans: shortlist path reads u16 IDs without UTF-8.
    let tok_batch = store.fetch_tokens_many(&[8, 2, 5, 0]).unwrap();
    assert_eq!(
        tok_batch.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        vec![8, 2, 5, 0]
    );
    assert_eq!(tok_batch[1].1, store.fetch_tokens(2).unwrap());

    store.pad_to_page().unwrap();
    let data_len = std::fs::metadata(dir.path().join("passages.bin"))
        .unwrap()
        .len();
    assert_eq!(data_len % 16384, 0);
}

#[test]
fn page_align_helper() {
    assert_eq!(mrl_index::passage::align_page(1), 16384);
    assert_eq!(mrl_index::passage::align(100, 64), 128);
}

#[test]
fn unified_gemma2_ingest_and_query_roundtrip() {
    use mrl_index::{Gemma2Config, Gemma2Embedder};

    let dir = tempdir().unwrap();
    let mut cfg = EngineConfig::default();
    cfg.embed_dims = 768;
    cfg.strata_depth = StrataDepth::Full64b;
    cfg.shortlist_cand = 64;
    cfg.final_top_k = 5;

    let embedder = Gemma2Embedder::new_deterministic(Gemma2Config::micro_test(), 2026);
    let mut engine = IndexEngine::create(dir.path(), cfg)
        .unwrap()
        .with_gemma2_embedder(embedder);

    // 64 passages flush directly via zero-FWHT `ingest_packed_to_paxblocks`;
    // 6 tail passages remain in the ring buffer to test hybrid disk + ring unified search.
    let mut items = Vec::with_capacity(70);
    for i in 0..70 {
        let text = if i == 42 {
            "quantum superconducting transmon qubit coherence time".to_string()
        } else if i == 67 {
            "ring buffer instant visibility for freshly ingested passages".to_string()
        } else {
            format!("synthetic passage number {i} discussing distributed systems and consensus")
        };
        items.push((text, 1000 + i as u32, source::DOC));
    }

    let ids = engine.ingest_batch(&items).unwrap();
    assert_eq!(ids.len(), 70);
    assert_eq!(engine.block_count(), 1);
    assert_eq!(engine.passage_count(), 70);

    // Query a passage stored in the flushed PaxBlock64 on disk (via u32 token shortlist)
    let res_disk = engine
        .query("quantum superconducting transmon qubit coherence time")
        .unwrap();
    assert_eq!(res_disk.hits.len(), 5);
    assert_eq!(res_disk.hits[0].passage_id, 42);
    assert!(
        res_disk.hits[0].text.contains("transmon qubit"),
        "deferred top-k text fetch failed: {}",
        res_disk.hits[0].text
    );

    // Query a passage still residing in the uncommitted ring buffer.
    // Note: query uses the asymmetric pinned `PrefixKVCache` ("task: search result | query: "),
    // while passages are embedded without the query prefix, yielding cosine ~0.81 and rank #1.
    let res_ring = engine
        .query("ring buffer instant visibility for freshly ingested passages")
        .unwrap();
    assert_eq!(res_ring.hits.len(), 5);
    assert_eq!(res_ring.hits[0].passage_id, 67);
    assert!(
        res_ring.hits[0].exact_score > 0.75,
        "expected top-1 asymmetric query-doc dot > 0.75, got {}",
        res_ring.hits[0].exact_score
    );
}
