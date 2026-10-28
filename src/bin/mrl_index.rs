//! CLI for mrl-index: init, ingest, query, demo, bench.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use mrl_index::block::source;
use mrl_index::config::{EngineConfig, SlabResidency, StrataDepth};
use mrl_index::IndexEngine;

#[derive(Parser, Debug)]
#[command(name = "mrl-index", version, about = "Bitplane-transposed local vector retrieval engine")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Create a new index directory.
    Init {
        dir: PathBuf,
        #[arg(long, default_value_t = 384)]
        dims: usize,
        #[arg(long, default_value = "l1_rabitq")]
        strata: String,
        #[arg(long, default_value_t = 64)]
        shortlist: usize,
    },
    /// Append a text passage (or lines from --file).
    Ingest {
        dir: PathBuf,
        #[arg(long)]
        text: Option<String>,
        #[arg(long)]
        file: Option<PathBuf>,
        #[arg(long, default_value = "doc")]
        source: String,
        #[arg(long, default_value_t = 0)]
        timestamp: u32,
        #[arg(long)]
        flush: bool,
    },
    /// Run a retrieval query.
    Query {
        dir: PathBuf,
        query: String,
        #[arg(long)]
        time_lo: Option<u32>,
        #[arg(long)]
        time_hi: Option<u32>,
    },
    /// Synthetic end-to-end demo (ingest + query + timings).
    Demo {
        #[arg(long, default_value = "/tmp/mrl-index-demo")]
        dir: PathBuf,
        #[arg(long, default_value_t = 2_048)]
        docs: usize,
        #[arg(long, default_value_t = 384)]
        dims: usize,
    },
    /// Micro-benchmark scan throughput on synthetic blocks.
    Bench {
        #[arg(long, default_value_t = 2_048)]
        blocks: usize,
        #[arg(long, default_value_t = 384)]
        dims: usize,
    },
    /// Full recall + ingest suite against Modal MRL embeddings (or --synthetic plumbing).
    BenchSuite {
        #[arg(long, default_value = "run")]
        label: String,
        #[arg(long)]
        out: Option<PathBuf>,
        /// Directory with manifest.json + embeddings.bin + texts.jsonl from Modal.
        #[arg(long, default_value = "data/sec10k_mrl")]
        embeddings: PathBuf,
        /// Use isotropic random vectors (NOT an MRL evaluation — plumbing only).
        #[arg(long)]
        synthetic: bool,
    },
    /// Clean MS MARCO retrieval bench: query emb + exact neighbors + qrels (no self-retrieval).
    BenchRetrieval {
        #[arg(long, default_value = "msmarco")]
        label: String,
        #[arg(long)]
        out: Option<PathBuf>,
        /// Binary bench dir from scripts/parquet_to_bench_bin.py
        #[arg(long, default_value = "data/msmarco_gemma2/smoke_bin")]
        data: PathBuf,
        /// Cap queries (default: all in suite).
        #[arg(long)]
        queries: Option<usize>,
        #[arg(long, default_value_t = 64)]
        shortlist: usize,
        #[arg(long, default_value = "l1_rabitq")]
        strata: String,
    },
}

fn parse_strata(s: &str) -> Result<StrataDepth> {
    Ok(match s {
        "l0_only" | "L0_ONLY" => StrataDepth::L0Only,
        "l1_rabitq" | "L1_RABITQ" => StrataDepth::L1Rabitq,
        "full_64b" | "FULL_64B" => StrataDepth::Full64b,
        other => bail!("unknown strata depth: {other}"),
    })
}

fn parse_source(s: &str) -> u32 {
    match s.to_lowercase().as_str() {
        "slack" => source::SLACK,
        "email" | "mail" => source::EMAIL,
        "code" => source::CODE,
        _ => source::DOC,
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Init {
            dir,
            dims,
            strata,
            shortlist,
        } => {
            let mut cfg = EngineConfig::default();
            cfg.embed_dims = dims;
            cfg.strata_depth = parse_strata(&strata)?;
            cfg.shortlist_cand = shortlist;
            IndexEngine::create(&dir, cfg)?;
            println!("initialized mrl-index at {}", dir.display());
        }
        Cmd::Ingest {
            dir,
            text,
            file,
            source,
            timestamp,
            flush,
        } => {
            let mut engine = IndexEngine::open(&dir)?;
            let src = parse_source(&source);
            let mut n = 0usize;
            if let Some(t) = text {
                engine.ingest(t, timestamp, src)?;
                n += 1;
            }
            if let Some(path) = file {
                let body = std::fs::read_to_string(&path)
                    .with_context(|| format!("read {}", path.display()))?;
                for (i, line) in body.lines().enumerate() {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    let ts = if timestamp == 0 {
                        (i + 1) as u32
                    } else {
                        timestamp
                    };
                    engine.ingest(line.to_string(), ts, src)?;
                    n += 1;
                }
            }
            if flush {
                engine.flush()?;
            }
            println!("ingested {n} passages (total {})", engine.passage_count());
        }
        Cmd::Query {
            dir,
            query,
            time_lo,
            time_hi,
        } => {
            let engine = IndexEngine::open(&dir)?;
            let result = engine.query_filtered(&query, time_lo, time_hi)?;
            println!(
                "query ok — total {:.2} ms (embed {:.2} | scan {:.2} | fetch {:.2} | rerank {:.2})  blocks={} shortlist={}",
                result.stats.total_ms,
                result.stats.embed_ms,
                result.stats.scan_ms,
                result.stats.fetch_ms,
                result.stats.rerank_ms,
                result.stats.blocks_scanned,
                result.stats.shortlist
            );
            for (rank, hit) in result.hits.iter().enumerate() {
                let preview: String = hit.text.chars().take(96).collect();
                println!(
                    "{:>2}. id={}  exact={:.4}  scan={:.4}  {}",
                    rank + 1,
                    hit.passage_id,
                    hit.exact_score,
                    hit.scan_score,
                    preview
                );
            }
        }
        Cmd::Demo { dir, docs, dims } => {
            run_demo(dir, docs, dims)?;
        }
        Cmd::Bench { blocks, dims } => {
            run_bench(blocks, dims)?;
        }
        Cmd::BenchSuite {
            label,
            out,
            embeddings,
            synthetic,
        } => {
            let report = if synthetic {
                eprintln!(
                    "warning: --synthetic uses isotropic unit vectors; this is NOT an MRL evaluation"
                );
                let scan = mrl_index::benchmark::run_scan_recall_synthetic(
                    &label,
                    8_192,
                    32,
                    384,
                    64,
                    StrataDepth::L1Rabitq,
                    42,
                );
                let ingest = mrl_index::benchmark::run_ingest_bench(&label, 4_096, 384);
                mrl_index::benchmark::BenchSuiteReport {
                    label: label.clone(),
                    scan,
                    ingest,
                }
            } else {
                mrl_index::benchmark::run_suite_mrl(&label, &embeddings).with_context(|| {
                    format!(
                        "load MRL embeddings from {} — run: modal run modal/embed_sec10k_mrl.py",
                        embeddings.display()
                    )
                })?
            };
            report.print_summary();
            if let Some(path) = out {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&path, report.to_json_pretty())?;
                println!("wrote {}", path.display());
            }
        }
        Cmd::BenchRetrieval {
            label,
            out,
            data,
            queries,
            shortlist,
            strata,
        } => {
            let strata = parse_strata(&strata)?;
            let report = mrl_index::retrieval_bench::run_retrieval_bench(
                &label,
                &data,
                queries,
                shortlist,
                strata,
            )
            .with_context(|| {
                format!(
                    "load retrieval bench from {} — run: python3 scripts/parquet_to_bench_bin.py <parquet_suite> <bin_dir>",
                    data.display()
                )
            })?;
            report.print_summary();
            if let Some(path) = out {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&path, report.to_json_pretty())?;
                println!("wrote {}", path.display());
            }
        }
    }
    Ok(())
}

fn demo_corpus(n: usize) -> Vec<(String, u32, u32)> {
    let topics = [
        ("rust ownership borrow checker lifetimes", source::CODE),
        ("slack standup yesterday blockers", source::SLACK),
        ("email invoice payment q3 budget", source::EMAIL),
        ("vector search bitplane popcnt simd", source::DOC),
        ("matryoshka representation learning mrl", source::DOC),
        ("apple silicon mlx embedding gemma", source::DOC),
        ("cauchy schwarz pruning upper bound", source::DOC),
        ("passage index jsonl zero copy fetch", source::CODE),
        ("ring buffer append only flush block", source::CODE),
        ("rag latency budget personal workstation", source::DOC),
    ];
    (0..n)
        .map(|i| {
            let (topic, src) = topics[i % topics.len()];
            let text = format!(
                "doc-{i}: {topic} — note {i} discusses practical details for local retrieval systems and workstation RAG."
            );
            (text, (1_700_000_000 + i) as u32, src)
        })
        .collect()
}

fn run_demo(dir: PathBuf, docs: usize, dims: usize) -> Result<()> {
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    let mut cfg = EngineConfig::default();
    cfg.embed_dims = dims;
    cfg.strata_depth = StrataDepth::Full64b;
    cfg.slab_residency = SlabResidency::RamLocked;
    cfg.shortlist_cand = 64;
    cfg.final_top_k = 5;

    println!("mrl-index demo — {docs} docs, dims={dims}");
    let mut engine = IndexEngine::create(&dir, cfg)?;

    let corpus = demo_corpus(docs);
    let t_ing = Instant::now();
    engine.ingest_batch(&corpus)?;
    engine.flush()?;
    let ingest_ms = t_ing.elapsed().as_secs_f64() * 1000.0;

    println!(
        "ingested {} passages into {} PaxBlock64 macro-blocks in {:.1} ms ({:.0} docs/s)",
        engine.passage_count(),
        engine.block_count(),
        ingest_ms,
        docs as f64 / (ingest_ms / 1000.0)
    );

    let queries = [
        "bitplane transposed vector scan popcnt",
        "rust borrow checker ownership",
        "mlx embedding on apple silicon",
        "slack standup blockers",
    ];

    for q in queries {
        let result = engine.query(q)?;
        println!();
        println!("Q: {q}");
        println!(
            "   {:.2} ms total  [embed {:.2} | scan {:.2} | fetch {:.2} | rerank {:.2}]  shortlist={}",
            result.stats.total_ms,
            result.stats.embed_ms,
            result.stats.scan_ms,
            result.stats.fetch_ms,
            result.stats.rerank_ms,
            result.stats.shortlist
        );
        for (i, hit) in result.hits.iter().enumerate() {
            let preview: String = hit.text.chars().take(72).collect();
            println!(
                "   {:>2}. [{:.3}] {}",
                i + 1,
                hit.exact_score,
                preview
            );
        }
    }

    println!();
    println!("index path: {}", dir.display());
    Ok(())
}

fn run_bench(blocks: usize, dims: usize) -> Result<()> {
    use mrl_index::block::PaxBlock64;
    use mrl_index::config::EngineConfig;
    use mrl_index::quantize::encode_block;
    use mrl_index::scan::{scan_blocks, QueryPlan};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut data: Vec<PaxBlock64> = Vec::with_capacity(blocks);
    for _ in 0..blocks {
        let mut vecs: Vec<Vec<f32>> = Vec::with_capacity(64);
        for _ in 0..64 {
            let mut v = vec![0.0f32; dims];
            for x in &mut v {
                *x = rng.gen_range(-1.0..1.0);
            }
            // l2 normalize
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            for x in &mut v {
                *x /= n;
            }
            vecs.push(v);
        }
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let ts = vec![1u32; 64];
        let src = vec![source::DOC; 64];
        data.push(encode_block(&refs, &ts, &src, dims));
    }

    let mut q = vec![0.0f32; dims];
    for x in &mut q {
        *x = rng.gen_range(-1.0..1.0);
    }
    let n = q.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for x in &mut q {
        *x /= n;
    }

    let plan = QueryPlan::from_query(&q, dims);
    let mut cfg = EngineConfig::default();
    cfg.embed_dims = dims;
    cfg.strata_depth = StrataDepth::Full64b;
    cfg.shortlist_cand = 64;

    // Warmup
    let _ = scan_blocks(&data, &plan, &cfg, None, None);

    let t = Instant::now();
    let hits = scan_blocks(&data, &plan, &cfg, None, None);
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    let rows = blocks * 64;
    println!(
        "scanned {rows} vectors ({blocks} blocks) in {ms:.3} ms  → {:.1} Mvecs/s  shortlist_top={}",
        (rows as f64 / (ms / 1000.0)) / 1e6,
        hits.first().map(|h| h.score).unwrap_or(0.0)
    );
    Ok(())
}
