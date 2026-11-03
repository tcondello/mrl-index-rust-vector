# Algorithmic & System Primitives — Composition Design

**Status:** Steps C + B + B′ + `QueryPipeline` + `Encoder` on this branch  
**Branch:** `cursor/scan-stage-primitives-4432`  
**Freeze line:** MS MARCO / EmbeddingGemma-2 @768 quality (ANN R@10 at fixed `w_s4` + shortlist) must not regress when refactoring; CS-prune speedups must remain available as composable stages. Production path is **`Full64b`** (historical benches labeled `l1_rabitq` already scored S4; `L1Rabitq` is now the true S1-without-S4 ablation).

---

## 1. Problem

The math is mostly already freestanding (`fwht`, `fold_tail`, `encode_block`, `dot_768`, `Shortlist`, `Embedder`). What hurts optimize / extend / maintain is **fusion of stages**:

- `scan_block` owns prefix → CS → S1 → CS → S4 → push as one function.
- `IndexEngine::query_*` owns embed → plan → scan → fetch → rerank as one function.
- `retrieval_bench` reimplements orchestration instead of sharing a pipeline.
- Geometry (64 / 128 / 192 / 768 / `B=9`) and scan knobs are scattered across modules.

Every optimization (CS prune, `w_s4`, fp16 rerank, token fetch) becomes a mid-function edit plus a full-stack re-bench.

---

## 2. Goals

1. **Optimize** one stage in isolation (time it, A/B it, Pareto it) without forking the engine.
2. **Extend** by adding or swapping a stage / trait impl (new fold, Metal epilogue, cross-encoder rerank).
3. **Maintain** a small contract surface: geometry + scan config + stage graph.
4. **Share** one query orchestration between live engine and MS MARCO benches.

Non-goals for the first cut: new ANN algorithm, cross-encoder, rewriting Gemma/Metal, changing `PaxBlock64` on-disk layout.

---

## 3. Two primitive layers

### 3.1 Algorithmic (pure, no I/O)

| Family | Primitive | Today |
|--------|-----------|--------|
| Transform | `Fwht`, `RotateMid`, `FoldTail` | `rotate.rs` |
| Quantize | `Affine4BitPrefix`, `SignBitplane`, `PackRho` | `quantize.rs` |
| Plan | `BuildQueryPlan` | `QueryPlan::from_query` |
| Score | `PrefixIp` (SWAR), `RabitqIp`, `TailFoldIp` | inside `scan_block` |
| Bound / prune | `CsResidual`, `CsTail`, `S4AbsBudget` | CS-prune branch / local |
| Exact | `DotF32` / `DotF16` / `DotI8` | `embed.rs`, `lowprec.rs` |
| Offline fusion | `AugmentProjWeights`, `FoldProjWeights` | `rotate.rs` |

**Scan stage contract** (block-local):

```text
inputs:  block, plan, ScanConfig, tau, active mask
state:   scores[64], survivors: u64, mid_n/high_n
output:  updated scores + survivors  (or early empty)
```

Score stages **add** to `scores` for bits in `survivors`.  
Prune stages **narrow** `survivors` using upper bounds vs `tau`.

### 3.2 System (I/O, batching, policy)

| Family | Primitive | Today |
|--------|-----------|--------|
| Policy | `SketchGeometry`, `ScanConfig` | scattered consts + `EngineConfig` fields |
| Storage | `PassageStore`, `SegmentSlab`, `IdMap` | `passage.rs`, `segment.rs`, `engine` |
| Ingest | `RingBuffer`, `Encoder` trait | `ring.rs` → `encode_block` |
| Runtime | `Shortlist`, `ParallelBlockScan`, `TemporalFilter` | `scan.rs` |
| Neural | `Embedder`, `UnifiedEmbedder` | `rerank.rs`, `gemma_embed.rs` |
| Harness | `QueryPipeline`, `RetrievalBenchHarness` | split across `engine` / `retrieval_bench` |

`ScanConfig` (first-class subset of engine knobs):

- `shortlist_cand`, `cs_tolerance`, `stratum4_weight`
- `strata_depth` (must actually gate S1/S4)
- temporal bounds / `temporal_filter_enable`
- `numa_sharding`

`SketchGeometry` (768 path default):

- `embed_dims`, `mrl_prefix=64`, `rabitq_dims=128`, `tail_start=192`
- `tail_fold_dims=64`, `tail_fold_blocks=9`
- default `stratum4_weight` policy (`0.20` locked for Gemma@768)

---

## 4. Composition model

```text
QueryPipeline =
    EmbedQuery
  → BuildQueryPlan(geometry)
  → ParallelBlockScan( ScanKernel )
  → FetchCandidates          // texts and/or token spans
  → Rerank                   // exact IP | packed Gemma | future cross-encoder

ScanKernel = ordered list of ScanStage
  default Gemma@768:
    TemporalFilter?
    → PrefixIp + CsSplitResidual
    → RabitqIp
    → CsTail
    → S4AbsBudget            // optional; proven ~16–18% scan win
    → TailFoldIp
    → PushShortlist
```

**Motivating example — CS prune as composition, not a mid-function patch:**

| Stage | Role |
|-------|------|
| `CsSplitResidual` | After prefix: `ub = prefix + α(‖q_mid‖‖x_mid‖ + ‖q_tail‖‖x_tail‖)` |
| `S4AbsBudget` | Before S4 bits: skip if `score_s1 + α·‖x_tail‖·w_s4·Σ\|pos\| < τ` |

Local validation (1.04M Wikipedia): scan **18.10 → 15.17 ms (−16.2%)**, identical top-5 / exact scores. Cloud MS MARCO 8.8M: ~260 → ~212 ms/q at same ANN R@10.

---

## 5. Rollout

| Step | Deliverable | Behavior |
|------|-------------|----------|
| **C** | This design doc | — |
| **B** | Split `scan_block` into named stages + default kernel; port CS-prune stages | Bit-identical to CS-prune path (or main + prune) |
| **B′** | `ScanConfig` / `SketchGeometry`; `StrataDepth` gates S1/S4 (`Full64b` default) | No quality change vs historical S4-on path |
| **Pipeline** | `QueryPipeline` shared by engine + `retrieval_bench` (plan → scan → exact-IP) | Same metrics |
| **Encoder** | `Encoder` / `FusedPackedEncoder` for ring, mmap benches, fused Gemma ingest | Bit-identical blocks |

Each step keeps a freeze line: fixed fixtures (MS MARCO bin and/or live Wikipedia smoke), compare ANN R@10 / top-k identity and scan ms.

---

## 6. Testing strategy

1. **Unit:** each score/prune stage on a synthetic `PaxBlock64` (survivor masks, monotonic scores).
2. **Parity:** staged kernel vs previous monolithic kernel on a pinned query set (same shortlist IDs + scores).
3. **Regression:** `cargo test`; optional `bench-retrieval --queries N` on `1m_bin`.
4. **Perf:** `--scan-stats` keep-rates (`keep_s4bud ≈ 10%` expected with prune enabled).

---

## 7. Explicit non-goals / freeze

- Do not change `PaxBlock64` byte layout in the primitives PR.
- Do not change default `w_s4=0.20` without a new Pareto + lock note.
- Do not merge Metal/Gemma fused paths into the scan kernel; they remain `UnifiedEmbedder` / encode-side fusion.
- Do not weaken quality gates to “pass” a refactor.

---

## 8. Open decisions (defaults proposed)

| Question | Proposal |
|----------|----------|
| CS prune on by default? | **Yes** — validated locally and on MS MARCO |
| Stages as free fns or trait objects? | **Free fns + `ScanKernel` fn table** first; traits only if we need dyn plugins |
| Where does geometry live? | New `src/geometry.rs` (follow-up); consts stay in `block.rs` until then |
| Bench shares pipeline when? | After scan split is green |

---

## 9. Implementation note for Step B

Target shape in `src/scan.rs` (names indicative):

```rust
struct BlockScanCtx<'a> { block, plan, cfg, tau, alpha, ... }
struct BlockScanState { scores, survivors, mid_n, high_n, active }

fn stage_prefix_ip_and_cs_split(ctx, state);
fn stage_rabitq_ip(ctx, state);
fn stage_cs_tail(ctx, state);
fn stage_s4_abs_budget(ctx, state);   // no-op if budget==0 / no tail
fn stage_tail_fold_ip(ctx, state);
fn finish_push(...);

pub fn scan_block(...) {
  // temporal + active setup
  stage_prefix_ip_and_cs_split(...);
  if state.survivors == 0 { return; }
  stage_rabitq_ip(...);
  stage_cs_tail(...);
  if state.survivors == 0 { return; }
  stage_s4_abs_budget(...);
  if state.survivors == 0 { return; }
  stage_tail_fold_ip(...);
  finish_push(...);
}
```

Public API (`scan_block`, `scan_blocks`, `QueryPlan`, `Shortlist`) stays stable so engine / benches need no call-site churn in Step B.
