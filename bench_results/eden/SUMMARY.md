# EDEN-prod probe summary

**Branch:** `cursor/eden-prod-probe-4432`  
**Corpus:** MS MARCO EmbeddingGemma-2 @768  
**Date:** 2026-11-06 (America/Chicago)

## Verdict

**Ship candidate:** EDEN-prod on the **64-d prefix only** (`b=4`), keeping RaBitQ mid-band + folded tail.

**Do not ship yet:** EDEN mid-band `b=1` (smoke win, 1M regression).

## Phase results

| Phase | What | Result |
|---|---|---|
| 0 offline smoke | Prefix EDEN b=4 vs min–max | **Win** — recon MSE ↓, Pearson 0.995 vs 0.988, prefix R@10 0.872 vs 0.785 |
| 1 Pax smoke | Full scan+rerank | **Win** — R@10 0.917 vs 0.913, scan ~0.95× |
| 2 Pax 1M | Full scan+rerank (6980 q, sl=256) | **Win** — R@10 **0.8837 vs 0.8812** (+0.25pp), scan **0.90×** (serial A/B) / ~0.93× parallel dual-path timing |
| Follow-up bitsweep | Prefix b=2/3/4 | Only **b=4** beats min–max |
| Follow-up midband offline | Mid EDEN b=1 vs RaBitQ | Offline win (Pearson/R@10) |
| Follow-up dual smoke | Prefix+mid EDEN | **+2.7pp** R@10 (0.940 vs 0.913) |
| Follow-up dual 1M | Prefix+mid EDEN | **Loss** — R@10 **0.864 vs 0.881** (−1.7pp) |

## Artifacts

- `phase0_smoke.json`
- `phase1_smoke.json`
- `phase2_1m.json`
- `followup_bitsweep_smoke.json`
- `followup_midband_smoke.json`
- `followup_dual_smoke.json`
- `followup_dual_1m.json`

## How to reproduce

```bash
cargo run --release --example eden_phase0 -- data/msmarco_gemma2/smoke_bin
cargo run --release --example eden_phase1 -- data/msmarco_gemma2/smoke_bin 256
cargo run --release --example eden_phase1 -- data/msmarco_gemma2/1m_bin 256
EDEN_DUAL_SKIP_PREFIX=1 cargo run --release --example eden_followup_dual -- data/msmarco_gemma2/1m_bin 256
```

## Next engineering (if adopting prefix EDEN)

1. Persist `EdenFit` mean + seed in index manifest.
2. Default `EngineConfig.prefix_codec = EdenProd` behind a flag; keep MinMax4 for bit-identical replay.
3. Revisit mid-band EDEN only with a layout that does not steal `meta.offset`, plus 1M-first validation (smoke lied).
