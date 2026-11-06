# mrl-index — overview (for collaborators)

Local-first **bitplane** vector search for Matryoshka (MRL) embeddings.

## Idea

MRL models put most of the useful signal in the first dimensions. We store a compact sketch (~64 bytes/vector), spend more bits on the head of the vector and fewer on the noisy tail, then rerank only a short candidate list with exact embeddings.

## Layout (`PaxBlock64`)

Each 4 KiB page holds 64 vectors:

| Band | Dims | Coding (default) |
|---|---|---|
| Prefix | 0..64 | **EDEN-prod** 4-bit (center → normalize → structured Hadamard → Lloyd–Max, unbiased IP scale) |
| Mid | 64..192 | FWHT + 1-bit RaBitQ |
| Tail | 192..768 | Folded FWHT → 1-bit sketch |

At query time we score the head first. If a document already cannot finish in the shortlist—even if the remaining bits were as good as possible—we stop and move on without reading the rest of its code.

## What we measured (MS MARCO + EmbeddingGemma-2 @768)

| Setup | ANN R@10 (sl=256) | Scan |
|---|---|---|
| Min–max 4-bit prefix (legacy) | 0.8812 | baseline |
| **EDEN-prod prefix b=4** (default) | **0.8837** | ~0.90–0.95× latency |
| EDEN prefix + EDEN mid b=1 | 0.864 on 1M | *regressed — not shipped* |

Details: `bench_results/eden/SUMMARY.md`.

## Try it

```bash
cargo build --release
./target/release/mrl-index demo --docs 2048

# Clean retrieval bench (needs converted suite under data/msmarco_gemma2/*_bin)
./target/release/mrl-index bench-retrieval \
  --label eden-1m --data data/msmarco_gemma2/1m_bin \
  --shortlist 256 --out reports/bench_eden_1m.json
```

Index dirs now write `eden_prefix.json` (fitted mean + seed) next to `manifest.json` so encode and query stay aligned.

## Contact / sharing

Designed for personal workstations and research collab. Default path is the EDEN prefix stack above; mid-band EDEN remains experimental (`MidCodec::EdenB1`) and is off by default after the 1M regression.
