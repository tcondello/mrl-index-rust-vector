# Step 3 results — tail fold into stratum4

Folded dims `192..768` into `stratum4_mid` (block Rademacher + FWHT 1-bit), packed mid/tail ρ into `VectorMeta::flags`, scaled RaBitQ by ‖x_mid‖, unrolled `dot_768`.

## vs Step 2.5 (pre–tail-fold)

| Config | ANN R@10 before → after | qrels R@10 before → after | nDCG@10 |
|--------|-------------------------|---------------------------|---------|
| 1M / sl256 | 0.753 → **0.857** | 0.780 → **0.807** | 0.612 → **0.629** |
| 1M / sl1024 | 0.873 → **0.939** | 0.806 → **0.821** | 0.629 → **0.637** |
| 8.8M / sl1024 | 0.887 → **0.952** | 0.522 → **0.538** | 0.330 → **0.336** |

## Success metric

Target: ANN R@10 ≥ 0.887 at shortlist ≤ 256 on full MS MARCO, **or** ≥ 0.95 at shortlist 1024.

| Full 8.8M | ANN R@10 | Verdict |
|-----------|----------|---------|
| sl256 Step3 | **0.8809** | near-miss (was ~0.60 @sl64 / ~0.75@sl256 without fold) |
| sl1024 Step3 | **0.9519** | **PASS (≥0.95)** |

Embedding oracle qrels R@10 on full = 0.547.
