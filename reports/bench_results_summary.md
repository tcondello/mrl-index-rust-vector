# Clean MS MARCO + EmbeddingGemma-2 retrieval results

Config: dims=768, strata=l1_rabitq, shortlist=64, no synthetic / no self-retrieval

| Scale | n | queries | shortlist_cov@10 | reranked R@10 | reranked R@100 | qrels R@10 | qrels nDCG@10 | scan ms/q | Mvecs/s |
|-------|---|---------|------------------|---------------|----------------|------------|---------------|-----------|---------|
| smoke | 8,192 | 128 | 0.6250 | 0.6250 | 0.2434 | 0.9727 | 0.9461 | 1.584 | 5.17 |
| 1m | 1,000,000 | 6980 | 0.5853 | 0.5853 | 0.2067 | 0.7161 | 0.5747 | 183.828 | 5.44 |
