# RAG evaluation by retriever and chunking strategy

## Summary (every retriever x chunking pair)

| Retriever / Chunking | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | Evidence in ctx | Answer accuracy | Citation hit |
|---|---|---|---|---|---|---|---|---|
| hybrid_rerank / section | 0.577 | 0.769 | 0.885 | 0.885 | 0.740 | 0.885 | 0.885 | 0.885 |
| hybrid_rerank / sentence | 0.462 | 0.654 | 0.654 | 0.808 | 0.598 | 0.654 | 0.808 | 0.731 |

## [hybrid_rerank] Retrieval (answerable cases, n=26)

| Chunking | Evidence intact | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Precision@1 | Precision@3 | Precision@5 | Precision@10 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| section | 0.947 | 0.577 | 0.769 | 0.885 | 0.885 | 0.596 | 0.788 | 0.904 | 0.904 | 0.615 | 0.295 | 0.208 | 0.104 | 0.740 |
| sentence | 0.947 | 0.462 | 0.654 | 0.654 | 0.808 | 0.462 | 0.673 | 0.692 | 0.827 | 0.462 | 0.295 | 0.185 | 0.108 | 0.598 |

## [hybrid_rerank] Recall@3 by question type

| question_type | section | sentence |
|---|---|---|
| cross_doc | 0.667 | 1.000 |
| lookup | 1.000 | 1.000 |
| multi_passage | 0.583 | 0.333 |
| numeric | 0.857 | 0.679 |

## [hybrid_rerank] Generation (up to 5 chunks, budget-capped, passed to claude-haiku-5-5)

| Chunking | Answer accuracy | Citation hit | Evidence in ctx | Quote verified | Abstain accuracy | Parse failures | Judge failures | Avg context chars |
|---|---|---|---|---|---|---|---|---|
| section | 0.885 | 0.885 | 0.885 | 1.000 | 1.000 | 0.000 | 0.000 | 8360.400 |
| sentence | 0.808 | 0.731 | 0.654 | 1.000 | 1.000 | 0.000 | 0.000 | 10676.867 |

Notes: Hit@k = every evidence quote found in the top-k chunks; Recall@k = share of evidence quotes found; Precision@k = share of top-k chunks containing evidence; MRR = mean 1/rank of the first chunk containing evidence; Evidence intact = share of evidence quotes that sit whole inside a single chunk (the ceiling for retrieval); Evidence in ctx = share of questions whose evidence is still inside the context actually sent to the LLM. Answer accuracy and Citation hit count unscored/failed cases as wrong. Parse failures = answers whose JSON could not be parsed even after the tolerant parser and one retry; Judge failures = answers for which no verdict could be obtained. Retrievers: dense = bge-m3 embeddings; bm25 = keyword search; hybrid = dense + BM25 merged by Reciprocal Rank Fusion (k=60, 30 candidates each); *_rerank = top 20 candidates re-scored per filing by the cross-encoder BAAI/bge-reranker-v2-m3.