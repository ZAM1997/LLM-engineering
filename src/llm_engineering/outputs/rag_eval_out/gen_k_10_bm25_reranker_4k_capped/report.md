# RAG evaluation by retriever and chunking strategy

## Summary (every retriever x chunking pair)

| Retriever / Chunking | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | Evidence in ctx | Answer accuracy | Citation hit |
|---|---|---|---|---|---|---|---|---|
| dense / char | 0.231 | 0.231 | 0.269 | 0.462 | 0.275 | 0.385 | 0.577 | 0.423 |
| dense / section | 0.346 | 0.500 | 0.692 | 0.846 | 0.526 | 0.538 | 0.692 | 0.577 |
| dense / sentence | 0.385 | 0.500 | 0.577 | 0.654 | 0.479 | 0.500 | 0.615 | 0.538 |
| hybrid / char | 0.269 | 0.346 | 0.423 | 0.500 | 0.363 | 0.500 | 0.731 | 0.500 |
| hybrid / section | 0.385 | 0.654 | 0.808 | 0.885 | 0.597 | 0.654 | 0.808 | 0.654 |
| hybrid / sentence | 0.462 | 0.615 | 0.615 | 0.731 | 0.587 | 0.577 | 0.615 | 0.577 |
| hybrid_rerank / char | 0.269 | 0.385 | 0.462 | 0.500 | 0.403 | 0.500 | 0.692 | 0.538 |
| hybrid_rerank / section | 0.577 | 0.769 | 0.885 | 0.885 | 0.740 | 0.731 | 0.808 | 0.769 |
| hybrid_rerank / sentence | 0.462 | 0.654 | 0.654 | 0.808 | 0.598 | 0.615 | 0.769 | 0.654 |

## [dense] Retrieval (answerable cases, n=26)

| Chunking | Evidence intact | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Precision@1 | Precision@3 | Precision@5 | Precision@10 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| char | 0.868 | 0.231 | 0.231 | 0.269 | 0.462 | 0.231 | 0.231 | 0.288 | 0.481 | 0.231 | 0.077 | 0.062 | 0.054 | 0.275 |
| section | 0.947 | 0.346 | 0.500 | 0.692 | 0.846 | 0.365 | 0.538 | 0.712 | 0.846 | 0.385 | 0.192 | 0.162 | 0.096 | 0.526 |
| sentence | 0.947 | 0.385 | 0.500 | 0.577 | 0.654 | 0.385 | 0.519 | 0.596 | 0.673 | 0.385 | 0.192 | 0.131 | 0.077 | 0.479 |

## [dense] Recall@3 by question type

| question_type | char | section | sentence |
|---|---|---|---|
| cross_doc | 0.000 | 0.333 | 0.500 |
| lookup | 0.667 | 0.667 | 1.000 |
| multi_passage | 0.167 | 0.333 | 0.167 |
| numeric | 0.214 | 0.643 | 0.571 |

## [dense] Generation (up to 10 chunks, budget-capped, passed to claude-haiku-5-5)

| Chunking | Answer accuracy | Citation hit | Evidence in ctx | Quote verified | Abstain accuracy | Parse failures | Judge failures | Avg context chars |
|---|---|---|---|---|---|---|---|---|
| char | 0.577 | 0.423 | 0.385 | 0.972 | 1.000 | 0.000 | 0.000 | 4000.000 |
| section | 0.692 | 0.577 | 0.538 | 1.000 | 1.000 | 0.000 | 0.000 | 4000.000 |
| sentence | 0.615 | 0.538 | 0.500 | 1.000 | 1.000 | 0.000 | 0.000 | 4000.000 |

## [hybrid] Retrieval (answerable cases, n=26)

| Chunking | Evidence intact | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Precision@1 | Precision@3 | Precision@5 | Precision@10 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| char | 0.868 | 0.269 | 0.346 | 0.423 | 0.500 | 0.269 | 0.385 | 0.481 | 0.538 | 0.269 | 0.141 | 0.108 | 0.065 | 0.363 |
| section | 0.947 | 0.385 | 0.654 | 0.808 | 0.885 | 0.404 | 0.712 | 0.846 | 0.904 | 0.423 | 0.256 | 0.192 | 0.104 | 0.597 |
| sentence | 0.947 | 0.462 | 0.615 | 0.615 | 0.731 | 0.481 | 0.635 | 0.635 | 0.769 | 0.500 | 0.231 | 0.146 | 0.100 | 0.587 |

## [hybrid] Recall@3 by question type

| question_type | char | section | sentence |
|---|---|---|---|
| cross_doc | 0.167 | 0.333 | 0.500 |
| lookup | 0.667 | 1.000 | 1.000 |
| multi_passage | 0.250 | 0.750 | 0.333 |
| numeric | 0.429 | 0.714 | 0.714 |

## [hybrid] Generation (up to 10 chunks, budget-capped, passed to claude-haiku-5-5)

| Chunking | Answer accuracy | Citation hit | Evidence in ctx | Quote verified | Abstain accuracy | Parse failures | Judge failures | Avg context chars |
|---|---|---|---|---|---|---|---|---|
| char | 0.731 | 0.500 | 0.500 | 0.955 | 1.000 | 0.000 | 0.000 | 4000.000 |
| section | 0.808 | 0.654 | 0.654 | 1.000 | 1.000 | 0.000 | 0.000 | 4000.000 |
| sentence | 0.615 | 0.577 | 0.577 | 1.000 | 1.000 | 0.000 | 0.000 | 4000.000 |

## [hybrid_rerank] Retrieval (answerable cases, n=26)

| Chunking | Evidence intact | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Precision@1 | Precision@3 | Precision@5 | Precision@10 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| char | 0.868 | 0.269 | 0.385 | 0.462 | 0.500 | 0.288 | 0.423 | 0.519 | 0.538 | 0.308 | 0.167 | 0.123 | 0.065 | 0.403 |
| section | 0.947 | 0.577 | 0.769 | 0.885 | 0.885 | 0.596 | 0.788 | 0.904 | 0.904 | 0.615 | 0.295 | 0.208 | 0.104 | 0.740 |
| sentence | 0.947 | 0.462 | 0.654 | 0.654 | 0.808 | 0.462 | 0.673 | 0.692 | 0.827 | 0.462 | 0.295 | 0.185 | 0.108 | 0.598 |

## [hybrid_rerank] Recall@3 by question type

| question_type | char | section | sentence |
|---|---|---|---|
| cross_doc | 0.167 | 0.667 | 1.000 |
| lookup | 0.667 | 1.000 | 1.000 |
| multi_passage | 0.417 | 0.583 | 0.333 |
| numeric | 0.429 | 0.857 | 0.679 |

## [hybrid_rerank] Generation (up to 10 chunks, budget-capped, passed to claude-haiku-5-5)

| Chunking | Answer accuracy | Citation hit | Evidence in ctx | Quote verified | Abstain accuracy | Parse failures | Judge failures | Avg context chars |
|---|---|---|---|---|---|---|---|---|
| char | 0.692 | 0.538 | 0.500 | 1.000 | 1.000 | 1.000 | 0.000 | 4000.000 |
| section | 0.808 | 0.769 | 0.731 | 0.977 | 1.000 | 0.000 | 0.000 | 4000.000 |
| sentence | 0.769 | 0.654 | 0.615 | 1.000 | 1.000 | 0.000 | 0.000 | 4000.000 |

Notes: Hit@k = every evidence quote found in the top-k chunks; Recall@k = share of evidence quotes found; Precision@k = share of top-k chunks containing evidence; MRR = mean 1/rank of the first chunk containing evidence; Evidence intact = share of evidence quotes that sit whole inside a single chunk (the ceiling for retrieval); Evidence in ctx = share of questions whose evidence is still inside the (budget-capped, 4000 chars) context actually sent to the LLM. Answer accuracy and Citation hit count unscored/failed cases as wrong. Parse failures = answers whose JSON could not be parsed even after the tolerant parser and one retry; Judge failures = answers for which no verdict could be obtained. Retrievers: dense = bge-m3 embeddings; bm25 = keyword search; hybrid = dense + BM25 merged by Reciprocal Rank Fusion (k=60, 30 candidates each); *_rerank = top 20 candidates re-scored per filing by the cross-encoder BAAI/bge-reranker-v2-m3.