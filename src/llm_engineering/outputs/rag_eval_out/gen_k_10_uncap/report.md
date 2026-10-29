# RAG evaluation by chunking strategy

## Retrieval (answerable cases, n=26)

| Chunking | Evidence intact | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Precision@1 | Precision@3 | Precision@5 | Precision@10 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| char | 0.868 | 0.231 | 0.231 | 0.269 | 0.462 | 0.231 | 0.231 | 0.288 | 0.481 | 0.231 | 0.077 | 0.062 | 0.054 | 0.275 |
| section | 0.947 | 0.346 | 0.500 | 0.692 | 0.846 | 0.365 | 0.538 | 0.712 | 0.846 | 0.385 | 0.192 | 0.162 | 0.096 | 0.526 |
| sentence | 0.947 | 0.385 | 0.500 | 0.577 | 0.654 | 0.385 | 0.519 | 0.596 | 0.673 | 0.385 | 0.192 | 0.131 | 0.077 | 0.479 |

## Recall@3 by question type

| question_type | char | section | sentence |
|---|---|---|---|
| cross_doc | 0.000 | 0.333 | 0.500 |
| lookup | 0.667 | 0.667 | 1.000 |
| multi_passage | 0.167 | 0.333 | 0.167 |
| numeric | 0.214 | 0.643 | 0.571 |

## Generation (up to 10 chunks, budget-capped, passed to claude-haiku-5-5)

| Chunking | Answer accuracy | Citation hit | Evidence in ctx | Quote verified | Abstain accuracy | Parse failures | Judge failures | Avg context chars |
|---|---|---|---|---|---|---|---|---|
| char | 0.654 | 0.500 | 0.462 | 1.000 | 1.000 | 0.000 | 0.000 | 5073.600 |
| section | 0.923 | 0.846 | 0.846 | 1.000 | 1.000 | 0.000 | 0.000 | 16005.733 |
| sentence | 0.769 | 0.692 | 0.654 | 1.000 | 1.000 | 0.000 | 0.000 | 15432.600 |

Notes: Hit@k = every evidence quote found in the top-k chunks; Recall@k = share of evidence quotes found; Precision@k = share of top-k chunks containing evidence; MRR = mean 1/rank of the first chunk containing evidence; Evidence intact = share of evidence quotes that sit whole inside a single chunk (the ceiling for retrieval); Evidence in ctx = share of questions whose evidence is still inside the (budget-capped, 0 chars) context actually sent to the LLM. Answer accuracy and Citation hit count unscored/failed cases as wrong. Parse failures = answers whose JSON could not be parsed even after the tolerant parser and one retry.