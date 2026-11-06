# RAG Evaluation Framework — Financial Document Q&A

An evaluation harness for a Retrieval-Augmented Generation (RAG) pipeline that answers
questions over SEC filings (Regions Financial Corp. 10-Q and 8-K). The project compares
**chunking strategies**, **retrieval methods**, and **context budgets**, with separate
metrics for retrieval quality and generation quality, so improvements can be attributed
to the right stage of the pipeline.

> This is a personal/learning project focused on *how to evaluate a RAG system
> rigorously*, not just build one. Most of the engineering effort went into the harness
> itself: catching silent failures, removing scoring bugs, and designing metrics that
> separate "did retrieval find the evidence" from "did the LLM use it correctly."

---

## Table of contents

- [Why this project](#why-this-project)
- [Pipeline overview](#pipeline-overview)
- [Chunking strategies](#chunking-strategies)
- [Retrieval methods](#retrieval-methods)
- [Evaluation metrics](#evaluation-metrics)
- [Results](#results)
- [Debugging journey: bugs found and fixed](#debugging-journey-bugs-found-and-fixed)
- [Repo structure](#repo-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Limitations](#limitations)
- [Next steps](#next-steps)

---

## Why this project

Most RAG tutorials skip evaluation and go straight to "it works." This project asks a
narrower question first: **for a corpus of dense, tabular financial filings, which
chunking and retrieval design actually gets the right passage in front of the model —
and how would you know?**

SEC filings are a deliberately hard test case:
- Heavy use of markdown tables with repeated header text ("Quarter Ended 6/30/2026")
  across many rows.
- Exact figures matter (`$583 million`, `3.66%`) — a close paraphrase is wrong.
- Some questions need evidence from **two different filings** (the 10-Q and the 8-K).
- Some questions have **no answer** in the corpus, and the model needs to say so.

## Pipeline overview

```
Question
   │
   ├─► Dense retrieval (bge-m3 embeddings via ChromaDB)
   ├─► BM25 keyword retrieval (in-memory, numeric-aware tokenizer)
   │
   └─► Reciprocal Rank Fusion (hybrid) ──► Cross-encoder reranker (optional)
                                                     │
                                          Top-k chunks, packed into a
                                          fixed character budget
                                                     │
                                          Claude (claude-haiku-5-5) answers
                                          with citations + verbatim quotes
                                                     │
                                          LLM-as-judge grades the answer
                                          against a reference answer
                                                     │
                                       Retrieval metrics + generation metrics
```

Every stage is scored independently: **retrieval metrics** check whether the right text
was found, and **generation metrics** check whether the model used it faithfully and
correctly, so a weak final score can be traced back to the stage that caused it.

## Chunking strategies

| Strategy | How it splits the document | Strength | Weakness |
|---|---|---|---|
| `char` | Fixed 512-character windows, 60-char overlap | Simple baseline | Cuts mid-sentence, mid-table, mid-number |
| `sentence` | 5-sentence windows, 1-sentence overlap | Keeps prose coherent | Still splits tables across chunks |
| `section` | Splits on real document structure (`#`/`##`/`###` headings, bold section titles, and `==== START OF <file> ====` filing separators); tiny sections are merged forward, oversized sections are split on line boundaries | Keeps a table together with its heading and surrounding explanation | Larger chunks cost more context budget |

`section` chunking required a custom parser (`chunk_section.py`) because the filings use
markdown headings and bold section titles, not the `## ` pattern a naive splitter
assumes — the first version of this chunker collapsed each entire filing into a single
chunk (see [Debugging journey](#debugging-journey-bugs-found-and-fixed)).

## Retrieval methods

| Method | What it does |
|---|---|
| `dense` | Embedding similarity search only (bge-m3) |
| `bm25` | Keyword search only (in-memory Okapi BM25, numbers kept whole: `3.66`, `1,277`) |
| `hybrid` | `dense` + `bm25` candidate lists merged by **Reciprocal Rank Fusion** (RRF, k=60) — combines rankings without needing comparable scores |
| `dense_rerank` | `dense` candidates re-scored by a cross-encoder |
| `hybrid_rerank` | `hybrid` candidates re-scored by a cross-encoder (`BAAI/bge-reranker-v2-m3`) — reads the question and chunk together for a precise relevance score |

For questions spanning both filings, each filing is retrieved and ranked separately,
then interleaved by rank, so one filing can't crowd out the other.

## Evaluation metrics

**Retrieval (did we find the right text?)**

| Metric | Meaning |
|---|---|
| `Evidence intact` | Share of gold evidence quotes that sit whole inside a single chunk — the ceiling for retrieval, measured before any search happens |
| `Hit@k` | 1 if *every* gold evidence quote is found in the top-k chunks, else 0 (strict) |
| `Recall@k` | Share of gold evidence quotes found in the top-k chunks (partial credit) |
| `Precision@k` | Share of the top-k chunks that contain gold evidence |
| `MRR` | Mean of `1 / rank` of the first chunk containing evidence (rewards ranking high) |

**Generation (did the model answer correctly and honestly?)**

| Metric | Meaning |
|---|---|
| `Evidence in ctx` | Share of questions whose gold evidence survived the context budget and actually reached the LLM |
| `Answer accuracy` | Share of answers graded correct by an LLM-as-judge against a reference answer |
| `Citation hit` | Share of answers where the chunk the model *cited* actually contains the gold evidence |
| `Quote verified` | Share of the model's quoted text that is verbatim in the chunks it cited (checks for fabrication) |
| `Abstain accuracy` | On unanswerable questions, share correctly answered "not found" |
| `Parse / Judge failures` | Answer or judge responses that could not be parsed as valid JSON, even after a tolerant parser and one retry |

## Results

Test set: 26 answerable + 4 unanswerable questions over the Regions Financial 10-Q and
8-K (lookup, numeric, multi-passage, and cross-document question types).

**Effect of chunking (dense retrieval, 4,000-char context budget):**

| Chunking | Hit@3 | Recall@3 | MRR | Answer accuracy |
|---|---|---|---|---|
| char | 0.231 | 0.231 | 0.275 | 0.577 |
| sentence | 0.500 | 0.519 | 0.479 | 0.615 |
| **section** | **0.500** | **0.538** | **0.526** | **0.692** |

**Effect of retrieval method (section chunking, 4,000-char budget):**

| Retriever | Hit@3 | MRR | Answer accuracy |
|---|---|---|---|
| dense | 0.500 | 0.526 | 0.692 |
| hybrid (BM25 + dense, RRF) | 0.654 | 0.597 | 0.808 |
| **hybrid + rerank** | **0.769** | **0.740** | **0.808** |

**Effect of context budget (section chunking, hybrid + rerank):**

| Chunks sent | Avg. context | Answer accuracy | Wrong answers (confident) |
|---|---|---|---|
| 3 (4,000-char cap) | ~4.0k chars | 0.808 | 0 |
| 5 | ~8.3k chars | 0.885 | 2 |
| 7 | ~11.9k chars | **0.923** | 1 |
| 10 (uncapped) | ~16.9k chars | **0.962** | 0 |

**Headline result:** structure-aware (`section`) chunking + hybrid retrieval + reranking
raised **Hit@3 from 0.50 → 0.77** and **answer accuracy from 69% → 92%** at a fixed
context budget, compared with the naive character-chunking + dense-only baseline.
When the chunking bug was still present (see below), the equivalent retrieval numbers
were a misleading ~0.88 — the bug needed to be found before any real comparison was
possible.

Across every configuration, `section` chunking produced the safest failure mode:
wrong/confident answers were rare to nonexistent, and the model correctly abstained
("not found in the provided context") on all unanswerable questions in every run.

## Debugging journey: bugs found and fixed

The evaluation harness itself surfaced four pipeline bugs that would otherwise have
produced misleading comparisons:

1. **Silent chunking failure.** The first `section` chunker split on `\n## `, a pattern
   the filings never use (they use `#`, `###`, and bold section titles). Every filing
   collapsed into a single chunk. The collection "health check" reported this as `OK`
   because it only checks that a collection is non-empty and queryable — not that the
   chunking is sensible. Fixed by writing a filing-aware chunker that splits on real
   heading patterns and filing boundaries (`chunk_section.py`).
2. **False-negative evidence matching.** Gold evidence quotes failed to match retrieved
   chunks that visibly contained them, because the raw filing text has markdown escapes
   (`0\.42`, `\(1\)`) and smart-quote/dash variants the exact-substring matcher didn't
   normalize. Fixed with a `norm()` function (Unicode normalization, markdown
   un-escaping, footnote-marker stripping) applied consistently to evidence matching,
   quote verification, and the "evidence intact" ceiling.
3. **Stale generation cache.** Re-chunking the corpus didn't invalidate previously
   cached LLM answers, because the cache key didn't include a signature of the actual
   context sent. Fixed by hashing the packed context into the cache key and bumping a
   `PIPELINE_VERSION` string whenever the prompt or parsing logic changes.
4. **Unequal context budgets across strategies.** Early comparisons let each strategy
   send up to `k` chunks regardless of chunk size, so `char` (small chunks) received far
   less total context than `section` (large chunks) at the same `k`, confounding the
   comparison. Fixed with `pack_context()`, which fills chunks in rank order up to a
   fixed **character budget**, shared across all strategies, truncating only the
   lowest-ranked included chunk.

Smaller fixes along the way: joining all text blocks from the LLM response (the first
version silently dropped multi-block responses), a tolerant JSON parser that recovers
from stray backslash escapes, a judge prompt that puts the verdict first so a truncated
response still yields a usable grade, and not caching failed generations so they retry
automatically on the next run.

## Repo structure

```
.
├── evaluate_rag.py              # main evaluation harness (CLI + notebook entry point)
├── chunk_section.py             # filing-aware structural chunker
├── test/
│   └── rag_test_cases.jsonl     # 30 test cases: question, reference answer, evidence,
│                                 #   question_type, answerable flag
├── db/
│   └── chroma/                  # ChromaDB persistent store (char/section/sentence
│                                 #   collections per filing)
└── outputs/
    └── rag_eval_out/
        ├── report.md             # generated comparison tables (markdown)
        ├── results.jsonl         # per-case, per-strategy, per-retriever raw results
        ├── metrics_retrieval_<retriever>.csv
        ├── metrics_generation_<retriever>.csv
        ├── metrics_summary.csv   # one row per (retriever, chunking) pair
        └── llm_cache.json        # cached LLM calls, keyed by content hash
```

## Setup

```bash
pip install chromadb sentence-transformers torch pandas anthropic python-dotenv
```

Create a `.env` file with your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

The embedding model (`BAAI/bge-m3`) and reranker (`BAAI/bge-reranker-v2-m3`) download
automatically on first use (~2.5 GB combined).

## Usage

```python
# Free: retrieval metrics only, no LLM calls — run this first
main(["--retrieval-only"])

# Full comparison: all retrievers x all chunking strategies
main()

# Just the strongest configuration, at a specific chunk budget
main(["--retrievers", "hybrid_rerank", "--strategies", "section",
      "--gen-k", "7", "--ctx-budget", "0"])

# Smoke test on the first 5 cases
main(["--limit", "5"])
```

If a ChromaDB collection can't be queried (e.g. after moving the project directory):

```python
main(["--diagnose"])                               # inspect the store on disk
main(["--check-only"])                              # list broken/empty collections
main(["--rebuild-to", "db/chroma_rebuilt"])         # re-embed into a fresh store
```

Key CLI flags: `--retrievers` (`dense`, `bm25`, `hybrid`, `dense_rerank`,
`hybrid_rerank`), `--strategies` (`char`, `section`, `sentence`), `--gen-k` (max chunks
sent to the LLM), `--ctx-budget` (max total characters sent, `0` = uncapped), `--k`
(retrieval cutoffs to report).

## Limitations

- **Small test set.** 26 answerable + 4 unanswerable questions means one question is
  worth ~3.8 points of accuracy; differences under ~2 questions should be treated as
  noise, not a reliable signal.
- **Tuned on the test set.** Chunk size, retriever, and context budget were all selected
  by looking at results on these same 26 questions, so the final accuracy is likely
  optimistic and has not been confirmed on a held-out set.
- **A few gold-evidence labels are probably incomplete.** 2–3 questions are answered
  correctly by multiple configurations without their listed gold evidence ever being
  retrieved, suggesting the reference evidence quote doesn't cover every valid source
  passage for that question.
- **Single company, single quarter.** The corpus is one Regions Financial 10-Q and one
  8-K; generalization to other filers or filing types is untested.
- **Cross-document questions remain the weakest category** (Recall@3 ≈ 0.33–1.0 across
  only 3 questions — too few to draw a firm conclusion).

## Next steps

- Build a held-out test set (40+ new questions, more multi-passage/numeric/cross-doc
      cases) and report the final configuration's accuracy on it, untouched during tuning.
- Audit and correct the questions with suspected incomplete gold evidence.
- Try parent-child retrieval (small chunks for matching, parent section for context)
      as an alternative to tuning section chunk size directly.
- Add a prompt rule for multi-source questions ("answer each part separately, state
      explicitly what is missing") to reduce completeness errors on cross-doc questions.
- Test an adaptive-depth strategy: answer with a small chunk budget by default,
      escalate to a larger one only if the model abstains.
- Repeat the final configuration multiple times to quantify run-to-run LLM-judge
      variance before reporting a single number.

---

*Built as a personal learning project to practice rigorous RAG evaluation, not just RAG
construction. Feedback and issues welcome.*