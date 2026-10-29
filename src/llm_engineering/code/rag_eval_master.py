"""
evaluate_rag.py - compare chunking strategies (char / section / sentence) for the Regions 10-Q + 8-K RAG.

For every test case and every chunking strategy it:
  1. embeds the question (same embedding model as the chunks),
  2. retrieves the top-k chunks from the ChromaDB collections,
  3. (optional) asks Claude to answer from those chunks (packed into a fixed character budget), with citations +
     verbatim quotes, then grades the answer against the reference answer,
  4. computes Hit@k, Recall@k, Precision@k and MRR per strategy and prints comparison tables.

Usage
    python evaluate_rag.py --retrieval-only        # free: retrieval metrics only, no LLM calls
    python evaluate_rag.py                         # full run: retrieval + answers + citations + judge
    python evaluate_rag.py --limit 5               # smoke test on the first 5 cases
    python evaluate_rag.py --gen-k 10 --ctx-budget 4000   # equal-context comparison: every strategy gets <= 4000 chars

In a notebook cell call main() instead, e.g. main(), main(["--retrieval-only"]) or main(["--limit", "5"]).
If a collection cannot be queried: main(["--diagnose"]) inspects the files, main(["--check-only"]) lists the broken ones, main(["--rebuild-to", "<new_folder>"]) repairs them.
"""
import argparse
import hashlib
import json
import numbers
import re, unicodedata
import sys
import time
from pathlib import Path
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer


try:
    import llm_engineering
    BASE_DIR = Path(list(llm_engineering.__path__)[0])      # .../src/llm_engineering  (independent of the notebook's cwd)
except ImportError:
    BASE_DIR = Path.cwd()

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model = SentenceTransformer('BAAI/bge-m3')

CASES_PATH = BASE_DIR / "test" / "rag_test_cases.jsonl"     # <- your test-case file
CHROMA_DIR = BASE_DIR / "db" / "chroma"                      # <- the folder that contains chroma.sqlite3
COLLECTION_TEMPLATE = "{doc}_{strategy}"       # tenq_char, eightk_section, ... -> match your collection names
STRATEGIES = ["char", "section", "sentence"]
K_VALUES = [1, 3, 5]
GEN_K = 5                                      # max chunks passed to the LLM (raise it, e.g. 10, to fill the budget for small chunks)
CTX_BUDGET = 4000                              # max TOTAL chars of context sent to the LLM, same for every strategy (0/None = no limit)
MAX_CHARS_PER_CHUNK = 4000                     # safety cap on any single chunk; None = no limit
REPORT_K = 3                                   # k used for the "Recall@k by question type" table (independent of gen_k)
BALANCE_CROSS_DOC = True                       # for doc == "both", interleave results from the two filings by rank
LLM_MODEL = "claude-haiku-5-5"                 # answers + judge. No temperature is passed (deprecated for this model)
GEN_MAX_TOKENS = 1200                          # FIX: more room for the JSON answer (was 800)
PIPELINE_VERSION = "v4"                        # FIX: bump whenever prompt/context/parsing changes -> old cache entries are not reused
EMBED_MODEL = model
OUT_DIR = BASE_DIR / "outputs" / "rag_eval_out"


def _embed_texts(texts):
    """Embed a list of texts with the SAME model that embedded the chunks. Adjust the import to your project.
    If your embedding API distinguishes query vs document inputs (e.g. Voyage input_type), handle that here."""
    from llm_engineering.utils.chunk_embed import embed_texts      # <-- adjust this import path
    vecs = embed_texts(texts, EMBED_MODEL, device)
    return vecs.tolist() if hasattr(vecs, "tolist") else [list(v) for v in vecs]

def embed_query(text):
    return _embed_texts([text])[0]


_client = None
_multi_block_warned = False


def call_llm(prompt, max_tokens=GEN_MAX_TOKENS):
    """One Claude call with simple retry. Reads ANTHROPIC_API_KEY from the environment / .env.
    FIX: joins ALL text blocks (the old code kept only the first one, which could return a cut-off JSON),
    and warns when the response was stopped early."""
    global _client, _multi_block_warned
    from dotenv import load_dotenv
    from anthropic import Anthropic
    if _client is None:
        load_dotenv()
        _client = Anthropic()
    for attempt in range(3):
        try:
            msg = _client.messages.create(model=LLM_MODEL, max_tokens=max_tokens,
                                          messages=[{"role": "user", "content": prompt}])
            blocks = [b.text for b in msg.content if b.type == "text"]
            if len(blocks) > 1 and not _multi_block_warned:
                print(f"  [note] the API returned {len(blocks)} text blocks; all are joined now", flush=True)
                _multi_block_warned = True
            if msg.stop_reason == "max_tokens":
                print(f"  [warning] response hit max_tokens={max_tokens} and was cut off", flush=True)
            elif msg.stop_reason not in ("end_turn", "stop_sequence", None):
                print(f"  [warning] unusual stop_reason: {msg.stop_reason}", flush=True)
            return "".join(blocks)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


# ============================================================== text helpers

def norm(s):
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\\([().*_\-|$])", r"\1", s)     # un-escape \( \) \. \$ etc.
    s = re.sub(r"\(\d\)", " ", s)                # footnote markers (1)
    s = s.replace("’", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    s = re.sub(r"[|*#_]", " ", s)                # markdown table/bold syntax
    return re.sub(r"\s+", " ", s).strip().lower()


def clean_md(t):
    """FIX: remove markdown backslash escapes (0\\.42, \\(1\\), \\-) before the text goes into the prompt, so the model
    does not copy stray backslashes into its JSON (the cause of invalid-escape parse failures). norm() strips the same
    escapes, so evidence matching and quote verification are unaffected."""
    return re.sub(r"\\([().*_\-|$\[\]#+!])", r"\1", t)


def triage(case, doc_text, chunks, retrieved_ids, k=5):
    """Diagnostic helper (not called by main): classify why an evidence quote was missed.
    case: test case dict; doc_text: full document text; chunks: {chunk_id: text}; retrieved_ids: ranked ids."""
    out = []
    nd = norm(doc_text)
    for ev in case["evidence"]:
        ne = norm(ev)
        in_doc = ne in nd
        in_chunk = [cid for cid, ch in chunks.items() if ne in norm(ch)]
        in_topk = [c for c in retrieved_ids[:k] if c in in_chunk]
        if not in_doc:     t = "A: quote not found in doc (evidence/normalization bug)"
        elif not in_chunk: t = "C: quote split across chunks"
        elif not in_topk:  t = "B: retrieval miss"
        else:              t = "A: in top-k, old matcher too strict"
        out.append((ev[:60], t))
    return out


_BAD_ESCAPE = re.compile(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})')     # a backslash that is not a valid JSON escape


def parse_json(text):
    """FIX: tolerant parser. Strips code fences, extracts the {...} object, and if strict parsing fails retries
    with stray backslashes (e.g. '\\.' or '\\u.') escaped. strict=False also allows raw control characters."""
    if not text:
        return None
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S).strip()
    m = re.search(r"\{.*\}", t, flags=re.S)
    candidates = [t] + ([m.group(0)] if m and m.group(0) != t else [])
    for cand in candidates:
        for variant in (cand, _BAD_ESCAPE.sub(r"\\\\", cand)):
            try:
                return json.loads(variant, strict=False)
            except json.JSONDecodeError:
                continue
    return None

def parse_verdict(text):
    """Judge reply -> {"correct": bool, "reason": str} or None. Recovers the verdict even from a truncated reply."""
    d = parse_json(text)
    if d and isinstance(d.get("correct"), bool):
        return d
    m = re.search(r'"correct"\s*:\s*(true|false)', text or "")
    return ({"correct": m.group(1) == "true", "reason": "recovered from truncated judge output"}
            if m else None)

def load_cases(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def docs_of(case):
    return ["tenq", "eightk"] if case["doc"] == "both" else [case["doc"]]


# ============================================================== ChromaDB access
def _persistent(path):
    import chromadb
    return chromadb.PersistentClient(path=str(path))


def open_client(path):
    if not (Path(path) / "chroma.sqlite3").exists():
        sys.exit(f"No chroma.sqlite3 in {Path(path).resolve()}\n"
                 "Set CHROMA_DIR to the folder that contains chroma.sqlite3 "
                 "(opening a wrong path would silently create an empty database).")
    return _persistent(path)


def collection_names(client):
    return {c if isinstance(c, str) else c.name for c in client.list_collections()}


def load_collections(client, docs, strategies):
    names = collection_names(client)
    cols = {}
    for d in docs:
        for s in strategies:
            name = COLLECTION_TEMPLATE.format(doc=d, strategy=s)
            if name not in names:
                sys.exit(f"Collection '{name}' not found.\nAvailable: {sorted(names)}\n"
                         f"Edit COLLECTION_TEMPLATE / STRATEGIES at the top of this script.")
            cols[(d, s)] = client.get_collection(name)
    return cols


def load_corpus(col, batch=500):
    texts, off = [], 0
    while True:
        res = col.get(include=["documents"], limit=batch, offset=off)
        if not res["ids"]:
            break
        texts += res["documents"]
        off += len(res["ids"])
    return texts


def check_collections(cols, probe_emb):
    """A collection is healthy only if it has chunks AND a vector query on it works."""
    rows, ok = [], True
    for (d, s), col in sorted(cols.items()):
        name = COLLECTION_TEMPLATE.format(doc=d, strategy=s)
        n = col.count()
        if n == 0:
            status = "EMPTY - no chunks were ever added"
        else:
            try:
                col.query(query_embeddings=[probe_emb], n_results=1, include=["distances"])
                status = "OK"
            except Exception as e:                       # e.g. "hnsw segment reader: Nothing found on disk"
                status = "QUERY FAILED: " + (str(e).splitlines() or [""])[0][:100]
        ok = ok and status == "OK"
        rows.append((name, n, status))
    return rows, ok


def retrieve(cols, strategy, case, q_emb, k_max):
    """Top-k_max chunks. For doc == 'both' the two filings are searched and merged:
    interleaved by rank when BALANCE_CROSS_DOC (so one filing cannot take every slot), else sorted by distance."""
    per_doc = {}
    for d in docs_of(case):
        col = cols[(d, strategy)]
        n = min(k_max, col.count())
        try:
            res = col.query(query_embeddings=[q_emb], n_results=n, include=["documents", "distances"])
        except Exception as e:
            raise RuntimeError(f"Query failed on collection '{COLLECTION_TEMPLATE.format(doc=d, strategy=strategy)}' "
                               f"({col.count()} chunks): {e}\nRun main(['--check-only']) to list the broken collections, "
                               f"then main(['--rebuild-to', <new_folder>]).") from e
        per_doc[d] = [{"id": cid, "doc": d, "text": text, "distance": float(dist)}
                      for cid, text, dist in zip(res["ids"][0], res["documents"][0], res["distances"][0])]
    if BALANCE_CROSS_DOC and len(per_doc) > 1:
        lists = [sorted(v, key=lambda x: x["distance"]) for v in per_doc.values()]
        merged = [x for tup in zip(*lists) for x in tup]
        longest = max(lists, key=len)
        merged += longest[min(len(l) for l in lists):]
        return merged[:k_max]
    items = [x for v in per_doc.values() for x in v]
    items.sort(key=lambda x: x["distance"])
    return items[:k_max]


def _space_of(col):
    try:
        cfg = col.configuration
        cfg = cfg if isinstance(cfg, dict) else dict(cfg)
        sp = (cfg.get("hnsw") or {}).get("space")
        if sp:
            return sp
    except Exception:
        pass
    try:
        return (col.metadata or {}).get("hnsw:space")
    except Exception:
        return None


def _create_collection(client, name, space):
    try:
        return client.create_collection(name=name, configuration={"hnsw": {"space": space}})
    except TypeError:                                    # older chromadb versions
        return client.create_collection(name=name, metadata={"hnsw:space": space})


def diagnose_store(path):
    """Read-only look at the store: file sizes per index folder, and whether chroma.sqlite3's vector-segment ids
    match the folders on disk. (Assumes Chroma's 'collections' and 'segments' tables; prints the table list if not.)"""
    import sqlite3
    path = Path(path)
    try:
        import chromadb
        print(f"chromadb {chromadb.__version__} | python: {sys.executable}")
    except Exception as e:
        print(f"chromadb could not be imported here: {e}")
    folders = {q.name: q for q in path.iterdir() if q.is_dir()}
    print(f"\nIndex folders on disk ({len(folders)}):")
    for n, q in sorted(folders.items()):
        sizes = ", ".join(f"{f.name}={f.stat().st_size}B" for f in sorted(q.iterdir()) if f.is_file())
        print(f"  {n}: {sizes or 'EMPTY FOLDER'}")
    db = sqlite3.connect(f"{(path / 'chroma.sqlite3').resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = db.execute("SELECT c.name, s.id, s.scope FROM segments s JOIN collections c ON c.id = s.collection "
                          "ORDER BY c.name").fetchall()
    except sqlite3.Error as e:
        print(f"\nCould not read the segments table: {e}")
        print("Tables:", [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")])
        return
    vec = [(n, i) for n, i, sc in rows if str(sc).upper().endswith("VECTOR")]
    print("\nCollections and their vector-segment ids (from chroma.sqlite3):")
    for n, i in vec:
        print(f"  {n:22s} {i}  {'folder present' if i in folders else '<-- NO MATCHING FOLDER ON DISK'}")
    orphans = sorted(set(folders) - {i for _, i in vec})
    if orphans:
        print("\nFolders on disk that no collection uses:", orphans)


def rebuild_store(src_dir, dst_dir, space=None, batch=500):
    """Copy every collection into a NEW folder, re-embedding the stored chunk text with your embedding model.
    Only chunk text + metadata are read from the old store (the broken part is its vector index); the old store is untouched."""
    if Path(src_dir).resolve() == Path(dst_dir).resolve():
        sys.exit("--rebuild-to must be a different folder from CHROMA_DIR.")
    if (Path(dst_dir) / "chroma.sqlite3").exists():
        sys.exit(f"{Path(dst_dir).resolve()} already contains a Chroma database; choose an empty folder.")
    src, dst = open_client(src_dir), _persistent(dst_dir)
    for name in sorted(collection_names(src)):
        old = src.get_collection(name)
        ids, docs, metas, off = [], [], [], 0
        while True:
            res = old.get(include=["documents", "metadatas"], limit=batch, offset=off)
            if not res["ids"]:
                break
            ids += res["ids"]
            docs += res["documents"]
            metas += res["metadatas"] if res.get("metadatas") is not None else [None] * len(res["ids"])
            off += len(res["ids"])
        if not ids:
            print(f"  {name}: EMPTY in the old store - nothing to copy (re-run your chunk/embed/add step for it)")
            continue
        new = _create_collection(dst, name, space or _space_of(old) or "l2")
        for i in range(0, len(ids), batch):
            m = metas[i:i + batch]
            new.upsert(ids=ids[i:i + batch], documents=docs[i:i + batch],
                       embeddings=_embed_texts(docs[i:i + batch]),
                       metadatas=m if all(m) else None)       # Chroma rejects empty metadata dicts
        print(f"  {name}: copied and re-embedded {new.count()} chunks")
    print(f"\nDone. Set CHROMA_DIR to: {Path(dst_dir).resolve()}")


# ============================================================== retrieval metrics
def covered(chunk_text, ev_norms):
    t = norm(chunk_text)
    return {i for i, e in enumerate(ev_norms) if e in t}


def retrieval_metrics(retrieved, ev_norms, ks):
    """A chunk 'covers' an evidence quote if it contains it. Returns metrics + per-chunk coverage."""
    cov = [covered(r["text"], ev_norms) for r in retrieved]
    out = {}
    for k in ks:
        top = cov[:k]
        union = set().union(*top) if top else set()
        out[f"Hit@{k}"] = float(len(union) == len(ev_norms))             # every evidence quote found
        out[f"Recall@{k}"] = len(union) / len(ev_norms)                  # share of evidence quotes found
        out[f"Precision@{k}"] = sum(1 for c in top if c) / k             # share of top-k chunks that contain evidence
    first = next((i for i, c in enumerate(cov) if c), None)
    out["MRR"] = 1 / (first + 1) if first is not None else 0.0           # 1 / rank of first chunk with evidence
    return out, cov


# ============================================================== generation + grading
GEN_PROMPT = """You answer questions about Regions Financial Corp. SEC filings using ONLY the numbered passages below.

Return ONLY a JSON object, no other text:
{{"answerable": true or false, "answer": "<concise answer>", "citations": [<passage numbers used>], "quotes": ["<short verbatim quote from a cited passage>"]}}

Rules:
- If the passages do not contain the answer, set "answerable" to false, "answer" to "Not found in the provided context", and "citations" to [].
- Never use outside knowledge. Cite every passage you used. Each quote must be copied exactly from a cited passage and be under 25 words.
- Write plain text inside the JSON strings: do not add backslash escapes (no \\u sequences); quotes may contain ordinary characters only.

Passages:
{context}

Question: {question}"""

JUDGE_PROMPT = """You are grading an answer against a reference answer.

Question: {question}
Reference answer: {reference}
Model answer: {answer}

The model answer is correct if it states the same facts and numbers as the reference answer (same values; rounding the reference itself uses is fine).
Extra correct detail is fine. Missing or wrong numbers make it incorrect.
Return ONLY JSON, verdict first: {{"correct": true or false, "reason": "<at most 15 words>"}}"""

def pack_context(retrieved, gen_k, budget):
    """Take up to gen_k chunks in rank order until the TOTAL character budget is used up.
    Only the lowest-ranked included chunk is cut. Same budget for every strategy -> equal-context comparison."""
    included, used = [], 0
    for c in retrieved[:gen_k]:
        text = c["text"]
        if MAX_CHARS_PER_CHUNK:
            text = text[:MAX_CHARS_PER_CHUNK]                        # safety cap on a single chunk
        if budget:
            room = budget - used
            if room <= 0:
                break
            text = text[:room]
        included.append({**c, "text": text})
        used += len(text)
    return included


def build_context(chunks):
    """FIX: markdown backslash escapes are removed from the text shown to the model (see clean_md)."""
    return "\n\n".join(f"[{i}] (source: {c['doc']}, id: {c['id']})\n{clean_md(c['text'])}"
                       for i, c in enumerate(chunks, 1))


def generate_and_grade(case, ctx, ev_norms):
    prompt = GEN_PROMPT.format(context=build_context(ctx), question=case["question"])
    raw = call_llm(prompt)
    data = parse_json(raw)
    if data is None:                                                 # one retry on a JSON parse failure
        raw = call_llm(prompt + "\n\nReturn ONLY the JSON object, nothing else. Keep it short and valid.")
        data = parse_json(raw)
    parse_ok = data is not None
    data = data or {"answerable": None, "answer": raw, "citations": [], "quotes": []}

    cov_ctx = [covered(c["text"], ev_norms) for c in ctx]            # coverage measured on the text actually sent
    cited_idx = [i for i in data.get("citations", []) if isinstance(i, int) and 1 <= i <= len(ctx)]
    cited_text = norm(" ".join(ctx[i - 1]["text"] for i in cited_idx))
    quotes = [q for q in data.get("quotes", []) if isinstance(q, str) and q.strip()]
    nan = float("nan")
    out = {
        "answer": data.get("answer"), "model_answerable": data.get("answerable"), "parse_ok": parse_ok,
        "cited_ids": [ctx[i - 1]["id"] for i in cited_idx], "quotes": quotes,
        "quotes_verified": (sum(norm(q) in cited_text for q in quotes) / len(quotes)) if quotes else nan,
        "ctx_chars": sum(len(c["text"]) for c in ctx),               # chars actually sent
        "n_chunks_sent": len(ctx),
        "evidence_in_ctx": (float(set().union(*cov_ctx) == set(range(len(ev_norms)))) if ev_norms else nan),
        "correct": nan, "judge_reason": None, "cited_hit": nan, "abstained_correctly": nan,
    }
    abstained = data.get("answerable") is False
    if not case["answerable"]:
        out["abstained_correctly"] = float(abstained)
    else:
        out["cited_hit"] = float(any(cov_ctx[i - 1] for i in cited_idx))   # cited chunk really contains gold evidence
        if abstained:
            out["correct"], out["judge_reason"] = 0.0, "model abstained"
        else:
            jp = JUDGE_PROMPT.format(question=case["question"], reference=case["reference_answer"],
                                     answer=out["answer"])
            verdict = parse_verdict(call_llm(jp, max_tokens=600)) or parse_verdict(call_llm(jp, max_tokens=600))  # retry once
            if verdict and isinstance(verdict.get("correct"), bool):
                out["correct"], out["judge_reason"] = float(verdict["correct"]), verdict.get("reason")
    return out


def is_complete(res, case):
    """FIX: only fully-scored results are cached, so a failed call is retried on the next run instead of being reused."""
    if not res["parse_ok"]:
        return False
    if case["answerable"] and res["model_answerable"] is not False and pd.isna(res["correct"]):
        return False                                                  # judge output could not be parsed
    return True


# ============================================================== reporting
def md_table(df, floatfmt="{:.3f}"):
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for v in row:
            if isinstance(v, numbers.Real) and not pd.isna(v):
                cells.append(floatfmt.format(v))
            else:
                cells.append("" if pd.isna(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summarize(rows, ks, gen_k, ceilings, strategies, with_gen):
    df = pd.DataFrame(rows)
    ans = df[df["has_evidence"]]
    ret_cols = ([f"Hit@{k}" for k in ks] + [f"Recall@{k}" for k in ks] +
                [f"Precision@{k}" for k in ks] + ["MRR"])
    ret = ans.groupby("strategy")[ret_cols].mean().reindex(strategies)
    ret.insert(0, "Evidence intact", pd.Series(ceilings))
    ret.index.name = "Chunking"

    pieces = {"Retrieval (answerable cases, n=%d)" % ans["case_id"].nunique(): ret}
    rk = REPORT_K if f"Recall@{REPORT_K}" in ans.columns else gen_k
    by_type = ans.pivot_table(index="question_type", columns="strategy",
                              values=f"Recall@{rk}", aggfunc="mean")[strategies]
    pieces[f"Recall@{rk} by question type"] = by_type

    gen = None
    if with_gen:
        df["judge_failed"] = df["has_evidence"] & df["correct"].isna() & (df["model_answerable"] != False)
        a = df[df["has_evidence"]]
        u = df[~df["has_evidence"]]
        gen = pd.DataFrame({                                          # nulls count as wrong
            "Answer accuracy":  a.groupby("strategy")["correct"].apply(lambda s: s.fillna(0).mean()),
            "Citation hit":     a.groupby("strategy")["cited_hit"].apply(lambda s: s.fillna(0).mean()),
            "Evidence in ctx":  a.groupby("strategy")["evidence_in_ctx"].mean(),
            "Quote verified":   a.groupby("strategy")["quotes_verified"].mean(),
            "Abstain accuracy": u.groupby("strategy")["abstained_correctly"].mean(),
            "Parse failures":   df.groupby("strategy")["parse_ok"].apply(lambda s: int((~s.astype(bool)).sum())),  # FIX: now reported
            "Judge failures":   df.groupby("strategy")["judge_failed"].sum().astype(int),     # NEW
            "Avg context chars": df.groupby("strategy")["ctx_chars"].mean(),
        }).reindex(strategies)
        gen.index.name = "Chunking"
        pieces[f"Generation (up to {gen_k} chunks, budget-capped, passed to {LLM_MODEL})"] = gen
    return df, ret, by_type, gen, pieces


# ============================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=CASES_PATH)
    ap.add_argument("--chroma-dir", default=CHROMA_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--k", type=int, nargs="+", default=K_VALUES)
    ap.add_argument("--gen-k", type=int, default=GEN_K)
    ap.add_argument("--ctx-budget", type=int, default=CTX_BUDGET,
                    help="max TOTAL context chars sent to the LLM, same for every strategy (0 = no limit)")
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES)
    ap.add_argument("--limit", type=int, default=None, help="only the first N cases (smoke test)")
    ap.add_argument("--retrieval-only", action="store_true", help="skip answer generation and grading (no LLM cost)")
    ap.add_argument("--check-only", action="store_true", help="only report which collections are empty/unqueryable, then exit")
    ap.add_argument("--diagnose", action="store_true", help="print index-folder sizes and check them against chroma.sqlite3, then exit")
    ap.add_argument("--rebuild-to", default=None, metavar="DIR",
                    help="copy all collections into a NEW folder, re-embedding the stored text, then exit")
    ap.add_argument("--space", choices=["l2", "cosine", "ip"], default=None,
                    help="distance metric for --rebuild-to (default: same as the old collection, else l2)")
    if argv is None:
        # Jupyter/IPython injects its own arguments (--f=...) into sys.argv, which argparse rejects
        argv = [] if "ipykernel" in sys.modules else sys.argv[1:]
    args = ap.parse_args(argv)
    for p in (args.cases, args.chroma_dir, args.out_dir):
        if any(ord(ch) < 32 for ch in str(p)):
            sys.exit(f"Bad path {p!r}: it contains a control character, usually from a Windows backslash "
                     "escape (e.g. \\r or \\t). Use forward slashes or a raw string, "
                     "e.g. r'src\\llm_engineering\\outputs\\rag_eval_out'.")

    if args.diagnose:
        diagnose_store(args.chroma_dir)
        return
    if args.rebuild_to:
        for p in [args.rebuild_to]:
            if any(ord(ch) < 32 for ch in str(p)):
                sys.exit(f"Bad path {p!r}: control character (use forward slashes or a raw string).")
        rebuild_store(args.chroma_dir, args.rebuild_to, args.space)
        return
    if not Path(args.cases).is_file():
        sys.exit(f"Test-case file not found: {Path(args.cases).resolve()}\n"
                 "A leading '/' (or '\\\\') means the drive root, not your project's src folder.")
    cases = load_cases(args.cases)[: args.limit]
    ks = sorted(set(args.k + [args.gen_k, REPORT_K]))
    k_max = max(ks)
    strategies = args.strategies
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    client = open_client(args.chroma_dir)
    needed_docs = sorted({d for c in cases for d in docs_of(c)})
    cols = load_collections(client, needed_docs, strategies)

    health, healthy = check_collections(cols, embed_query("health check"))
    print("Collection health check:")
    for name, n, status in health:
        print(f"  {name:22s} {n:>7} chunks   {status}")
    if not healthy or args.check_only:
        if not healthy:
            sys.exit("\nAt least one collection is empty or cannot be queried. Fix: main(['--rebuild-to', <new_folder>]) "
                     "re-embeds the stored chunk text into a fresh store (EMPTY collections must be re-indexed from your chunks).")
        return

    # ceiling: share of evidence quotes that survive chunking inside ONE chunk (retrieval cannot exceed this)
    corpus = {key: [norm(t) for t in load_corpus(col)] for key, col in cols.items()}
    ceilings = {}
    for s in strategies:
        total = intact = 0
        for c in cases:
            for ev in c["evidence"]:
                total += 1
                e = norm(ev)
                intact += any(e in chunk for d in docs_of(c) for chunk in corpus[(d, s)])
        ceilings[s] = intact / total if total else float("nan")

    cache_path = out / "llm_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    rows, total_jobs, job = [], len(cases) * len(strategies), 0
    for c in cases:
        q_emb = embed_query(c["question"])          # embed once; reused for every chunking strategy
        ev_norms = [norm(e) for e in c["evidence"]]
        for s in strategies:
            job += 1
            retrieved = retrieve(cols, s, c, q_emb, k_max)
            row = {"case_id": c["id"], "strategy": s, "doc": c["doc"], "question_type": c["question_type"],
                   "answerable": c["answerable"], "has_evidence": bool(ev_norms)}
            cov = [set() for _ in retrieved]
            if ev_norms:
                m, cov = retrieval_metrics(retrieved, ev_norms, ks)
                row.update(m)
            row["retrieved"] = [{"id": r["id"], "doc": r["doc"], "distance": round(r["distance"], 4),
                                 "covers_evidence": sorted(cv)} for r, cv in zip(retrieved, cov)]
            if not args.retrieval_only:
                ctx = pack_context(retrieved, args.gen_k, args.ctx_budget)
                sig = hashlib.md5("".join(x["id"] + x["text"] for x in ctx).encode()).hexdigest()[:8]
                # FIX: PIPELINE_VERSION in the key, so changing prompt/parsing/context code never reuses old results
                key = f"{PIPELINE_VERSION}|{c['id']}|{s}|{args.gen_k}|{args.ctx_budget}|{LLM_MODEL}|{sig}"
                if key in cache:
                    res = cache[key]
                else:
                    res = generate_and_grade(c, ctx, ev_norms)
                    if is_complete(res, c):                  # FIX: failed / unscored results are NOT cached -> retried next run
                        cache[key] = res
                        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                    else:
                        print(f"  [note] {c['id']} | {s}: parse/judge failure (not cached, will be retried next run)",
                              flush=True)
                row.update(res)
            rows.append(row)
            print(f"[{job}/{total_jobs}] {c['id']} | {s:8s}"
                  + (f" | MRR={row['MRR']:.2f}" if ev_norms else " | (unanswerable)"), flush=True)

    df, ret, by_type, gen, pieces = summarize(rows, ks, args.gen_k, ceilings, strategies, not args.retrieval_only)

    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    ret.to_csv(out / "metrics_retrieval.csv")
    if gen is not None:
        gen.to_csv(out / "metrics_generation.csv")

    report = ["# RAG evaluation by chunking strategy\n"]
    for title, table in pieces.items():
        report += [f"## {title}\n", md_table(table), ""]
    report += [
        "Notes: Hit@k = every evidence quote found in the top-k chunks; Recall@k = share of evidence quotes found; "
        "Precision@k = share of top-k chunks containing evidence; MRR = mean 1/rank of the first chunk containing evidence; "
        "Evidence intact = share of evidence quotes that sit whole inside a single chunk (the ceiling for retrieval); "
        f"Evidence in ctx = share of questions whose evidence is still inside the (budget-capped, {args.ctx_budget} chars) "
        "context actually sent to the LLM. Answer accuracy and Citation hit count unscored/failed cases as wrong. "
        "Parse failures = answers whose JSON could not be parsed even after the tolerant parser and one retry.",
    ]
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n" + "\n".join(report))
    print(f"\nSaved to {out.resolve()}: report.md, metrics_retrieval.csv, results.jsonl"
          + ("" if gen is None else ", metrics_generation.csv, llm_cache.json"))


if __name__ == "__main__":
    # main(["--rebuild-to", str(BASE_DIR / "db" / "chroma_store_rebuilt")])
    main(["--gen-k", "10", "--ctx-budget", "0"])
    # main(["--gen-k","10"])