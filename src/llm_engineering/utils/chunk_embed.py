"""This module contains utility functions for text chunking.
Three types of chunking are included in this script: chunking by characters, chunking by section aka paragraphs heading and chunking by sentences
The script will be used to chunk text from a earnings call a considerable chunk size of 512 is used and about 11.7% i.e. 60 characters are overlapped 
to ensure that the context is not lost when chunking the text. A smaller chunk size can be used for other cases like FAQs, short snippets, etc.
A more sophisticated chunking method can be implemented using NLP and create chunks for related sentences.

Also, the script provides embedding functionality using the Hugging Face model BAAI/bge-m3. 
This model is a multilingual model that can be used for various NLP tasks. 
"""

import re


def chunk_by_char(text, chunk_size = 512, chunk_overlap = 60):
    chunks = []
    start_idx = 0
    
    while start_idx < len(text):
        end_idx = min(start_idx + chunk_size, len(text))
        chunk_text = text[start_idx:end_idx]
        chunks.append(chunk_text)
        
        start_idx = (
            end_idx - chunk_overlap if end_idx < len(text) else len(text)
        )
    
    return chunks

import re

FILE_MARK = re.compile(r"=+ START OF (\S+) =+")
# markdown headings (# .. ###) or a standalone **bold** line
HEADING = re.compile(r"^(?:#{1,3}\s+.+|\*\*[^*\n]{3,120}\*\*\s*)$")
# "headings" that are really footnotes, not sections
NOISE = re.compile(r"^#{1,3}\s*(?:\\?\*|NM\b|_+|See the company)", re.I)

def _split_blocks(text):
    blocks, head, buf = [], "", []
    for line in text.split("\n"):
        s = line.strip()
        if HEADING.match(s) and not NOISE.match(s):
            if "".join(buf).strip():
                blocks.append((head, "\n".join(buf).strip()))
            head, buf = s.lstrip("# ").strip("* ").strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        blocks.append((head, "\n".join(buf).strip()))
    return blocks

def _split_big(body, max_chars):
    if len(body) <= max_chars:
        return [body]
    parts, cur = [], ""
    for line in body.split("\n"):          # break on lines, never mid-row
        if cur and len(cur) + len(line) + 1 > max_chars:
            parts.append(cur.strip()); cur = ""
        cur += line + "\n"
    if cur.strip():
        parts.append(cur.strip())
    return parts

def chunk_by_section(document_text, min_chars=300, max_chars=2000, with_context=True):
    chunks = []
    pieces = FILE_MARK.split(document_text)
    files = [("", pieces[0])] + list(zip(pieces[1::2], pieces[2::2]))
    for fname, ftxt in files:
        if not ftxt.strip():
            continue
        merged, carry_head, carry_body = [], "", ""
        for head, body in _split_blocks(ftxt):
            if carry_body:                                  # merge tiny blocks forward
                head = carry_head or head
                body = carry_body + "\n\n" + body
                carry_head, carry_body = "", ""
            if len(body) < min_chars:
                carry_head, carry_body = head, body
            else:
                merged.append((head, body))
        if carry_body:
            if merged: merged[-1] = (merged[-1][0], merged[-1][1] + "\n\n" + carry_body)
            else: merged.append((carry_head, carry_body))
        for head, body in merged:
            for part in _split_big(body, max_chars):
                chunks.append(f"[{fname} | {head}]\n{part}" if with_context and head else part)
    return chunks

def chunk_by_sentence(text, max_sentences_per_chunk=5, overlap_sentences=1):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    
    chunks = []
    start_idx = 0
    
    while start_idx < len(sentences):
        end_idx = min(start_idx + max_sentences_per_chunk, len(sentences))
        current_chunk = sentences[start_idx:end_idx]
        chunks.append(" ".join(current_chunk))
        
        start_idx += max_sentences_per_chunk - overlap_sentences
        
        if start_idx < 0:
            start_idx = 0
    
    return chunks

def embed_texts(texts: list[str], 
                model, 
                device, 
                batch_size: int = 4,
                ) -> list[list[float]]:
    if not texts:
        return []

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        device=device,
        show_progress_bar=True,
    )

    return embeddings.tolist()

# def is_hit(retrieved_chunks, evidence):
#     # Hit only if every needed passage is covered somewhere in the top-k
#     joined = [norm(c) for c in retrieved_chunks]
#     return all(any(norm(ev) in c for c in joined) for ev in evidence)