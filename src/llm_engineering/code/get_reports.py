"""This script is used to get the reports from the SEC Edgar database. 
The reports are downloaded in the form of text files and stored in the data directory."""
# from langchain_text_splitters import MarkdownHeaderTextSplitter

# # 1. Get the markdown text from edgartools
# markdown_text = filing.markdown()

# # 2. Split your 10-Q by its structural headers (e.g., Item 1, Item 2)
# # This ensures tables stay intact within their corresponding text sections
# headers_to_split_on = [
#     ("#", "Header 1"),
#     ("##", "Header 2"),
#     ("###", "Header 3"),
# ]

# splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
# chunks = splitter.split_text(markdown_text)

# Before using edgartools, you must set your identity to comply with SEC requirements
import os
from edgar import set_identity, Company
from llm_engineering.utils.chunk_embed import chunk_by_char, chunk_by_section, chunk_by_sentence, embed_texts
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from dotenv import load_dotenv
import chromadb as cdb
load_dotenv()
import torch
login(token = os.getenv("HUGGING_FACE_TOKEN"))
set_identity("Ziyad Maknojia ziyadm3@gmail.com")
set_identity("ziyadm3@gmail.com")
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


tenq = (Company("RF")
        .get_filings(form = "10-Q", filing_date = "2026-08-06")[0]).markdown()
                    
eightk_pull = (Company("RF")
          .get_filings(form = "8-K", filing_date = "2026-07-17")[0])

eightk = ""
for exhibit in eightk_pull.attachments.exhibits:
    doc_name = exhibit.document.lower()
    
    # Target BOTH EX-99.1 and EX-99.2 documents
    if "exhibit991" in doc_name or "exhibitx992" in doc_name:
        print(f"Extracting complete contents from: {exhibit.document}")
        
        # Pull text (or use exhibit.markdown() if you want to keep financial tables clean)
        content_text = exhibit.markdown() 
        
        # Append it to our master data container
        eightk += f"\n\n==================== START OF {exhibit.document.upper()} ====================\n\n"
        eightk += content_text

# 4. Verify or write the combined content out
print(f"\nExtraction Finished! Combined length of text: {len(eightk)} characters.")

# Chunking the 10-Q and 8-K reports using different methods
tenq_chunk_char = chunk_by_char(tenq)
tenq_chunk_section = chunk_by_section(tenq)
tenq_chunk_sentence = chunk_by_sentence(tenq)

eightk_chunk_char = chunk_by_char(eightk)
eightk_chunk_section = chunk_by_section(eightk)
eightk_chunk_sentence = chunk_by_sentence(eightk)

# Loading Hugging Face model for embeddings. Using free tier model of BAAI/bge-m3 model. 
# This model is a multilingual model that can be used for various NLP tasks.

model = SentenceTransformer('BAAI/bge-m3')

# Fitting the model to the chunks and getting embeddings for each chunk
tenq_chunk_char_embeddings = embed_texts(tenq_chunk_char, model, device = device)
tenq_chunk_section_embeddings = embed_texts(tenq_chunk_section, model, device = device)
tenq_chunk_sentence_embeddings = embed_texts(tenq_chunk_sentence, model, device = device)
    
eightk_chunk_char_embeddings = embed_texts(eightk_chunk_char, model, device = device)
eightk_chunk_section_embeddings = embed_texts(eightk_chunk_section, model, device = device)
eightk_chunk_sentence_embeddings = embed_texts(eightk_chunk_sentence, model, device = device)

# Setting up ChromaDB client to store the embeddings and chunks for later retrieval and analysis.

chroma_client = cdb.PersistentClient(path = "Z:\Projects\LLM-engineering\src\llm_engineering\db\chroma")

datasets = [
    # (doc_type, strategy, chunks, embeddings)
    ("tenq",   "char",     tenq_chunk_char,       tenq_chunk_char_embeddings),
    ("tenq",   "section",  tenq_chunk_section,    tenq_chunk_section_embeddings),
    ("tenq",   "sentence", tenq_chunk_sentence,   tenq_chunk_sentence_embeddings),
    ("eightk", "char",     eightk_chunk_char,     eightk_chunk_char_embeddings),
    ("eightk", "section",  eightk_chunk_section,  eightk_chunk_section_embeddings),
    ("eightk", "sentence", eightk_chunk_sentence, eightk_chunk_sentence_embeddings),
]

collections = {}
for doc_type, strategy, chunks, embeddings in datasets:
    name = f"{doc_type}_{strategy}"
    col = chroma_client.get_or_create_collection(name=name)
    col.upsert(
        ids=[f"{name}_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings,
        metadatas=[{"doc_type": doc_type, "strategy": strategy, "chunk_index": i}
                   for i in range(len(chunks))],
    )
    collections[name] = col