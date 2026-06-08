# index_pdf.py

import chromadb                                       # The vector database
from sentence_transformers import SentenceTransformer #type: ignore  # The embedding library

# Reuse the chunking functions we built in Stage 2 — DRY principle.
from chunk_pdf import extract_pages_from_pdf, chunk_with_metadata

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
PDF_PATH = "document.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Where ChromaDB will save its files. Just a folder name — it creates it
# automatically. Anything inside is the persisted database state.
CHROMA_DB_DIR = "chroma_db"

# A "collection" in Chroma is like a table in SQL — a named group of vectors
# that all share a schema. One database can hold many collections.
COLLECTION_NAME = "pdf_chunks"


# ---------------------------------------------------------------------------
# Step 1: Set up the ChromaDB client (persistent — saves to disk)
# ---------------------------------------------------------------------------
# PersistentClient saves to disk. There's also an in-memory client, but we
# want our work to survive restarts, so persistent is the choice for RAG.
print(f"Connecting to ChromaDB at: ./{CHROMA_DB_DIR}/")
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)


# Check if a collection with our name already exists from a previous run.
# list_collections() returns a list of Collection objects.
# [c.name for c in ...] is a list comprehension that pulls just the names.
#
# Example: if you ran this script yesterday, list_collections() might return
# Collection objects whose .name values are ["pdf_chunks"].
# We extract just the names into a plain list to check membership.
existing = [c.name for c in client.list_collections()]
if COLLECTION_NAME in existing:
    print(f"Collection '{COLLECTION_NAME}' already exists — deleting it for a clean rebuild.")
    client.delete_collection(name=COLLECTION_NAME)

# Create the collection.
# metadata={"hnsw:space": "cosine"} tells Chroma to use cosine similarity
# (measures angle between vectors - matches what we want for embeddings).
collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}, #Hierarchical Navigable Small World
)
print(f"Created fresh collection: {COLLECTION_NAME}")


# ---------------------------------------------------------------------------
# Step 2: use the new v2 functoons to get chunks with metadata attached
# ---------------------------------------------------------------------------
print("\n--- Chunking PDF with metadata ---")
# extract_pages_from_pdf returns: [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]
# (Pages kept separate so we can track page numbers per chunk.)
pages = extract_pages_from_pdf(PDF_PATH)

# chunk_with_metadata returns:
# [{"text": "...", "page": 4, "section": "abstract", "chunk_index": 0}, ...]
# Each chunk is now a DICT with text + metadata, not just a plain string.
chunks = chunk_with_metadata(pages, CHUNK_SIZE, CHUNK_OVERLAP)


# ---------------------------------------------------------------------------
# Step 3: Embed each chunk's text using sentence-transformers
# ---------------------------------------------------------------------------
print(f"\n--- Loading embedding model: {EMBEDDING_MODEL} ---")
model = SentenceTransformer(EMBEDDING_MODEL)

# Our chunks are dicts like {"text": "...", "page": 4, ...}.
# But the embedding model only wants TEXT, not metadata.
# So we extract just the "text" field from each chunk into a separate list.
#
# This [c["text"] for c in chunks] is called a LIST COMPREHENSION.
# It's a one-line way to build a new list by transforming each item in another list.
#
# Example with 2 chunks:
#   chunks = [{"text": "hello", "page": 1}, {"text": "world", "page": 2}]
#   chunk_texts = ["hello", "world"]
#
# Equivalent long version:
#   chunk_texts = []
#   for c in chunks:
#       chunk_texts.append(c["text"])
chunk_texts = [c["text"] for c in chunks]

print(f"\n--- Embedding {len(chunks)} chunks ---")
embeddings = model.encode(chunk_texts, show_progress_bar=True)

# embeddings is a NumPy array of shape (num_chunks, 384).
# ChromaDB wants a plain Python list-of-lists, so we convert with .tolist().
#
# Example:
#   NumPy array: [[0.1, 0.2, ...], [0.3, 0.4, ...]]
#   After .tolist(): [[0.1, 0.2, ...], [0.3, 0.4, ...]]
# (Looks the same when printed, but the underlying type changed.)
embeddings_list = embeddings.tolist()


# ---------------------------------------------------------------------------
# Step 4: Build the metadata dict from each chunk
# ---------------------------------------------------------------------------
# Chroma's add() takes 4 parallel lists, all the same length.
#   ids        → unique ID per chunk
#   embeddings → the 384-dim vectors
#   documents  → the text per chunk
#   metadatas  → a dict of metadata per chunk - this is v2 upgrade)
#
# IDs: just unique strings. "chunk_0", "chunk_1", etc.
#
# Example: for 3 chunks, ids = ["chunk_0", "chunk_1", "chunk_2"]
ids = [f"chunk_{c['chunk_index']}" for c in chunks]

# Metadatas: one dict per chunk. IMPORTANT RULE — values must be primitives
# (str, int, float, bool). No lists or nested dicts.
#
# Example for one chunk:
#   {
#     "source": "document.pdf",     # str
#     "page": 4,                    # int
#     "section": "abstract",        # str
#     "chunk_index": 0,             # int
#     "chunk_length": 998,          # int
#   }
metadatas = [
    {
        "source": PDF_PATH,
        "page": c["page"],
        "section": c["section"],
        "chunk_index": c["chunk_index"],
        "chunk_length": len(c["text"]),
    }
    for c in chunks
]

# Step 5: Insert everything into ChromaDB in one call
print(f"\n--- Adding {len(chunks)} chunks to ChromaDB ---")
collection.add(
    ids=ids,
    embeddings=embeddings_list,
    documents=chunk_texts,
    metadatas=metadatas,
)

print(f"\n✅ Indexed {collection.count()} chunk with metadata.")

# ---------------------------------------------------------------------------
# Step 6: Sanity check — pull 3 items back out of the DB to verify
# the metadata actually made it in correctly.
print("\n--- Sample stored items ---")

# collection.get(limit=3, include=["metadatas", "documents"]) returns:
#   {
#     "ids": ["chunk_0", "chunk_1", "chunk_2"],
#     "documents": ["text of chunk 0", "text of chunk 1", "text of chunk 2"],
#     "metadatas": [{"page": 1, ...}, {"page": 1, ...}, {"page": 2, ...}]
#   }
sample = collection.get(limit=3, include=["metadatas", "documents"])

# zip(list1, list2) pairs up items at the same index from two lists.
# Example: zip(["A", "B"], [1, 2]) gives us [("A", 1), ("B", 2)]
# enumerate() adds a counter starting from 0.
# So this loop iterates:
#   i=0, doc="text 0", meta={"page": 1, ...}
#   i=1, doc="text 1", meta={"page": 1, ...}
#   i=2, doc="text 2", meta={"page": 2, ...}
for i, (doc, meta) in enumerate(zip(sample["documents"], sample["metadatas"])):
    print(f"\nItem {i}: page={meta['page']}, section={meta['section']}")
    print(f"  Text preview: {doc[:120]}...")
