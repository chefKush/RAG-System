# index_pdf.py
# Purpose: ONE-TIME job — chunk the PDF, embed every chunk, store everything
# in a ChromaDB collection persisted to disk. After this runs once, the DB
# is ready to be queried by query_db.py as many times as we want.

import chromadb                                       # The vector database
from sentence_transformers import SentenceTransformer #type: ignore  # The embedding library

# Reuse the chunking functions we built in Stage 2 — DRY principle.
from chunk_pdf import extract_text_from_pdf, split_into_chunks

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


# ---------------------------------------------------------------------------
# Step 2: Get or create the collection
# ---------------------------------------------------------------------------
# If we re-run this script, we DON'T want duplicate chunks piling up. So we
# delete any existing collection with the same name and create a fresh one.
# In production you'd handle this more carefully (e.g., upserts by ID),
# but for learning a clean reset is clearer.
existing = [c.name for c in client.list_collections()]
if COLLECTION_NAME in existing:
    print(f"Collection '{COLLECTION_NAME}' already exists — deleting it for a clean rebuild.")
    client.delete_collection(name=COLLECTION_NAME)

# Create the collection.
# metadata={"hnsw:space": "cosine"} tells Chroma to use cosine similarity
# for distance calculations — matches what we learned in Stage 3.
# Without this, Chroma defaults to L2 (Euclidean) which behaves differently.
collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}, #Hierarchical Navigable Small World
)
print(f"Created fresh collection: {COLLECTION_NAME}")


# ---------------------------------------------------------------------------
# Step 3: Chunk the PDF (Stage 2 logic, reused)
# ---------------------------------------------------------------------------
print("\n--- Chunking PDF ---")
full_text = extract_text_from_pdf(PDF_PATH)
chunks = split_into_chunks(full_text, CHUNK_SIZE, CHUNK_OVERLAP)


# ---------------------------------------------------------------------------
# Step 4: Embed all chunks (Stage 3 logic, reused)
# ---------------------------------------------------------------------------
print(f"\n--- Loading embedding model: {EMBEDDING_MODEL} ---")
model = SentenceTransformer(EMBEDDING_MODEL)

print(f"\n--- Embedding {len(chunks)} chunks ---")
embeddings = model.encode(chunks, show_progress_bar=True)

# ChromaDB wants embeddings as a list of lists (not a numpy array).
# .tolist() converts cleanly.
embeddings_list = embeddings.tolist()


# ---------------------------------------------------------------------------
# Step 5: Add everything to ChromaDB
# ---------------------------------------------------------------------------
# Chroma's add() takes four parallel lists, each with one entry per chunk:
#   ids        → unique string ID for each vector (we'll use "chunk_0", etc.)
#   embeddings → the 384-dim vectors
#   documents  → the original text (so we can retrieve it later)
#   metadatas  → arbitrary extra info per chunk (source, position, etc.)
#
# Note: Chroma stores the text alongside the vector — so when we query,
# we get the text BACK automatically. No second lookup needed. Convenient.

ids = [f"chunk_{i}" for i in range(len(chunks))]

metadatas = [
    {
        "source": PDF_PATH,
        "chunk_index": i,
        "chunk_length": len(chunks[i]),
    }
    for i in range(len(chunks))
]

print(f"\n--- Adding {len(chunks)} chunks to ChromaDB ---")
collection.add(
    ids=ids,
    embeddings=embeddings_list,
    documents=chunks,
    metadatas=metadatas,
)


# ---------------------------------------------------------------------------
# Step 6: Verify what we stored
# ---------------------------------------------------------------------------
count = collection.count()
print(f"\n✅ Collection now contains {count} vectors.")
print(f"✅ Database persisted to: ./{CHROMA_DB_DIR}/")
print("\nYou can now run query_db.py without re-running this script.") 


"""
data_level0.bin       ← The actual 384-dim embedding vectors
header.bin            ← HNSW index header
length.bin            ← Vector count metadata
link_lists.bin        ← HNSW graph connections between vectors
"""