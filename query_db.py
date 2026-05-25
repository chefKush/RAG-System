# query_db.py
# Purpose: Load the already-indexed ChromaDB and retrieve the most relevant
# chunks for any question. This is the FAST half of RAG — runs in milliseconds.

import chromadb
from sentence_transformers import SentenceTransformer #type: ignore

CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "pdf_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Number of chunks to retrieve per query. Tunable — try 3, 5, 10.
TOP_K = 5


# ---------------------------------------------------------------------------
# Step 1: Connect to the persisted database
# ---------------------------------------------------------------------------
# Notice we don't re-chunk or re-embed the PDF here. The DB already has
# everything. We just open it.
print(f"Connecting to ChromaDB at: ./{CHROMA_DB_DIR}/")
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

# get_collection (not create) — fails loudly if indexing wasn't run first.
collection = client.get_collection(name=COLLECTION_NAME)
print(f"Loaded collection '{COLLECTION_NAME}' with {collection.count()} vectors.")


# ---------------------------------------------------------------------------
# Step 2: Load the SAME embedding model used during indexing
# ---------------------------------------------------------------------------
# Critical: questions must be embedded with the same model as the chunks,
# or they live in different vector spaces and similarity is meaningless.
print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Step 3: The retrieve function — the heart of the query side of RAG
# ---------------------------------------------------------------------------
def retrieve(question: str, top_k: int = TOP_K):
    """
    Given a question, return the top_k most relevant chunks from the DB.
    """
    print(f"\n{'=' * 70}")
    print(f"QUESTION: {question}")
    print(f"{'=' * 70}")

    # Embed the question using the same model that embedded the chunks.
    # encode() returns a numpy array; we convert to list for Chroma.
    question_embedding = model.encode(question).tolist()

    # The ACTUAL similarity search.
    # Chroma takes the query vector, runs HNSW (approximate nearest neighbor)
    # internally, and returns the closest stored vectors plus their data.
    # n_results = how many to return. include= controls what fields come back.
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Chroma wraps results in an outer list because query() supports batching
    # multiple questions at once. We sent one question, so we read index [0].
    retrieved_docs = results["documents"][0]
    retrieved_metas = results["metadatas"][0]
    retrieved_dists = results["distances"][0]

    # IMPORTANT: Chroma returns DISTANCE, not SIMILARITY.
    # With cosine space: distance = 1 - cosine_similarity.
    # So a distance of 0.3 means a similarity of 0.7. Lower distance = better.
    print(f"\nRetrieved top {top_k} chunks (lower distance = more similar):\n")
    for rank, (doc, meta, dist) in enumerate(
        zip(retrieved_docs, retrieved_metas, retrieved_dists), start=1
    ):
        similarity = 1 - dist
        print(f"--- Rank {rank} | chunk_index={meta['chunk_index']} | "
              f"distance={dist:.4f} | similarity={similarity:.4f} ---") #FLOAT FORMATTING .4f , .3f etc controls how many decimal places to show
        # Print only the first 300 chars so output stays readable.
        print(doc[:300] + ("..." if len(doc) > 300 else ""))
        print()

    return retrieved_docs


# ---------------------------------------------------------------------------
# Step 4: Run a few test questions
# ---------------------------------------------------------------------------
# CHANGE THESE to questions actually relevant to YOUR PDF.
# Since your PDF is "Attention Is All You Need", I picked questions for it.
if __name__ == "__main__":
    test_questions = [
        "What is the Transformer architecture?",
        "How does self-attention work?",
        "What datasets were used for training?",
    ]

    for q in test_questions:
        retrieve(q)