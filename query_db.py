# query_db.py
# Purpose: Retrieve top-K relevant chunks from ChromaDB.
# Now supports optional metadata filters so we can scope searches
# to specific sections, pages, or documents.

import chromadb # type: ignore
from sentence_transformers import SentenceTransformer #type: ignore

CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "pdf_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Number of chunks to retrieve per query. Tunable — try 3, 5, 10.
TOP_K = 5



print(f"Connecting to ChromaDB at: ./{CHROMA_DB_DIR}/")
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

# get_collection (not create) — fails loudly if indexing wasn't run first.
collection = client.get_collection(name=COLLECTION_NAME)
print(f"Loaded collection with {collection.count()} vectors.")


#Load the SAME embedding model used during indexing
# ---------------------------------------------------------------------------
# Critical: questions must be embedded with the same model as the chunks,
# or they live in different vector spaces and similarity is meaningless.
print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Step 2: The retrieve function — now accepts an optional `where` filter
def retrieve(question: str, top_k: int = TOP_K, where: dict | None = None):
    """
    Retrieve top_k chunks similar to the question, optionally filtered by metadata.

    Parameters:
        question: the user's question (string)
        top_k: how many chunks to return (default 5)
        where: optional filter dict using ChromaDB's MongoDB-style syntax
               Examples:
                 {"section": "abstract"}                  → only abstract chunks
                 {"section": {"$ne": "references"}}        → exclude references
                 {"page": {"$lte": 5}}                     → only pages 1-5
    """
    # Print a header so the output is readable when we run multiple demos
    print(f"\n{'=' * 70}")
    print(f"QUESTION: {question}")
    if where:
        print(f"FILTER:   {where}")
    else:
        print("FILTER:   (none — searching all chunks)")
    print(f"{'=' * 70}")

    # Embed the question. .tolist() converts NumPy array to plain Python list
    # (which is what ChromaDB wants).
    question_embedding = model.encode(question).tolist()

    # Build the query arguments. We build them as a dict first, then unpack
    # them when calling collection.query().
    #
    # Why? Because if `where` is None, we DON'T want to pass `where=None`
    # to ChromaDB — that causes an error. We only include `where` in the
    # arguments if the caller actually provided one.
    query_kwargs = {
        "query_embeddings": [question_embedding],   # list of vectors (we only have 1)
        "n_results": top_k,                          # how many results to return
        "include": ["documents", "metadatas", "distances"],   # what to include in response
    }

    # Only add the where filter if it was provided.
    # This is the v2 upgrade — everything above is identical to v1.
    if where:
        query_kwargs["where"] = where

    # **query_kwargs unpacks the dict as keyword arguments.
    # Equivalent to: collection.query(query_embeddings=..., n_results=..., ...)
    results = collection.query(**query_kwargs)

    # ChromaDB returns results wrapped in an outer list because query() supports
    # multiple questions at once. We only sent ONE question, so we take index [0].
    #
    # Example shape: results["documents"] = [["chunk 1 text", "chunk 2 text", ...]]
    # The outer [...] is the batch wrapper (one entry per query).
    # The inner [...] is the list of K retrieved chunks.
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    # Edge case: filter might be so strict that no chunks match.
    # Handle gracefully instead of crashing.
    if not docs:
        print("\n⚠️ No chunks matched. Your filter may be too restrictive.")
        return [], [], []

    # Print the results in a readable format.
    print(f"\nRetrieved {len(docs)} chunks:\n")
    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        # Convert cosine distance back to cosine similarity for human readability.
        # ChromaDB returns DISTANCE (lower = better) but humans think in
        # SIMILARITY (higher = better). With cosine: similarity = 1 - distance.
        similarity = 1 - dist
        print(f"--- Rank {rank} | page={meta['page']} | section={meta['section']} | similarity={similarity:.4f} ---")
        print(doc[:200] + "...\n")

    return docs, metas, dists
    
# ---------------------------------------------------------------------------
# Step 3: Demo block — runs only when this script is executed directly,
# not when imported by another script (like rag_app.py later).
if __name__ == "__main__":
    # Pick a question that exists in your paper.
    # For "Attention Is All You Need" — multi-head attention is a clean test.
    question = "How does multi-head attention work?"

    # ---- Demo 1: No filter (baseline, same behavior as v1) ----
    # Searches across ALL chunks, including bibliography.
    # This is your reference point — what we want to improve on.
    print("\n\n>>> DEMO 1: No filter — searches the whole document")
    retrieve(question, top_k=3)

    # ---- Demo 2: Exclude references and bibliography ----
    # The "$nin" operator means "not in this list."
    # ChromaDB skips any chunk whose section field matches one of these
    # BEFORE doing the similarity search. Result: cleaner content-only matches.
    #
    # Example of how this filter works:
    #   chunk's section = "abstract"           → kept (not in the list)
    #   chunk's section = "model architecture" → kept
    #   chunk's section = "references"         → DROPPED
    #   chunk's section = "bibliography"       → DROPPED
    print("\n\n>>> DEMO 2: Exclude references and bibliography")
    retrieve(
        question,
        top_k=3,
        where={"section": {"$nin": ["references", "bibliography"]}},
    )

    # ---- Demo 3: Only the first 5 pages ----
    # The "$lte" operator means "less than or equal."
    # Useful if a user knows the answer is in the early part of the document.
    #
    # Example of how this filter works:
    #   chunk on page 1  → kept
    #   chunk on page 5  → kept (boundary inclusive with $lte)
    #   chunk on page 6  → DROPPED
    #   chunk on page 15 → DROPPED
    print("\n\n>>> DEMO 3: Only pages 1-5")
    retrieve(question, top_k=3, where={"page": {"$lte": 5}})

    # ---- Demo 4: Combine two filters using $and ----
    # "$and" takes a LIST of conditions. ALL must be true for a chunk to pass.
    # Here: chunks must be on pages 1-8 AND NOT in references.
    #
    # Why use $and? Because ChromaDB requires it when you have multiple
    # conditions on different metadata fields. You can't just write:
    #   {"page": {"$lte": 8}, "section": {"$ne": "references"}}
    # That syntax is invalid in ChromaDB; it needs the explicit $and wrapper.
    print("\n\n>>> DEMO 4: Pages 1-8 AND not in references")
    retrieve(
        question,
        top_k=3,
        where={
            "$and": [
                {"page": {"$lte": 8}},
                {"section": {"$ne": "references"}},
            ]
        },
    )

    # ---- Demo 5: The vague question test — where filtering really shines ----
print("\n\n>>> DEMO 5a: 'What is this document about?' — NO FILTER")
retrieve("What is this document about?", top_k=3)

print("\n\n>>> DEMO 5b: 'What is this document about?' — EXCLUDE references")
retrieve(
    "What is this document about?",
    top_k=3,
    where={"section": {"$nin": ["references", "bibliography"]}},
)