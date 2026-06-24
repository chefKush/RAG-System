import chromadb # type: ignore
from  sentence_transformers import SentenceTransformer, CrossEncoder # type: ignore

CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "pdf_chunks"

BI_ENCODER_MODEL = "all-MiniLM-L6-v2"
# The cross-encoder: a completely different model — trained specifically
# to score relevance between a question and a passage.
# ms-marco = trained on 1M+ real Bing search queries (Microsoft).
# MiniLM-L-6-v2 = lightweight, fast on CPU, ~80MB download.
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RETRIEVAL_K = 25
RERANK_K = 3

print(f"Connecting to ChromaDB at: ./{CHROMA_DB_DIR}/")
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_collection(name=COLLECTION_NAME)
print(f"Loaded collection with {collection.count()} vectors.")

print(f"\nLoading bi-encoder model: {BI_ENCODER_MODEL}")
bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)
print("Bi-encoder ready")

print(f"\nLoading cross-encoder model: {CROSS_ENCODER_MODEL}")
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
# CrossEncoder takes (question, chunk) pairs and returns a relevance score.
# It does NOT produce embeddings — it produces a single float per pair.
# Higher score = more relevant. Scores are NOT constrained to 0-1 range
# (unlike cosine similarity). Raw logits from the model. That's fine —
# we only care about RELATIVE ranking, not absolute score values.
print("Cross-encoder ready")

def retrieve_and_rerank(question: str, where: dict | None = None):
    """
    Two-stage retrieval:
      Stage A: Bi-encoder retrieves RETRIEVAL_K candidates from ChromaDB
      Stage B: Cross-encoder re-scores all candidates, returns top RERANK_K

    Parameters:
        question: the user's question
        where: optional metadata filter (same syntax as query_db.py)
    """
    print(f"\n{'=' * 70}")
    print(f"QUESTION: {question}")
    if where:
        print(f"FILTER:   {where}")
    print(f"{'=' * 70}")

    # ----------------------------------------------------------------
    # STAGE A: Bi-encoder retrieval (same as query_db.py, just wider)
    # ----------------------------------------------------------------
    # We embed the question and ask ChromaDB for the top RETRIEVAL_K
    # most similar chunks. Same mechanism as v1/v2, but K=25 instead
    # of K=3. We're casting a wide net on purpose.
    print(f"\n[Stage A] Bi-encoder retrieving top {RETRIEVAL_K} candidates...")

    question_embedding = bi_encoder.encode(question).tolist()

    query_kwargs = {
        "query_embeddings": [question_embedding],
        "n_results":        RETRIEVAL_K,
        "include":          ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    # Unwrap from batch wrapper (same [0] indexing as query_db.py)
    candidates      = results["documents"][0]
    candidate_metas = results["metadatas"][0]
    candidate_dists = results["distances"][0]

    print(f"Retrieved {len(candidates)} candidates.")

    # Print the top 3 BI-ENCODER results BEFORE reranking.
    # This is our "before" snapshot — we'll compare it to "after" reranking.
    print(f"\n--- TOP 3 (BI-ENCODER, before reranking) ---")
    for i in range(min(3, len(candidates))):
        sim = 1 - candidate_dists[i]
        print(f"  Rank {i+1} | page={candidate_metas[i]['page']} | "
              f"section={candidate_metas[i]['section']} | "
              f"bi-encoder similarity={sim:.4f}")
        print(f"  {candidates[i][:150]}...\n")

    # ----------------------------------------------------------------
    # STAGE B: Cross-encoder reranking
    # ----------------------------------------------------------------
    
    # We now build a list of (question, chunk) pairs — one pair per candidate.
    # The cross-encoder scores each pair for relevance.
    #
    # Example with 3 candidates:
    #   question = "How does multi-head attention work?"
    #   pairs = [
    #     ("How does multi-head attention work?", "Multi-head attention allows..."),
    #     ("How does multi-head attention work?", "The Law will never be perfect..."),
    #     ("How does multi-head attention work?", "We employ h = 8 parallel..."),
    #   ]
    #
    # The cross-encoder looks at EACH PAIR JOINTLY and outputs a score.
    # Pair 1: high score (question and chunk clearly about the same topic)
    # Pair 2: low score  (chunk about law, nothing to do with attention)
    # Pair 3: high score (directly discusses multi-head attention)
    print(f"\n[Stage B] Cross-encoder re-scoring {len(candidates)} candidates...")

    pairs = [(question, chunk) for chunk in candidates]

    # cross_encoder.predict() takes a list of (question, chunk) pairs
    # and returns a list of scores — one score per pair.
    # All pairs are processed in one call (batched internally).
    #
    # Output example: [2.34, -1.23, 3.12, 0.45, -2.11, ...]
    # NOT probabilities. Raw scores. Higher = more relevant.
    scores = cross_encoder.predict(pairs)

    # Now attach each score to its corresponding candidate so we can sort.
    # zip() pairs the three parallel lists by index:
    #   candidates[0], candidate_metas[0], scores[0] → one entry
    #   candidates[1], candidate_metas[1], scores[1] → next entry
    #   ... etc
    #
    # We then sort by score (highest first) using the same lambda
    # trick we learned in chunk_pdf.py: key=lambda x: x[2] gives us
    # the score (third element of each tuple), negative to sort descending.
    scored = sorted(
        zip(candidates, candidate_metas, scores),
        key=lambda x: x[2],        # x[2] is the score
        reverse=True,               # highest score first
    )

    # Keep only the top RERANK_K results after sorting.
    # scored[:RERANK_K] slices the first RERANK_K items from the sorted list.
    top_reranked = scored[:RERANK_K]

    # ----------------------------------------------------------------
    # Print the AFTER comparison
    # ----------------------------------------------------------------
    print(f"\n--- TOP {RERANK_K} (CROSS-ENCODER, after reranking) ---")
    for rank, (doc, meta, score) in enumerate(top_reranked, start=1):
        print(f"  Rank {rank} | page={meta['page']} | "
              f"section={meta['section']} | "
              f"cross-encoder score={score:.4f}")
        print(f"  {doc[:150]}...\n")

    # Return the top reranked docs and their metadata for downstream use
    top_docs   = [doc   for doc, meta, score in top_reranked]
    top_metas  = [meta  for doc, meta, score in top_reranked]
    top_scores = [score for doc, meta, score in top_reranked]

    return top_docs, top_metas, top_scores

if __name__ == "__main__":
    # Test question — same one we used in Stage 1 so you can compare
    question = "How does multi-head attention work?"

    # Demo 1: No filter — compare bi-encoder vs cross-encoder rankings
    print("\n>>> DEMO 1: No filter")
    retrieve_and_rerank(question)

    # Demo 2: Exclude references + reranking combined
    # This is the full v2 pipeline: filter bad sections OUT first,
    # then rerank what's left. Best of both stages.
    print("\n\n>>> DEMO 2: Exclude references + reranking")
    retrieve_and_rerank(
        question,
        where={"section": {"$nin": ["references", "bibliography"]}},
    )

    # Demo 3: The dramatic test — a vague question where reranking helps most
    # Remember how "What is this document about?" returned garbage in v1?
    # Let's see if reranking cleans it up.
    print("\n\n>>> DEMO 3: Vague question — where reranking earns its keep")
    retrieve_and_rerank("What is this document about?")