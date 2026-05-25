# rag_app.py
# Purpose: The complete RAG application. User types a question, the system
# retrieves relevant chunks from ChromaDB, builds an augmented prompt, and
# sends it to Groq's LLM for a grounded answer.

import os
import chromadb
from dotenv import load_dotenv
from groq import Groq #type: ignore
from sentence_transformers import SentenceTransformer #type: ignore

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
load_dotenv()

CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "pdf_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

# How many chunks to retrieve and put into the prompt.
# Too few: might miss the answer. Too many: wastes tokens and adds noise.
# 3-5 is the sweet spot for most documents.
TOP_K = 4

# LLM temperature. RAG should be factual, so we use low temperature.
# 0.0 = deterministic, perfect for grounded Q&A.
TEMPERATURE = 0.1


# ---------------------------------------------------------------------------
# Step 1: Initialize all the components ONCE at startup
# ---------------------------------------------------------------------------
# Doing this here (outside the loop) means: we pay the loading cost once,
# then every user question is fast. This is how production CLI tools work.
print("=" * 70)
print("Initializing RAG application...")
print("=" * 70)

# Groq client (Stage 1)
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file.")
groq_client = Groq(api_key=api_key)
print("✅ Groq client ready")

# Embedding model (Stage 3)
print(f"Loading embedding model: {EMBEDDING_MODEL}...")
embed_model = SentenceTransformer(EMBEDDING_MODEL)
print("✅ Embedding model loaded")

# Vector DB connection (Stage 4)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = chroma_client.get_collection(name=COLLECTION_NAME)
print(f"✅ ChromaDB ready ({collection.count()} chunks indexed)")
print()


# ---------------------------------------------------------------------------
# Step 2: The retrieval function (essentially Stage 4 wrapped neatly)
# ---------------------------------------------------------------------------
def retrieve_chunks(question: str, top_k: int):
    """
    Embed the question, query ChromaDB, return top_k relevant chunks
    along with their similarity scores and source metadata.
    """
    # Embed the question with the SAME model used at indexing time.
    question_embedding = embed_model.encode(question).tolist()

    # Query the vector DB.
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Unwrap the batched response (we sent 1 question, read index [0]).
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Convert cosine distance → similarity for human-friendly display.
    similarities = [1 - d for d in distances]

    return chunks, metadatas, similarities


# ---------------------------------------------------------------------------
# Step 3: Build the augmented prompt (THE heart of RAG)
# ---------------------------------------------------------------------------
def build_prompt(question: str, chunks: list[str]) -> list[dict]:
    """
    Construct the full message list to send to the LLM.

    The structure:
    - System message: rules of behavior
    - User message: context + the actual question
    """
    # Join chunks into one block, clearly numbered so the LLM can refer to them.
    # Numbering also helps us spot in the output which chunk got used.
    context_block = "\n\n".join(
        f"[Chunk {i + 1}]\n{chunk}"
        for i, chunk in enumerate(chunks)
    )

    # The system prompt — the most important text in the whole app.
    # Read each instruction carefully; each one prevents a specific failure.
    system_prompt = (
        "You are a helpful assistant that answers questions based on the "
        "provided context from a document.\n\n"
        "RULES:\n"
        "1. Answer ONLY using information from the context below. "
        "Do not use outside knowledge.\n"
        "2. If the context does not contain the answer, reply: "
        "\"The document does not contain enough information to answer this.\"\n"
        "3. Be concise and direct. No filler phrases.\n"
        "4. When useful, reference which chunk(s) you used, e.g. \"[Chunk 2]\".\n"
    )

    # The user prompt — context first, question second.
    # Question at the END is intentional: research shows LLMs follow the
    # most recent instruction best, so we want the question fresh in mind.
    user_prompt = (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {question}\n\n"
        f"Based ONLY on the context above, answer the question."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ---------------------------------------------------------------------------
# Step 4: The generation function (Stage 1 logic, parametrized)
# ---------------------------------------------------------------------------
def generate_answer(messages: list[dict]) -> tuple[str, dict]:
    """
    Send the augmented prompt to Groq and return the answer + token usage.
    """
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
    )

    answer = response.choices[0].message.content
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    return answer, usage


# ---------------------------------------------------------------------------
# Step 5: The full RAG function — ties retrieval and generation together
# ---------------------------------------------------------------------------
def answer_question(question: str):
    """
    Full RAG flow with heavy print statements so the learner SEES every step.
    """
    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    # ---- RETRIEVE ----
    print(f"\n[1/3] Retrieving top {TOP_K} chunks from ChromaDB...")
    chunks, metadatas, similarities = retrieve_chunks(question, TOP_K)

    print(f"\nRetrieved chunks (with similarity scores):")
    for i, (chunk, meta, sim) in enumerate(zip(chunks, metadatas, similarities), 1):
        print(f"\n  ── Chunk {i} | chunk_index={meta['chunk_index']} | "
              f"similarity={sim:.4f} ──")
        # Print preview of each retrieved chunk.
        preview = chunk[:200].replace("\n", " ")
        print(f"  {preview}...")

    # ---- AUGMENT ----
    print(f"\n[2/3] Building augmented prompt with {len(chunks)} chunks...")
    messages = build_prompt(question, chunks)

    # Show exact size of what we're sending to the LLM. Useful for cost awareness.
    total_chars = sum(len(m["content"]) for m in messages)
    print(f"  Total prompt size: {total_chars} characters "
          f"(~{total_chars // 4} tokens estimated)")

    # ---- GENERATE ----
    print(f"\n[3/3] Sending to Groq ({LLM_MODEL})...")
    answer, usage = generate_answer(messages)

    # ---- DISPLAY ANSWER ----
    print("\n" + "─" * 70)
    print("ANSWER:")
    print("─" * 70)
    print(answer)
    print("─" * 70)

    # ---- DISPLAY METADATA ----
    print(f"\nTokens — prompt: {usage['prompt_tokens']}, "
          f"completion: {usage['completion_tokens']}, "
          f"total: {usage['total_tokens']}")


# ---------------------------------------------------------------------------
# Step 6: The CLI loop
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("RAG QUESTION-ANSWERING SYSTEM")
    print("Ask questions about your PDF. Type 'exit' or 'quit' to leave.")
    print("=" * 70)

    while True:
        try:
            question = input("\n❓ Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            # Handle Ctrl+C cleanly instead of crashing.
            print("\nGoodbye!")
            break

        # Allow exiting gracefully.
        if question.lower() in {"exit", "quit", "q", ""}:
            print("Goodbye!")
            break

        # Defend against accidental empty input.
        if len(question) < 3:
            print("Please ask a real question (at least 3 characters).")
            continue

        try:
            answer_question(question)
        except Exception as e:
            # Catch errors per-question so one bad query doesn't kill the loop.
            print(f"\n❌ Error processing question: {e}")
            print("Try rephrasing or check your internet connection.")


if __name__ == "__main__":
    main()