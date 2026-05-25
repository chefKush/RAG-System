# embed_chunks.py
# Purpose: Take the chunks from Stage 2 and turn each one into a 384-dim
# vector that represents its meaning. Then prove the embeddings actually
# capture meaning by comparing some sentences.

from sentence_transformers import SentenceTransformer  # The embedding library #type: ignore
import numpy as np                                     # For math on vectors

# Import the functions we wrote in Stage 2 so we don't duplicate code.
# Python treats chunk_pdf.py as a module we can pull from.
from chunk_pdf import extract_text_from_pdf, split_into_chunks

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
PDF_PATH = "document.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# The embedding model. "all-MiniLM-L6-v2" is the standard small/fast choice.
# - "all": trained on diverse data (good general-purpose)
# - "MiniLM": a compressed BERT variant — small but strong
# - "L6": 6 transformer layers (small, fast)
# - "v2": improved training version
# Output dimension: 384.
MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Step 1: Load the embedding model
# ---------------------------------------------------------------------------
# The FIRST time this line runs, it downloads ~90 MB of model weights
# from HuggingFace and caches them on disk (in ~/.cache/huggingface/).
# Every run after that, loading is instant — it just reads from disk.
print(f"Loading embedding model: {MODEL_NAME}")
print("(First run downloads ~90 MB. Subsequent runs are instant.)")

model = SentenceTransformer(MODEL_NAME)

# Quick visibility: show the model's output dimension. This number is critical
# because the vector database in Stage 4 must be configured to expect it.
embedding_dim = model.get_sentence_embedding_dimension()
print(f"Model loaded. Embedding dimension: {embedding_dim}")


# ---------------------------------------------------------------------------
# Step 2: Sanity check — prove embeddings actually capture meaning
# ---------------------------------------------------------------------------
# Before we embed our entire PDF, let's prove the model works as advertised.
# We'll embed three sentences and compare them. If embeddings are real,
# the two similar ones should score much closer to each other than to the
# unrelated one.

print("\n" + "=" * 70)
print("SANITY CHECK: Do embeddings really capture meaning?")
print("=" * 70)

test_sentences = [
    "Cars are useful for transportation.",
    "Automobiles help people get around.",      # Similar meaning to #1
    "Bananas are a yellow tropical fruit.",     # Unrelated
]

# model.encode() converts a list of strings into a 2D numpy array.
# Shape: (number_of_sentences, embedding_dim) → here (3, 384).
test_embeddings = model.encode(test_sentences)
print(f"\nShape of embeddings array: {test_embeddings.shape}")
print("(3 sentences × 384 dimensions each)")


def cosine_similarity(vec_a, vec_b):
    """
    Cosine similarity = dot product divided by the product of magnitudes.
    Result is between -1 and 1. For embeddings, usually 0 to 1.
    Higher = more similar in meaning.

    This is the math the vector database will do for us in Stage 4.
    We're doing it manually here just so you SEE it.
    """
    dot_product = np.dot(vec_a, vec_b)
    magnitude_a = np.linalg.norm(vec_a)
    magnitude_b = np.linalg.norm(vec_b)
    return dot_product / (magnitude_a * magnitude_b)


# Compare each pair
sim_cars_autos = cosine_similarity(test_embeddings[0], test_embeddings[1])
sim_cars_bananas = cosine_similarity(test_embeddings[0], test_embeddings[2])
sim_autos_bananas = cosine_similarity(test_embeddings[1], test_embeddings[2])

print(f"\nSimilarity ('cars' vs 'automobiles'):  {sim_cars_autos:.4f}  ← should be HIGH")
print(f"Similarity ('cars' vs 'bananas'):      {sim_cars_bananas:.4f}  ← should be LOW")
print(f"Similarity ('autos' vs 'bananas'):     {sim_autos_bananas:.4f}  ← should be LOW")
print("\nIf the first number is much higher than the other two, embeddings work. ✅")


# ---------------------------------------------------------------------------
# Step 3: Embed all the chunks from our PDF
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("EMBEDDING YOUR PDF CHUNKS")
print("=" * 70)

# Re-use Stage 2's functions to get our chunks.
full_text = extract_text_from_pdf(PDF_PATH)
chunks = split_into_chunks(full_text, CHUNK_SIZE, CHUNK_OVERLAP)

print(f"\nEmbedding {len(chunks)} chunks...")
print("(On CPU this takes a few seconds for ~50 chunks.)")

# show_progress_bar=True gives a nice live progress bar while encoding.
# encode() processes the chunks in batches efficiently under the hood.
chunk_embeddings = model.encode(chunks, show_progress_bar=True)

print(f"\nDone! Shape of result: {chunk_embeddings.shape}")
print(f"({len(chunks)} chunks × {embedding_dim} dimensions each)")


# ---------------------------------------------------------------------------
# Step 4: Visibility — peek inside one embedding
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PEEK INSIDE AN EMBEDDING")
print("=" * 70)
print(f"\nChunk 0 text (first 200 chars):")
print(f"  {chunks[0][:200]}...")
print(f"\nChunk 0 embedding (first 10 of {embedding_dim} numbers):")
print(f"  {chunk_embeddings[0][:10]}")
print(f"\nChunk 0 embedding statistics:")
print(f"  min value:  {chunk_embeddings[0].min():.4f}")
print(f"  max value:  {chunk_embeddings[0].max():.4f}")
print(f"  mean value: {chunk_embeddings[0].mean():.4f}")


# ---------------------------------------------------------------------------
# Step 5: Mini retrieval demo — find the chunk most similar to a question
# ---------------------------------------------------------------------------
# This is a preview of what Stage 4's vector database will do automatically.
# We'll do it by hand here so you SEE the mechanism.

print("\n" + "=" * 70)
print("MINI RETRIEVAL DEMO (manual, no database yet)")
print("=" * 70)

# Make up a question. You can change this to something relevant to YOUR PDF.
question = "What is this document about?"
print(f"\nQuestion: {question}")

# Embed the question using the SAME model — this is crucial.
# Question embeddings and chunk embeddings must live in the same vector space,
# which only works if they were produced by the same model.
question_embedding = model.encode(question)
print(f"Question embedding shape: {question_embedding.shape}")

# Compute similarity between the question and every chunk.
similarities = []
for i, chunk_vec in enumerate(chunk_embeddings):
    score = cosine_similarity(question_embedding, chunk_vec)
    similarities.append((i, score))

# Sort by score, highest first.
similarities.sort(key=lambda pair: pair[1], reverse=True)

# Show the top 3 most relevant chunks.
print("\nTop 3 most relevant chunks for the question:")
for rank, (chunk_index, score) in enumerate(similarities[:3], start=1):
    print(f"\n--- Rank {rank} | Chunk #{chunk_index} | similarity: {score:.4f} ---")
    print(chunks[chunk_index][:250] + "...")