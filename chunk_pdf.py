# chunk_pdf.py
# Purpose: Load a PDF, extract its text, and split it into overlapping chunks
# that are the right size for embedding and retrieval.

from pypdf import PdfReader #type: ignore  # The PDF text extraction library

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
# Path to the PDF in our project folder. Change "document.pdf" to your filename.
PDF_PATH = "document.pdf"

# Chunk size and overlap, measured in CHARACTERS (not words or tokens).
# Why characters? It's the simplest, most predictable unit. A more advanced
# project would chunk by tokens, but characters are perfect for learning.
#
# 1000 chars ≈ 200 words ≈ 250 tokens — a comfortable paragraph-sized chunk.
# 150 chars ≈ 30 words overlap — about one sentence, enough to preserve context.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# ---------------------------------------------------------------------------
# JOB 1: Extract all text from the PDF
# ---------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Open a PDF file and return all its text as one big string.
    Pages are joined with a newline so the boundary is preserved
    in case we want to inspect it later.
    """
    print(f"Opening PDF: {pdf_path}")

    # PdfReader parses the PDF structure and gives us access to each page.
    reader = PdfReader(pdf_path)

    # reader.pages is a list-like object; len() gives total page count.
    num_pages = len(reader.pages)
    print(f"Total pages found: {num_pages}")

    # We'll collect each page's text in this list, then join at the end.
    # Building a list and joining once is faster than concatenating strings
    # in a loop — a small but real Python performance habit.
    all_text_parts = []

    for page_number, page in enumerate(reader.pages, start=1):
        # extract_text() pulls out the text content from a single page.
        # It can return None for pages that are blank or contain only images.
        page_text = page.extract_text()

        if page_text:
            all_text_parts.append(page_text)
            # Visibility: how much did this page contribute?
            print(f"  Page {page_number}: extracted {len(page_text)} characters")
        else:
            print(f"  Page {page_number}: no extractable text (image-only?)")

    # Join all pages into one big string separated by newlines.
    full_text = "\n".join(all_text_parts)

    print(f"\nTotal characters extracted: {len(full_text)}")
    return full_text


# ---------------------------------------------------------------------------
# JOB 2: Split the text into overlapping chunks
# ---------------------------------------------------------------------------
def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Cut the text into overlapping windows.

    Imagine a window of size `chunk_size` sliding across the text.
    After each chunk, the window doesn't jump forward by its full width —
    it backs up by `overlap` characters so the next chunk starts inside
    the previous one. That overlapping region is the "context bridge."
    """
    print(f"\nSplitting text into chunks (size={chunk_size}, overlap={overlap})...")

    chunks = []          # List that will hold each chunk string
    start = 0            # Current window's start position in the text
    text_length = len(text)

    # Keep sliding the window until we've gone past the end of the text.
    while start < text_length:
        # End of this window. min() prevents going past the end of the string
        # for the final chunk, which may be shorter than chunk_size.
        end = min(start + chunk_size, text_length)

        # Slice out this chunk and clean up surrounding whitespace.
        chunk = text[start:end].strip()

        # Skip any chunks that turned out empty (can happen with weird PDFs
        # full of whitespace). A chunk of pure spaces is useless to embed.
        if chunk:
            chunks.append(chunk)

        # Move the window forward by (chunk_size - overlap).
        # Example: size=1000, overlap=150 → window jumps forward 850 chars.
        # The next chunk's first 150 chars are the previous chunk's last 150.
        start += chunk_size - overlap

    print(f"Total chunks created: {len(chunks)}")
    return chunks


# ---------------------------------------------------------------------------
# RUN IT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Step 1: PDF → text
    full_text = extract_text_from_pdf(PDF_PATH)

    # Step 2: text → chunks
    chunks = split_into_chunks(full_text, CHUNK_SIZE, CHUNK_OVERLAP)

    # Step 3: Visibility — show what we built so you can SEE the chunks.
    # We don't print all of them (could be hundreds), just the first 3
    # and the last one, plus some stats.
    print("\n" + "=" * 70)
    print("PREVIEW OF CHUNKS")
    print("=" * 70)

    for i in range(min(3, len(chunks))):
        print(f"\n--- Chunk {i} (length: {len(chunks[i])} chars) ---")
        print(chunks[i][:300] + ("..." if len(chunks[i]) > 300 else ""))

    if len(chunks) > 3:
        print(f"\n--- Chunk {len(chunks) - 1} (last chunk, length: {len(chunks[-1])} chars) ---")
        print((chunks[-1])[:300] + ("..." if len(chunks[-1]) > 300 else ""))

    # Step 4: Confirm overlap visually.
    # The last 50 chars of chunk[0] should equal the first 50 chars
    # of chunk[1] (approximately — strip() may have shifted things slightly).
    if len(chunks) >= 2:
        print("\n" + "=" * 70)
        print("OVERLAP CHECK")
        print("=" * 70)
        print(f"End of chunk 0:   ...{chunks[0][-80:]!r}")
        print(f"Start of chunk 1: {chunks[1][:80]!r}...")
        print("(These should share roughly the same words.)")