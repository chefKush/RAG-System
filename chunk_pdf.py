# chunk_pdf.py
# v2: Each chunk now carries metadata - page number and section guess

import re # Regular expressions for simple section heading detection
from pypdf import PdfReader #type: ignore  # The PDF text extraction library

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
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
# JOB 1: Extract from the PDF
# ---------------------------------------------------------------------------
def extract_pages_from_pdf(pdf_path: str) -> list[dict]:
    """
    Returns a list like:
        [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]

    The key difference from v1: we DON'T join all pages into one string.
    We keep them separate so each chunk can later be labeled with its page.
    """
    print(f"Opening PDF: {pdf_path}")

    # PdfReader parses the PDF structure and gives us access to each page.
    reader = PdfReader(pdf_path)

    # reader.pages is a list-like object; len() gives total page count.
    print(f"Total pages : {len(reader.pages)}")

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():  # Check if we got any non-whitespace text
            pages.append({"page": page_number, "text": text})
            print(f"  Page {page_number}: extracted {len(text)} characters")


    return pages


# Common section names found in research papers, manuals, reports.
# Each entry is a regex pattern that will match a section header line.
SECTION_PATTERNS = [
    # Allow optional leading numbers like "1 Introduction" or "3.2 Methods".
    # The pattern (?:\d+(?:\.\d+)?\s+)? means:
    #   - optional digit(s),
    #   - optional ".digit(s)" for subsections,
    #   - followed by whitespace.
    r"^(?:\d+(?:\.\d+)?\s+)?abstract$",
    r"^(?:\d+(?:\.\d+)?\s+)?introduction$",
    r"^(?:\d+(?:\.\d+)?\s+)?background$",
    r"^(?:\d+(?:\.\d+)?\s+)?related work$",
    r"^(?:\d+(?:\.\d+)?\s+)?methods?$",
    r"^(?:\d+(?:\.\d+)?\s+)?methodology$",
    r"^(?:\d+(?:\.\d+)?\s+)?model architecture$",
    r"^(?:\d+(?:\.\d+)?\s+)?architecture$",
    r"^(?:\d+(?:\.\d+)?\s+)?experiments?$",
    r"^(?:\d+(?:\.\d+)?\s+)?results?$",
    r"^(?:\d+(?:\.\d+)?\s+)?evaluation$",
    r"^(?:\d+(?:\.\d+)?\s+)?training$",
    r"^(?:\d+(?:\.\d+)?\s+)?discussion$",
    r"^(?:\d+(?:\.\d+)?\s+)?conclusion$",
    r"^(?:\d+(?:\.\d+)?\s+)?why self-attention$",
    r"^(?:\d+(?:\.\d+)?\s+)?references?$",
    r"^(?:\d+(?:\.\d+)?\s+)?bibliography$",
    r"^(?:\d+(?:\.\d+)?\s+)?acknowledgements?$",
    r"^(?:\d+(?:\.\d+)?\s+)?appendix$",
]
# Pre-compile all patterns into one combined regex. re.IGNORECASE lets
# "Methods" / "METHODS" / "methods" all match the same pattern.
SECTION_REGEX = re.compile("|".join(SECTION_PATTERNS), re.IGNORECASE)


def guess_section(text_so_far: str) -> str:
    """
    Look at all the text we've seen up to a point in the document and find
    the most recent line that looks like a section header.
    Returns "body" if no header has appeared yet.
    """
    # Split the text into lines and walk backwards from the end.
    # The first header we find while walking backwards is the most recent.
    lines = text_so_far.split("\n")
    for line in reversed(lines):
        stripped = line.strip().lower()

        # A section header is usually short (<50 chars) and matches one of
        # our patterns. The length check prevents false matches like
        # "we present a new method for..." being tagged as "method".
        if 0 < len(stripped) < 50 and SECTION_REGEX.match(stripped):
            return stripped

    return "body"


# ---------------------------------------------------------------------------
def chunk_with_metadata(pages: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    """
    Slide a window across the document, attaching page + section to each chunk.
    Returns: list of dicts like:
      {"text": "...", "page": 3, "section": "introduction", "chunk_index": 7}
    """
    print(f"\nChunking with metadata (size={chunk_size}, overlap={overlap})...")

    # Step 1: Build one big text stream + remember where each page starts.
    # We need the joined text for chunking, but we ALSO need a way to map
    # "character position 4823" back to "this is page 3" later.
    full_text_parts = []
    page_boundaries = []      # list of (char_position_where_page_starts, page_number)
    char_position = 0

    for page in pages:
        page_boundaries.append((char_position, page["page"]))
        full_text_parts.append(page["text"])
        char_position += len(page["text"]) + 1   # +1 for the "\n" we add when joining

    full_text = "\n".join(full_text_parts)

    # Step 2: Helper that converts a character position to a page number.
    def char_pos_to_page(pos: int) -> int:
        page = page_boundaries[0][1]
        for boundary_pos, page_num in page_boundaries:
            if pos >= boundary_pos:
                page = page_num
            else:
                break
        return page

    # Step 3: The chunking loop, same as v1 but now attaching metadata.
    chunks = []
    start = 0
    chunk_index = 0
    text_length = len(full_text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text = full_text[start:end].strip()

        if chunk_text:
            page = char_pos_to_page(start)              # which page did this chunk start on?
            section = guess_section(full_text[:start])  # what section came before this chunk?

            chunks.append({
                "text": chunk_text,
                "page": page,
                "section": section,
                "chunk_index": chunk_index,
            })
            chunk_index += 1

        start += chunk_size - overlap

    print(f"Total chunks: {len(chunks)}")

    # Show section distribution so you can sanity-check the labels look reasonable.
    section_counts = {}
    for c in chunks:
        section_counts[c["section"]] = section_counts.get(c["section"], 0) + 1

    print("\nSection distribution:")
    for section, count in sorted(section_counts.items(), key=lambda x: -x[1]):
        print(f"  {section:25s} → {count} chunks")

    return chunks


# Optional: quick standalone test.
if __name__ == "__main__":
    pages = extract_pages_from_pdf(PDF_PATH)
    chunks = chunk_with_metadata(pages, CHUNK_SIZE, CHUNK_OVERLAP)

    print("\n" + "=" * 70)
    print("SAMPLE CHUNKS WITH METADATA")
    print("=" * 70)
    for i in [0, len(chunks) // 4, len(chunks) // 2, -1]:
        c = chunks[i]
        print(f"\n--- Chunk {c['chunk_index']} | page={c['page']} | section={c['section']} ---")
        print(c["text"][:200] + "...")