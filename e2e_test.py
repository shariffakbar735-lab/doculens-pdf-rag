"""
e2e_test.py - Live end-to-end test using real Ollama and ChromaDB.
Run with: .venv/Scripts/python e2e_test.py
"""

import io
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

import pymupdf as fitz
from pdf_processor import extract_pages, chunk_pages, process_pdf
import rag

FILENAME = "_e2e_test_report.pdf"

def make_pdf() -> bytes:
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 80),  "Annual Report 2024", fontsize=16)
    p1.insert_text((50, 120), "Total revenue for fiscal year 2024 was 4.7 billion dollars.", fontsize=11)
    p1.insert_text((50, 145), "Revenue grew by 23 percent compared to the previous year.", fontsize=11)
    p2 = doc.new_page()
    p2.insert_text((50, 80),  "Operating Expenses", fontsize=14)
    p2.insert_text((50, 120), "Research and development spending increased to 800 million dollars.", fontsize=11)
    p2.insert_text((50, 145), "The company hired 2500 new employees during 2024.", fontsize=11)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def run():
    failures = []

    # ── STEP 1: PDF extraction ─────────────────────────────────────────────
    print("\n=== Step 1: Create and process synthetic PDF ===")
    pdf_bytes = make_pdf()
    print(f"  PDF bytes: {len(pdf_bytes)}")
    # Use chunk_size=100 so even a short test PDF produces chunks spanning both pages
    pages = extract_pages(pdf_bytes, FILENAME)
    chunks = chunk_pages(pages, FILENAME, chunk_size=100, chunk_overlap=10)
    print(f"  Chunks extracted: {len(chunks)}")
    for c in chunks:
        print(f"    chunk {c.chunk_id[:8]}  page={c.page_number}  {c.text[:60]!r}")

    # Page numbers must be 1-based and span both pages
    page_nums = {c.page_number for c in chunks}
    if 1 not in page_nums:
        failures.append("FAIL: page 1 not found in chunks")
    else:
        print("  PASS: page 1 present")
    if 2 not in page_nums:
        failures.append("FAIL: page 2 not found in chunks")
    else:
        print("  PASS: page 2 present")

    # ── STEP 2: Index ─────────────────────────────────────────────────────
    print("\n=== Step 2: Index chunks into ChromaDB via Ollama embeddings ===")
    # Clean up any leftover from a previous run
    rag.delete_document(FILENAME)

    added = rag.index_chunks(chunks)
    print(f"  Chunks added: {added}")
    if added != len(chunks):
        failures.append(f"FAIL: expected {len(chunks)} chunks added, got {added}")
    else:
        print("  PASS: all chunks indexed")

    # ── STEP 3: Idempotency – re-index should add 0 ───────────────────────
    print("\n=== Step 3: Re-indexing same chunks (idempotency check) ===")
    added2 = rag.index_chunks(chunks)
    print(f"  Re-index added: {added2}")
    if added2 != 0:
        failures.append(f"FAIL: expected 0 on re-index, got {added2}")
    else:
        print("  PASS: idempotent (0 new chunks)")

    # ── STEP 4: Document list ─────────────────────────────────────────────
    print("\n=== Step 4: Document list ===")
    docs = rag.list_documents()
    print(f"  Indexed docs: {docs}")
    if FILENAME not in docs:
        failures.append("FAIL: document not in list after indexing")
    else:
        print(f"  PASS: {FILENAME!r} appears in document list")

    # ── STEP 5: Question WITH evidence ───────────────────────────────────
    print("\n=== Step 5: Answer question WITH supporting evidence ===")
    result = rag.answer("What was the total revenue in 2024?")
    print(f"  retrieved={result['retrieved']}  sources={len(result['sources'])}")
    for src in result["sources"]:
        print(f"    {src['filename']}, page {src['page_number']}, dist={src['distance']:.3f}")
    print(f"  Answer (first 300 chars): {result['answer'][:300]}")
    if not result["retrieved"]:
        failures.append("FAIL: retrieved=False for question with evidence")
    else:
        print("  PASS: answer returned with sources")

    # Check that at least one source is from the right file
    file_srcs = [s for s in result["sources"] if s["filename"] == FILENAME]
    if not file_srcs:
        failures.append("FAIL: no source matched the indexed filename")
    else:
        print(f"  PASS: {len(file_srcs)} source(s) from {FILENAME!r}")

    # ── STEP 6: Question with NO evidence ────────────────────────────────
    print("\n=== Step 6: Answer question with no supporting evidence ===")
    result2 = rag.answer("What is the airspeed velocity of an unladen swallow?")
    print(f"  Answer (first 300 chars): {result2['answer'][:300]}")
    # App retrieved something (top-K always returns if docs exist), but
    # the LLM should say it doesn't have information or give a caveat.
    # We only assert it doesn't crash and returns a string.
    if not isinstance(result2["answer"], str) or not result2["answer"].strip():
        failures.append("FAIL: empty answer for out-of-scope question")
    else:
        print("  PASS: handled safely without crash")

    # ── STEP 7: Invalid PDF ───────────────────────────────────────────────
    print("\n=== Step 7: Invalid PDF raises ValueError ===")
    try:
        process_pdf(b"this is not a pdf", "bad.pdf")
        failures.append("FAIL: no error raised for invalid PDF bytes")
    except ValueError as e:
        print(f"  PASS: ValueError raised: {e}")

    # ── STEP 8: Delete document ───────────────────────────────────────────
    print("\n=== Step 8: Delete document ===")
    deleted = rag.delete_document(FILENAME)
    print(f"  Deleted {deleted} chunks")
    if deleted != added:
        failures.append(f"FAIL: deleted {deleted} but expected {added}")
    else:
        print("  PASS: correct number of chunks deleted")

    docs_after = rag.list_documents()
    if FILENAME in docs_after:
        failures.append("FAIL: document still present after delete")
    else:
        print(f"  PASS: {FILENAME!r} no longer in document list")

    # ── Result summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("ALL 8 LIVE END-TO-END CHECKS PASSED")


if __name__ == "__main__":
    run()
