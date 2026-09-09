"""Ingest the 10-K PDFs into Chroma: clean -> detect sections -> reflow -> chunk -> embed.

Pipeline is running headers/footers differ per company,
and section headers appear in three different layouts (title on the
same line, on the next line, or absent from a footer-polluted document).

Run as: python -m src.ingest [--rebuild]
"""

import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import chromadb
import pymupdf

from src.config import CHROMA_DIR, DB_PATH, PDF_DIR, get_embed_model
from src.llm import LLM

TARGET_CHARS = 1500
OVERLAP_CHARS = 250
EMBED_BATCH_SIZE = 64

FILENAME_RE = re.compile(r"^([A-Z]+)_FY(\d{4})_(10-K)\.pdf$")
ITEM_NUM_RE = re.compile(r"^ITEM\s+(\d{1,2}[A-C]?)\s*[.\-–—:]?\s*(.*)$", re.IGNORECASE)
BARE_ITEM_RE = re.compile(r"^Item\s+\d{1,2}[A-C]?\s*$", re.IGNORECASE)
HEADER_LINE_MAX_CHARS = 170
KNOWN_TITLES = (
    "BUSINESS",
    "RISK FACTORS",
    "PROPERTIES",
    "LEGAL",
    "MINE SAFETY",
    "UNRESOLVED",
    "CYBERSECURITY",
    "MARKET FOR",
    "SELECTED",
    "MANAGEMENT",
    "QUANTITATIVE",
    "FINANCIAL STATEMENTS",
    "CHANGES IN",
    "CONTROLS",
    "OTHER INFORMATION",
    "DISCLOSURE",
    "DIRECTORS",
    "EXECUTIVE",
    "SECURITY OWNERSHIP",
    "CERTAIN RELATIONSHIPS",
    "PRINCIPAL ACCOUNT",
    "EXHIBIT",
    "FORM 10-K SUMMARY",
)


def company_names() -> dict[str, str]:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    rows = conn.execute("SELECT ticker, name FROM companies").fetchall()
    conn.close()
    return dict(rows)


def clean_lines(pdf_path: Path) -> list[tuple[int, str]]:
    """Return [(page_no, line)], with running headers/footers, page numbers,
    and nbsp noise stripped. A "running" line repeats near the top of >=30%
    of pages (Microsoft repeats its Item number on every page)."""
    doc = pymupdf.open(pdf_path)
    pages = [p.get_text() for p in doc]
    doc.close()

    counts = Counter()
    for text in pages:
        for line in {l.strip() for l in text.split("\n")[:4] if l.strip()}:
            counts[line] += 1
    repeating = {l for l, c in counts.items() if c >= max(2, int(len(pages) * 0.3))}

    out = []
    for pno, text in enumerate(pages, start=1):
        for raw in text.split("\n"):
            s = re.sub(r"[\xa0 ]+", " ", raw).strip()
            s = re.sub(r"\s{2,}", " ", s)
            if not s or s in repeating or BARE_ITEM_RE.match(s):
                continue
            if s.isdigit() and len(s) <= 3:
                continue
            if re.match(r"^[A-Za-z].{0,40}\|\s*20\d\d Form 10-K\s*\|\s*\d+$", s):
                continue  # Apple's per-page footer
            out.append((pno, s))
    return out


def detect_sections(lines: list[tuple[int, str]]) -> list[tuple[int, int, str, str]]:
    """Find Item-section starts. Title may be on the same line (Apple) or the
    next one (Alphabet). Titles are validated against KNOWN_TITLES to reject
    prose cross-references like "...described in Item 1A Risk Factors...".
    Where an item number appears twice (table of contents, then the real
    header), the later occurrence wins.

    Returns [(line_idx, page_no, "Item N", title)], sorted by line_idx.
    """
    hits = []
    for i, (pno, s) in enumerate(lines):
        if len(s) > HEADER_LINE_MAX_CHARS:
            continue
        m = ITEM_NUM_RE.match(s)
        if not m:
            continue
        num, tail = m.group(1), m.group(2).strip()
        if not tail and i + 1 < len(lines):
            nxt = lines[i + 1][1]
            if len(nxt) < HEADER_LINE_MAX_CHARS and nxt.upper().startswith(
                KNOWN_TITLES
            ):
                tail = nxt
        if not tail or not tail.upper().startswith(KNOWN_TITLES):
            continue
        hits.append((i, pno, f"Item {num.upper()}", tail))

    best: dict[str, tuple[int, int, str, str]] = {}
    for h in hits:
        best[h[2]] = (
            h  # later occurrence overwrites earlier -> real header beats TOC row
        )
    return sorted(best.values())


def reflow_by_section(
    lines: list[tuple[int, str]], sections: list[tuple[int, int, str, str]]
) -> list[dict]:
    """Join hard-wrapped lines into paragraphs, each tagged with its starting
    page and section. A paragraph never spans a section boundary."""
    section_of_line: dict[int, tuple[str, str]] = {}
    for k, (start_idx, _pno, num, title) in enumerate(sections):
        end_idx = sections[k + 1][0] if k + 1 < len(sections) else len(lines)
        for j in range(start_idx, end_idx):
            section_of_line[j] = (num, title)

    paragraphs, buf, buf_page, cur_section = [], [], None, None
    for idx, (pno, s) in enumerate(lines):
        section = section_of_line.get(idx, (None, "Front Matter"))
        if buf and section != cur_section:
            paragraphs.append(
                {"page": buf_page, "section": cur_section, "text": " ".join(buf)}
            )
            buf, buf_page = [], None
        cur_section = section
        if buf_page is None:
            buf_page = pno
        buf.append(s)
        if re.search(r'[.:;!?)"”]$', s) or len(s) < 60:
            paragraphs.append(
                {"page": buf_page, "section": cur_section, "text": " ".join(buf)}
            )
            buf, buf_page = [], None
    if buf:
        paragraphs.append(
            {"page": buf_page, "section": cur_section, "text": " ".join(buf)}
        )
    return paragraphs


def chunk_paragraphs(paragraphs: list[dict]) -> list[dict]:
    """Pack paragraphs into ~TARGET_CHARS chunks with overlap. A chunk never
    spans a section boundary; front matter / table-of-contents paragraphs
    (no section) are dropped as noise."""
    chunks: list[dict] = []
    buf_text, buf_page, buf_section = "", None, None
    for p in paragraphs:
        if p["section"][0] is None:
            continue
        same_section = p["section"] == buf_section
        if buf_text and (
            not same_section or len(buf_text) + len(p["text"]) + 1 > TARGET_CHARS
        ):
            chunks.append({"page": buf_page, "section": buf_section, "text": buf_text})
            carry = buf_text[-OVERLAP_CHARS:] if same_section else ""
            buf_text = (carry + " " + p["text"]).strip() if carry else p["text"]
            buf_page = p["page"]
        else:
            if not buf_text:
                buf_page = p["page"]
            buf_text = (buf_text + " " + p["text"]).strip() if buf_text else p["text"]
        buf_section = p["section"]
    if buf_text:
        chunks.append({"page": buf_page, "section": buf_section, "text": buf_text})
    return chunks


def build_records(
    pdf_path: Path, ticker: str, company: str, fiscal_year: int
) -> list[dict]:
    lines = clean_lines(pdf_path)
    sections = detect_sections(lines)
    paragraphs = reflow_by_section(lines, sections)
    chunks = chunk_paragraphs(paragraphs)

    records = []
    for i, c in enumerate(chunks):
        num, title = c["section"]
        section_label = f"{num}. {title.title()}"
        prefix = (
            f"{company} ({ticker}) FY{fiscal_year} 10-K, {section_label}, "
            f"page {c['page']}: "
        )
        records.append(
            {
                "id": f"{ticker}_{fiscal_year}_{i:04d}",
                "text": c["text"],
                "embed_text": prefix + c["text"],
                "metadata": {
                    "ticker": ticker,
                    "company": company,
                    "fiscal_year": fiscal_year,
                    "form_type": "10-K",
                    "section": section_label,
                    "page": c["page"],
                },
            }
        )
    return records


def ingest_all(rebuild: bool = False) -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    llm = LLM()
    embed_config = get_embed_model()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection_name = embed_config.collection_name()

    if rebuild:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(collection_name)

    if collection.count() > 0 and not rebuild:
        print(
            f"Collection '{collection_name}' already has {collection.count()} "
            "chunks. Pass --rebuild to re-ingest."
        )
        return

    names = company_names()
    total_chunks = 0
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        m = FILENAME_RE.match(pdf_path.name)
        if not m:
            print(f"Skipping unrecognized filename: {pdf_path.name}")
            continue
        ticker, fiscal_year, form_type = m.group(1), int(m.group(2)), m.group(3)
        company = names.get(ticker, ticker)

        records = build_records(pdf_path, ticker, company, fiscal_year)
        for start in range(0, len(records), EMBED_BATCH_SIZE):
            batch = records[start : start + EMBED_BATCH_SIZE]
            result = llm.embed([r["embed_text"] for r in batch], is_query=False)
            collection.add(
                ids=[r["id"] for r in batch],
                embeddings=result.vectors,
                documents=[r["text"] for r in batch],
                metadatas=[r["metadata"] for r in batch],
            )

        sections_found = len({r["metadata"]["section"] for r in records})
        print(f"{pdf_path.name}: {len(records)} chunks, {sections_found} sections")
        total_chunks += len(records)

    print(f"Done. {total_chunks} chunks in collection '{collection_name}'.")


if __name__ == "__main__":
    ingest_all(rebuild="--rebuild" in sys.argv)
