"""
Text chunker for systematic review documents.

Splits input text into semantically meaningful chunks by:
1. Section headers (e.g., "Methods", "Results", "Discussion")
2. Paragraph boundaries
3. Sentence-level fallback for very long paragraphs

Each chunk carries metadata: section_name, chunk_index, char_offset_start/end.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A single text chunk with provenance."""
    text: str
    index: int                      # sequential chunk ID
    section: str = ""               # detected section name
    char_start: int = 0             # offset in original document
    char_end: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        """Rough token count (~4 chars per token for English)."""
        return len(self.text) // 4


# Section header patterns common in systematic reviews
_SECTION_RE = re.compile(
    r"^(?:#+\s*)?("
    r"abstract|introduction|background|methods?|methodology|"
    r"study selection|search strategy|data extraction|"
    r"results?|findings|"
    r"gut microbiome|functional|metabolic|"
    r"disease phenotype|statistical|"
    r"discussion|limitations|"
    r"conclusion|summary|"
    r"references?|bibliography|"
    r"risk of bias|quality assessment|reporting bias|certainty"
    r")(?:\s.*)?$",
    re.IGNORECASE | re.MULTILINE,
)

# Sentence boundary (simplified but handles abbreviations reasonably)
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def chunk_document(
    text: str,
    max_chunk_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    """
    Chunk a systematic review document.

    Strategy:
    1. Split by section headers first
    2. Within sections, split by paragraph (double newline)
    3. If a paragraph exceeds max_chunk_tokens, split by sentence
    4. Add overlap between consecutive chunks for context continuity

    Returns list of Chunks with original char offsets preserved.
    """
    if not text or not text.strip():
        return []

    # Phase 1: Split into sections
    sections = _split_sections(text)

    # Phase 2: Split each section into paragraph-level chunks
    raw_chunks: list[tuple[str, str, int]] = []  # (text, section, char_start)
    for section_name, section_text, section_start in sections:
        paragraphs = _split_paragraphs(section_text, section_start)
        for para_text, para_start in paragraphs:
            if not para_text.strip():
                continue
            raw_chunks.append((para_text.strip(), section_name, para_start))

    # Phase 3: Enforce max size — split large chunks by sentence
    sized_chunks: list[tuple[str, str, int]] = []
    for chunk_text, section, start in raw_chunks:
        estimated_tokens = len(chunk_text) // 4
        if estimated_tokens <= max_chunk_tokens:
            sized_chunks.append((chunk_text, section, start))
        else:
            # Split by sentence
            sub_chunks = _split_by_sentence(chunk_text, max_chunk_tokens, start)
            for sub_text, sub_start in sub_chunks:
                sized_chunks.append((sub_text, section, sub_start))

    # Phase 4: Build Chunk objects with overlap
    chunks: list[Chunk] = []
    for i, (chunk_text, section, char_start) in enumerate(sized_chunks):
        # Add overlap from previous chunk's tail
        overlap_text = ""
        if i > 0 and overlap_tokens > 0:
            prev_text = sized_chunks[i - 1][0]
            overlap_chars = overlap_tokens * 4  # approximate
            if len(prev_text) > overlap_chars:
                # Find a sentence boundary near the overlap point
                tail = prev_text[-overlap_chars:]
                sent_break = tail.find(". ")
                if sent_break >= 0:
                    overlap_text = tail[sent_break + 2:]
                else:
                    overlap_text = tail

        full_text = (overlap_text + " " + chunk_text).strip() if overlap_text else chunk_text

        chunks.append(Chunk(
            text=full_text,
            index=i,
            section=section,
            char_start=char_start,
            char_end=char_start + len(chunk_text),
            metadata={
                "has_overlap": bool(overlap_text),
                "original_length": len(chunk_text),
            },
        ))

    return chunks


def _split_sections(text: str) -> list[tuple[str, str, int]]:
    """Split text into (section_name, section_text, char_offset) tuples."""
    matches = list(_SECTION_RE.finditer(text))

    if not matches:
        return [("full_document", text, 0)]

    sections = []
    # Content before first section header
    if matches[0].start() > 0:
        pre = text[:matches[0].start()]
        if pre.strip():
            sections.append(("preamble", pre, 0))

    for i, m in enumerate(matches):
        section_name = m.group(1).strip().lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end]
        if section_text.strip():
            sections.append((section_name, section_text, start))

    return sections


def _split_paragraphs(text: str, base_offset: int) -> list[tuple[str, int]]:
    """Split text by paragraph breaks (double newline)."""
    parts = re.split(r"\n\s*\n", text)
    paragraphs = []
    pos = 0
    for part in parts:
        # Find actual position in original text
        idx = text.find(part, pos)
        if idx < 0:
            idx = pos
        paragraphs.append((part, base_offset + idx))
        pos = idx + len(part)
    return paragraphs


def _split_by_sentence(
    text: str,
    max_tokens: int,
    base_offset: int,
) -> list[tuple[str, int]]:
    """Split a long paragraph into sentence groups that fit within max_tokens."""
    sentences = _SENT_RE.split(text)
    chunks = []
    current = ""
    current_start = base_offset

    for sent in sentences:
        candidate = (current + " " + sent).strip() if current else sent
        if len(candidate) // 4 > max_tokens and current:
            chunks.append((current.strip(), current_start))
            current = sent
            current_start = base_offset + text.find(sent, current_start - base_offset)
        else:
            current = candidate

    if current.strip():
        chunks.append((current.strip(), current_start))

    return chunks


# ── Agent-specific retrieval queries ──────────────────────────────────────

AGENT_QUERIES = {
    "ARGUS": [
        "claims with supporting and refuting evidence",
        "one-sided arguments or missing counter-evidence",
        "conclusions and findings with causal language",
        "certainty language and definitive claims",
        "framing and persuasive language",
    ],
    "LIBRA": [
        "hedging language: may, might, could, possibly, suggests",
        "certainty boosters: clearly, definitely, proves, establishes",
        "scope claims about populations and generalizability",
    ],
    "LENS": [
        "citation contexts with references [1] [2] etc",
        "claims that cite specific studies as evidence",
        "argumentation structure: premises and conclusions",
    ],
    "QUILL": [
        "passages with identified bias that need revision",
        "text with loaded or non-neutral language",
    ],
    "VIGIL": [
        "numerical data, statistics, and measurements",
        "citation references and specific factual claims",
    ],
}
