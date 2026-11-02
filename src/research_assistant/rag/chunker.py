"""Split paper text into overlapping chunks with stable boundaries."""

from __future__ import annotations


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    """Return whitespace-aligned chunks.

    Each chunk is at most ``size`` characters. The next chunk starts ``overlap``
    characters before the previous cut, snapped forward to a word boundary when
    one is nearby, so passages stay readable.
    """
    if size < 1:
        raise ValueError("chunk size must be >= 1")
    if overlap < 0 or overlap >= size:
        raise ValueError("chunk overlap must be >= 0 and smaller than size")

    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if len(cleaned) <= size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + size)
        if end < len(cleaned):
            split = cleaned.rfind(" ", start, end)
            if split > start + size // 2:
                end = split
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(cleaned):
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        else:
            space = cleaned.find(" ", next_start)
            if space != -1 and space < end:
                next_start = space + 1
        start = next_start
    return chunks
