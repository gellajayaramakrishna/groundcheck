"""Text helpers: sentence splitting (with character offsets), chunking, and turning
RAGTruth source records into plain text."""
import json
import re

# A sentence ends at . ! ? (optionally followed by quotes/brackets), then whitespace,
# then something that looks like the start of a new sentence.
_BOUNDARY = re.compile(r"([.!?]+[\"')\]]*)(\s+)(?=[\"'(\[]?[A-Z0-9])")
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc",
                  "e.g", "i.e", "u.s", "u.k", "inc", "ltd", "co", "no", "fig", "approx"}


def _split_line(text, offset):
    spans, start = [], 0
    for m in _BOUNDARY.finditer(text):
        words = text[start:m.start(1)].split()
        last = words[-1].lower().rstrip(".") if words else ""
        if last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha()):
            continue  # "Dr. Smith" or "J. K. Rowling" -> not a sentence end
        spans.append((start, m.end(1)))
        start = m.end()
    spans.append((start, len(text)))
    out = []
    for s, e in spans:
        piece = text[s:e]
        lead = len(piece) - len(piece.lstrip())
        piece = piece.strip()
        if len(piece) >= 3:
            out.append((offset + s + lead, offset + s + lead + len(piece), piece))
    return out


def split_sentences(text):
    """Return a list of (start, end, sentence) with offsets into the original text.
    Newlines are treated as hard boundaries (bullet points, paragraphs)."""
    result = []
    for m in re.finditer(r"[^\n]+", text):
        result.extend(_split_line(m.group(), m.start()))
    return result


def chunk_text(text, max_words=150, stride=120):
    """Split long text into overlapping windows of words, so each chunk fits the model."""
    words = text.split()
    if len(words) <= max_words:
        return [" ".join(words)] if words else []
    chunks = []
    for i in range(0, len(words), stride):
        chunks.append(" ".join(words[i:i + max_words]))
        if i + max_words >= len(words):
            break
    return chunks


def _flatten(value, prefix=""):
    lines = []
    if isinstance(value, dict):
        for k, v in value.items():
            lines.extend(_flatten(v, f"{prefix}{k}: " if not prefix else f"{prefix}{k} "))
    elif isinstance(value, list):
        for item in value:
            lines.extend(_flatten(item, prefix))
    elif value is not None:
        lines.append(f"{prefix}{value}")
    return lines


def source_to_text(source_info):
    """RAGTruth stores context in 3 shapes: plain article (Summary), question+passages (QA),
    or structured JSON about a business (Data2txt). Convert all of them to plain text."""
    if isinstance(source_info, str):
        return source_info
    if isinstance(source_info, dict) and "passages" in source_info:
        return f"Question: {source_info.get('question', '')}\n{source_info['passages']}"
    if isinstance(source_info, dict) and "review_info" in source_info:
        info = dict(source_info)
        reviews = info.pop("review_info") or []
        lines = _flatten(info)
        for r in reviews:
            lines.append(f"Review ({r.get('review_stars')} stars): {r.get('review_text', '')}")
        return "\n".join(lines)
    return json.dumps(source_info, ensure_ascii=False)


def overlaps(a_start, a_end, b_start, b_end):
    """True if two character ranges [a_start, a_end) and [b_start, b_end) share any character."""
    return a_start < b_end and b_start < a_end
