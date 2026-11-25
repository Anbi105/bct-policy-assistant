"""
STEP 1 - Build the knowledge base (run once, or whenever documents change).

    python ingest.py

What it does
  1. Reads every file in docs/  (.docx, .eml, .txt)
  2. Converts each one to plain text and saves it in data/text/   (this is the
     "convert the email to text" step from the assessment)
  3. Splits the text into small chunks, one per section of the document
  4. (optional) turns every chunk into an embedding vector
  5. Saves chunks.json (+ embeddings.npy) so the app can start instantly

Nothing slow (OCR, embedding) ever happens while a user is asking a question.
Screenshots and flowcharts inside the Word files are read by OCR (ocr.py) - no text is typed by hand.
"""
import email
import html
import json
import re
import sys
from email import policy
from pathlib import Path

import config
import ocr

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# --------------------------------------------------------------------------- #
# 1. File -> plain text
# --------------------------------------------------------------------------- #
BLIP = "{http://schemas.openxmlformats.org/drawingml/2006/main}blip"
EMBED = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
OCR_STATS = {"images": 0, "with_text": 0}


def docx_to_text(path: Path) -> str:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(str(path))
    out = []
    # Walk the body in order so paragraphs and tables stay where they were.
    for child in d.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            text = Paragraph(child, d).text.strip()
            if text:
                out.append(text)
            if config.USE_OCR:                       # screenshots / flowcharts in this paragraph
                for blip in child.iter(BLIP):
                    rid = blip.get(EMBED)
                    if rid and rid in d.part.related_parts:
                        OCR_STATS["images"] += 1
                        shot = ocr.ocr_image_bytes(d.part.related_parts[rid].blob)
                        if shot:
                            OCR_STATS["with_text"] += 1
                            out.append("[Screenshot] " + shot)
        elif tag == "tbl":
            for row in Table(child, d).rows:
                cells = []
                for cell in row.cells:
                    t = cell.text.strip()
                    if t and (not cells or cells[-1] != t):  # skip merged-cell repeats
                        cells.append(t)
                if cells:
                    out.append("Document information: " + " | ".join(cells))
    return "\n".join(out)


def eml_to_text(path: Path) -> str:
    with open(path, "rb") as f:
        msg = email.message_from_binary_file(f, policy=policy.default)
    body = msg.get_body(preferencelist=("plain", "html"))
    text = body.get_content() if body else ""
    if body is not None and body.get_content_type() == "text/html":
        text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    text = text.split("Follow us on social media")[0]  # drop the footer / links
    for sign_off in ("Thank you for your continued", "Warm Regards", "Team HR"):
        text = "\n".join(l for l in text.splitlines() if not l.strip().startswith(sign_off))

    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t ]+", " ", line).strip()
        line = re.sub(r"^\*\s+", "- ", line)           # bullets
        if line:
            lines.append(line)

    header = [
        "## Email details",
        f"Subject: {msg['subject']}",
        f"From: {msg['from']}",
        f"Date: {msg['date']}",
    ]
    return "\n".join(header + lines)


def file_to_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".docx":
        return docx_to_text(path)
    if ext == ".eml":
        return eml_to_text(path)
    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8")
    raise ValueError(f"Unsupported file type: {path.name}")


def pretty_title(path: Path) -> str:
    name = path.stem.replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip()
    if path.suffix.lower() == ".eml" and not name.lower().startswith("email"):
        name = "Email " + name
    return name


# --------------------------------------------------------------------------- #
# 2. Text -> chunks   (one chunk per section, long sections are split further)
# --------------------------------------------------------------------------- #
MAJOR_PATTERNS = [
    re.compile(r"^##\s+(.+)$"),                 # our own markers
    re.compile(r"^[A-Z]\.\s+\S.{0,55}$"),       # "A. Preface:"  "C. Workflow:"
    re.compile(r"^\d+\.\s+\S.{0,45}$"),         # "1. Eligibility"  "2. Work from home:"
]
SUB_PATTERN = re.compile(r"^[^.!?]{2,60}\s?:$")    # "Navigation:"  "Entering a Claim:"
SKIP_LINE = re.compile(r"^(\.|Pic\d+:.*|Contents|Table of Contents)$", re.I)
NOT_HEADINGS = {"note:"}


def classify(line: str):
    """Return ('major'|'sub'|None, title)."""
    if line.lower() in NOT_HEADINGS or line.startswith("[Screenshot]"):
        return None, ""
    for pat in MAJOR_PATTERNS:
        m = pat.match(line)
        if m and len(line.split()) <= 9:
            title = m.group(1) if pat.pattern.startswith("^##") else line
            return "major", title.strip().rstrip(": ")
    if SUB_PATTERN.match(line) and len(line.split()) <= 8:
        return "sub", line.rstrip(": ")
    return None, ""


def chunk_document(source: str, doc_title: str, text: str):
    prefix = re.sub(r"[^a-z0-9]+", "-", source.lower())[:30].strip("-")
    parent, sub = None, None
    body = []
    shots = []               # OCR text of screenshots -> separate chunks, so they never pollute the main text
    meta_lines = []          # table rows (version, author...) -> their own chunk
    chunks = []

    def section_name():
        if parent and sub:
            return f"{parent} > {sub}"
        return parent or sub or "Introduction"

    def emit(lines, section):
        words = " ".join(lines).split()
        if len(words) < 3:
            return
        n = len(chunks) + 1
        chunks.append({
            "id": f"{prefix}-{n:02d}",
            "source": source,
            "doc_title": doc_title,
            "section": section,
            "text": f"[{doc_title} | {section}]\n" + "\n".join(lines),
        })

    def emit_split(lines, section):
        piece, count = [], 0
        for line in lines:
            w = len(line.split())
            if piece and count + w > config.MAX_WORDS_PER_CHUNK:
                emit(piece, section)
                piece, count = [piece[-1]], len(piece[-1].split())   # 1-line overlap
            piece.append(line)
            count += w
        emit(piece, section)

    def flush():
        section = section_name()
        if body:
            emit_split(body, section)
            body.clear()
        if shots:
            emit_split(shots, section + " > Screenshots (OCR)")
            shots.clear()

    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line or SKIP_LINE.match(line):
            continue
        if line.startswith("Document information: "):
            meta_lines.append(line[len("Document information: "):])
            continue
        if line.startswith("[Screenshot]"):
            shots.append(line)
            continue
        kind, title = classify(line)
        if kind == "major":
            flush()
            parent, sub = title, None
        elif kind == "sub":
            flush()
            sub = title
        else:
            body.append(line)
    flush()
    emit(meta_lines, "Document information")
    return chunks


# --------------------------------------------------------------------------- #
# 3. Embeddings (optional but recommended)
# --------------------------------------------------------------------------- #
def embed_chunks(chunks):
    import numpy as np
    from fastembed import TextEmbedding

    print(f"Loading embedding model '{config.EMBED_MODEL}' (first run downloads ~130 MB)...")
    model = TextEmbedding(model_name=config.EMBED_MODEL)
    vectors = np.array(list(model.embed([c["text"] for c in chunks])), dtype="float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9   # unit length
    return vectors


def main():
    files = sorted(p for p in config.DOCS_DIR.iterdir()
                   if p.suffix.lower() in (".docx", ".eml", ".txt", ".md"))
    if not files:
        raise SystemExit(f"No documents found in {config.DOCS_DIR}")

    config.TEXT_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    if config.USE_OCR and any(p.suffix.lower() == ".docx" for p in files):
        print("Reading screenshots with OCR (first run takes a few minutes, then it is cached)...")
    for path in files:
        OCR_STATS.update(images=0, with_text=0)
        text = file_to_text(path)
        (config.TEXT_DIR / (path.stem + ".txt")).write_text(text, encoding="utf-8")
        chunks = chunk_document(path.stem, pretty_title(path), text)
        all_chunks.extend(chunks)
        words = len(text.split())
        extra = f"  (OCR: {OCR_STATS['with_text']}/{OCR_STATS['images']} images had text)" if OCR_STATS["images"] else ""
        print(f"  {path.name:50s} {words:5d} words -> {len(chunks):2d} chunks{extra}")

    meta = {"embed_model": config.EMBED_MODEL, "chunks": all_chunks}

    if config.EMB_FILE.exists():
        config.EMB_FILE.unlink()
    if config.USE_EMBEDDINGS:
        try:
            import numpy as np
            vectors = embed_chunks(all_chunks)
            np.save(config.EMB_FILE, vectors)
            print(f"Saved embeddings {vectors.shape} -> {config.EMB_FILE.name}")
        except Exception as e:  # keep going with keyword search only
            print(f"\n[warning] Could not create embeddings ({e}).")
            print("          The app will still work using keyword (BM25) search only.\n")
    else:
        print("USE_EMBEDDINGS=0 -> skipping embeddings (keyword search only).")

    config.CHUNKS_FILE.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone. {len(all_chunks)} chunks saved to {config.CHUNKS_FILE}")
    print("Text versions of every document are in data/text/ - open them to check the conversion.")


if __name__ == "__main__":
    main()
