"""
OCR: reads the text inside screenshots and flowcharts that are embedded in the Word manuals.

It is used only by ingest.py (once), never while a user is asking a question.
Results are cached in data/ocr_cache.json, so running ingest.py again is fast.

How it works for each image:
  1. skip tiny images (logos, icons)
  2. convert to grey + boost contrast + enlarge (small text is read much better when enlarged)
  3. run Tesseract and keep only words it is reasonably sure about (confidence >= 50)
  4. rebuild the text line by line in reading order
"""
import hashlib
import io
import json
import re
import shutil
from pathlib import Path

import config

OCR_VERSION = "v2"        # change this to force re-reading all images
MIN_SIDE = 150            # pixels; smaller images are logos/icons
MIN_CONF = 35             # Tesseract word confidence (0-100)
MAX_WORDS = 250           # per image

CACHE_FILE = config.DATA_DIR / "ocr_cache.json"
_cache = None
_ready = None


def tesseract_ready() -> bool:
    """Find the Tesseract program (Windows / Mac / Linux). Returns False if it is not installed."""
    global _ready
    if _ready is not None:
        return _ready
    try:
        import pytesseract
        from PIL import Image  # noqa: F401
    except ImportError:
        print("[OCR] pytesseract / Pillow not installed - screenshots will be skipped.")
        _ready = False
        return False

    candidates = [
        config.TESSERACT_CMD,
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/opt/homebrew/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/usr/bin/tesseract",
    ]
    for c in candidates:
        if c and Path(c).exists():
            pytesseract.pytesseract.tesseract_cmd = c
            _ready = True
            return True
    print("[OCR] Tesseract program not found - screenshots will be skipped.")
    print("      Install it (see the guide) or set TESSERACT_CMD in .env")
    _ready = False
    return False


def _load_cache():
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            _cache = {}
    return _cache


def _save_cache():
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(_cache, ensure_ascii=False), encoding="utf-8")


def _read_image(blob: bytes) -> str:
    import pytesseract
    from PIL import Image, ImageOps

    try:
        img = Image.open(io.BytesIO(blob))
        img.load()
    except Exception:
        return ""                                   # e.g. an unsupported format
    w, h = img.size
    if min(w, h) < MIN_SIDE:
        return ""
    img = ImageOps.autocontrast(img.convert("L"))
    scale = 3 if w < 1400 else 2 if w < 2000 else 1
    if scale > 1:
        img = img.resize((w * scale, h * scale), Image.LANCZOS)

    data = pytesseract.image_to_data(img, config="--psm 3", output_type=pytesseract.Output.DICT)
    lines = {}
    for i, word in enumerate(data["text"]):
        word = word.strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            continue
        if not word or conf < MIN_CONF or not re.search(r"[A-Za-z0-9]", word):
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)

    kept = [" ".join(ws) for _, ws in sorted(lines.items()) if any(len(x) >= 3 for x in ws)]
    return " ".join(" ".join(kept).split()[:MAX_WORDS])


def ocr_image_bytes(blob: bytes) -> str:
    """Return the text found in an image (cached)."""
    if not tesseract_ready():
        return ""
    cache = _load_cache()
    key = hashlib.sha1(blob).hexdigest() + OCR_VERSION
    if key not in cache:
        cache[key] = _read_image(blob)
        _save_cache()
    return cache[key]
