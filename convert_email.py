"""
OPTIONAL - convert the .eml email into a plain .txt file.

The assessment says: "copy the email content, convert it into text, use it as a document".
ingest.py already does this automatically for every file in docs/.
Run this script only if you want to show that step on its own:

    python convert_email.py

It saves the text to data/text/<email name>.txt and prints it.
"""
import sys
from pathlib import Path

from ingest import eml_to_text

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

docs = Path(__file__).parent / "docs"
emails = sorted(docs.glob("*.eml"))
if not emails:
    raise SystemExit("No .eml file found in the docs folder.")

for src in emails:
    out = Path(__file__).parent / "data" / "text" / (src.stem + ".txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    text = eml_to_text(src)
    out.write_text(text, encoding="utf-8")
    print(f"Saved: {out}\n")
    print(text)
