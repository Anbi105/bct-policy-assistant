"""
The RAG engine:   question  ->  retrieve chunks  ->  ask the LLM  ->  answer (+ log)

Retrieval = hybrid search
    * BM25      : classic keyword scoring (great for exact things like "21st", "G5", "LOP")
    * Embeddings: meaning-based (great for "how do I fix a missing swipe?")
    The two rankings are merged with Reciprocal Rank Fusion (RRF).
"""
import json
import math
import re
import time
from collections import Counter
from datetime import datetime

import numpy as np
import requests

import config

# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
STOPWORDS = set(
    "a an the is are was were be been am do does did of to in on at for from by with and or if "
    "it its this that these those i me my we our you your can could should would will shall may "
    "what which who whom how when where why as into than then there their about any all".split()
)


def tokenize(text: str):
    """lowercase -> split into words -> drop stop-words -> very light stemming."""
    tokens = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in STOPWORDS:
            continue
        if len(w) > 3 and w.endswith("s"):   # hours -> hour
            w = w[:-1]
        tokens.append(w[:6])                 # regularize / regularization / regularise -> "regula"
    return tokens


# --------------------------------------------------------------------------- #
# BM25 keyword search (written by hand so you can explain it)
# --------------------------------------------------------------------------- #
class BM25:
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = docs_tokens
        self.n = len(docs_tokens)
        self.avg_len = sum(len(d) for d in docs_tokens) / max(1, self.n)
        self.tf = [Counter(d) for d in docs_tokens]
        df = Counter()
        for d in docs_tokens:
            df.update(set(d))
        # rare words get a high weight, common words a low one
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query_tokens):
        out = np.zeros(self.n)
        for i, tf in enumerate(self.tf):
            length = len(self.docs[i])
            s = 0.0
            for t in query_tokens:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * length / self.avg_len))
            out[i] = s
        return out


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #
class Retriever:
    def __init__(self):
        if not config.CHUNKS_FILE.exists():
            raise SystemExit("data/chunks.json not found. Run:  python ingest.py")
        data = json.loads(config.CHUNKS_FILE.read_text(encoding="utf-8"))
        self.chunks = data["chunks"]
        self.bm25 = BM25([tokenize(c["text"]) for c in self.chunks])
        self.emb = None
        self.embedder = None

        if config.USE_EMBEDDINGS and config.EMB_FILE.exists():
            matrix = np.load(config.EMB_FILE)
            if len(matrix) == len(self.chunks):
                try:
                    from fastembed import TextEmbedding
                    self.embedder = TextEmbedding(model_name=data.get("embed_model", config.EMBED_MODEL))
                    self.emb = matrix
                except Exception as e:
                    print(f"[warning] embeddings disabled ({e}); using keyword search only")
            else:
                print("[warning] embeddings.npy is out of date - run python ingest.py again")

    @property
    def mode(self):
        return "hybrid (BM25 + embeddings)" if self.emb is not None else "keyword (BM25) only"

    def warm_up(self):
        """Run one dummy search so the first real question is not slow."""
        self.search("warm up")

    def search(self, question: str, k: int = None):
        k = k or config.TOP_K
        bm = self.bm25.scores(tokenize(question))
        rankings = []
        if bm.max() > 0:
            rankings.append(np.argsort(-bm))

        vec = None
        if self.emb is not None:
            qv = np.array(list(self.embedder.query_embed([question]))[0], dtype="float32")
            qv /= np.linalg.norm(qv) + 1e-9
            vec = self.emb @ qv                      # cosine similarity (vectors are unit length)
            rankings.append(np.argsort(-vec))

        # Reciprocal Rank Fusion: score = sum( 1 / (60 + rank) ) over both rankings
        rrf = np.zeros(len(self.chunks))
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                rrf[idx] += 1.0 / (60 + rank + 1)

        top = np.argsort(-rrf)[:k]
        hits = []
        for i in top:
            c = self.chunks[int(i)]
            hits.append({
                "id": c["id"],
                "doc_title": c["doc_title"],
                "section": c["section"],
                "text": c["text"],
                "bm25": round(float(bm[i]), 3),
                "vector": round(float(vec[i]), 3) if vec is not None else None,
                "rrf": round(float(rrf[i]), 5),
            })
        return hits


_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
NOT_FOUND = "I couldn't find this in the policy documents."

SYSTEM_PROMPT = f"""You are the HR policy assistant for employees of Bahwan CyberTek (BCT).
Answer the question using ONLY the CONTEXT excerpts provided. Never use outside knowledge.

Rules:
- If the answer is not in the context, reply exactly: "{NOT_FOUND}"
- Be short and precise: 1-3 sentences, or a short list when the answer is a set of steps or items.
- Quote numbers, dates, grades and names exactly as written in the context.
- If different excerpts give different limits for different situations (for example a minimum
  and a maximum, or a manual and an email), state each one and say what it applies to.
- End with the source in this format: [Source: <document name>]
"""


def build_messages(question, hits):
    context = "\n\n".join(f"--- Excerpt {i + 1} ---\n{h['text']}" for i, h in enumerate(hits))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
    ]


# --------------------------------------------------------------------------- #
# LLM call (streaming)
# --------------------------------------------------------------------------- #
class LLMError(Exception):
    pass


def stream_chat(messages):
    """Yield the answer piece by piece (works for Groq, Gemini and Ollama)."""
    if config.PROVIDER != "ollama" and not config.LLM_API_KEY:
        raise LLMError(f"No API key found. Add {config.KEY_ENV_NAME}=... to your .env file.")

    url = config.LLM_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config.LLM_API_KEY:
        headers["Authorization"] = "Bearer " + config.LLM_API_KEY
    payload = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": config.MAX_ANSWER_TOKENS,
        "stream": True,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, stream=True, timeout=(10, 90))
    except requests.RequestException as e:
        raise LLMError(f"Could not reach the LLM at {url}: {e}")

    if r.status_code != 200:
        hint = {401: " (check your API key)", 404: " (check LLM_MODEL name)",
                429: " (rate limit - wait a few seconds)"}.get(r.status_code, "")
        raise LLMError(f"{config.PROVIDER} returned HTTP {r.status_code}{hint}: {r.text[:300]}")

    for raw in r.iter_lines(chunk_size=16):
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = obj.get("choices") or []
        if choices:
            piece = (choices[0].get("delta") or {}).get("content")
            if piece:
                yield piece


# --------------------------------------------------------------------------- #
# Logging  (one JSON line per question -> logs/queries.jsonl)
# --------------------------------------------------------------------------- #
def log_query(record: dict):
    config.LOG_DIR.mkdir(exist_ok=True)
    with open(config.QUERY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# The main function: ask()
# --------------------------------------------------------------------------- #
def ask(question: str, tag: str = "app"):
    """Generator of events: sources -> token... -> done (or error)."""
    retriever = get_retriever()
    t0 = time.perf_counter()
    hits = retriever.search(question)
    retrieval_s = time.perf_counter() - t0
    yield {"type": "sources", "sources": hits, "retrieval_ms": round(retrieval_s * 1000, 1)}

    pieces, first_token_s, error = [], None, None
    try:
        for piece in stream_chat(build_messages(question, hits)):
            if first_token_s is None:
                first_token_s = time.perf_counter() - t0
            pieces.append(piece)
            yield {"type": "token", "text": piece}
    except LLMError as e:
        error = str(e)
        yield {"type": "error", "message": error}

    total_s = time.perf_counter() - t0
    answer = "".join(pieces).strip()
    log_query({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tag": tag,
        "question": question,
        "retrieved": [{k: h[k] for k in ("id", "doc_title", "section", "bm25", "vector", "rrf", "text")}
                      for h in hits],
        "answer": answer,
        "error": error,
        "llm": f"{config.PROVIDER}:{config.LLM_MODEL}",
        "retrieval_ms": round(retrieval_s * 1000, 1),
        "first_token_s": round(first_token_s, 2) if first_token_s else None,
        "total_s": round(total_s, 2),
    })
    yield {"type": "done", "first_token_s": first_token_s, "total_s": total_s}


def answer_text(question: str, tag: str = "app"):
    """Non-streaming helper (used by eval.py)."""
    sources, pieces, error, total = [], [], None, None
    for ev in ask(question, tag=tag):
        if ev["type"] == "sources":
            sources = ev["sources"]
        elif ev["type"] == "token":
            pieces.append(ev["text"])
        elif ev["type"] == "error":
            error = ev["message"]
        elif ev["type"] == "done":
            total = ev["total_s"]
    return {"answer": "".join(pieces).strip(), "sources": sources, "error": error, "total_s": total}
