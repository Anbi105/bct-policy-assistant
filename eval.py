"""
STEP 3 - Measure how well the assistant works.

    python eval.py                      full test: retrieval + LLM answer
    python eval.py --retrieval-only     only checks the search part (no API key needed)

For every question in evals.json we check two things:
  * Retrieval hit : did the retrieved chunks contain the facts needed for the answer?
  * Answer correct: does the final answer contain the expected facts?
                    (for questions NOT in the documents, the answer must say "couldn't find")
Results are printed and saved to logs/eval_results.json
"""
import argparse
import json
import sys
import time

import config
import rag

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PAUSE_BETWEEN_QUESTIONS = 6     # seconds, avoids the Groq free-tier rate limit (HTTP 429)
RETRY_WAIT = 20                 # seconds to wait when a 429 happens
MAX_ATTEMPTS = 4


def norm(s: str) -> str:
    """Lower-case and replace the special characters some models write
    (non-breaking spaces, special hyphens, curly quotes, bold marks)."""
    s = s.lower()
    for bad, good in {"\u202f": " ", "\u00a0": " ", "\u2009": " ", "\u2011": "-",
                      "\u2010": "-", "\u2013": "-", "\u2019": "'", "**": ""}.items():
        s = s.replace(bad, good)
    return " ".join(s.split())


def has_all(text: str, groups) -> bool:
    """Every group must be satisfied by at least one of its alternatives."""
    low = norm(text)
    return all(any(norm(alt) in low for alt in group) for group in groups)


def said_not_found(answer: str) -> bool:
    a = norm(answer)
    return "couldn't find" in a or "could not find" in a or "not found" in a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    args = ap.parse_args()

    items = json.loads((config.ROOT / "evals.json").read_text(encoding="utf-8"))
    retriever = rag.get_retriever()
    retriever.warm_up()
    print(f"Search mode: {retriever.mode}")
    if not args.retrieval_only:
        print(f"LLM: {config.PROVIDER} / {config.LLM_MODEL}")
    print()

    rows, ret_hits, ret_total, ans_ok = [], 0, 0, 0
    for n, it in enumerate(items, 1):
        q = it["question"]
        negative = it.get("expect_not_found", False)
        t0 = time.perf_counter()

        if args.retrieval_only:
            hits = retriever.search(q)
            answer, error = "", None
        else:
            for attempt in range(MAX_ATTEMPTS):
                res = rag.answer_text(q, tag="eval")
                if not (res["error"] and "429" in str(res["error"])):
                    break
                print(f"      (rate limit, waiting {RETRY_WAIT} s and trying again...)")
                time.sleep(RETRY_WAIT)
            hits, answer, error = res["sources"], res["answer"], res["error"]
            time.sleep(PAUSE_BETWEEN_QUESTIONS)
        secs = time.perf_counter() - t0

        retrieved_text = " ".join(h["text"] for h in hits)
        if negative:
            retrieval = None                                   # nothing to find
            correct = said_not_found(answer) if answer else None
        else:
            retrieval = has_all(retrieved_text, it["expect"])
            ret_total += 1
            ret_hits += retrieval
            correct = has_all(answer, it["expect"]) if answer else None
        if correct:
            ans_ok += 1

        fmt = lambda v: "n/a " if v is None else ("PASS" if v else "FAIL")
        print(f"{n:2d}. retrieval={fmt(retrieval)}  answer={fmt(correct)}  {secs:5.2f}s  {q}")
        if error:
            print(f"      ERROR: {error}")
        elif answer and not correct:
            print(f"      got: {answer[:160]}")
        rows.append({"question": q, "expected": it.get("expect") or "NOT FOUND",
                     "retrieval_hit": retrieval, "answer_correct": correct,
                     "answer": answer, "retrieved_ids": [h["id"] for h in hits],
                     "seconds": round(secs, 2), "error": error})

    print("\n" + "=" * 60)
    print(f"Retrieval hit : {ret_hits}/{ret_total} answerable questions")
    if not args.retrieval_only:
        print(f"Answers right : {ans_ok}/{len(items)}  ({100 * ans_ok / len(items):.0f}%)")
    config.LOG_DIR.mkdir(exist_ok=True)
    out = config.LOG_DIR / "eval_results.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Details saved to {out}")


if __name__ == "__main__":
    main()
