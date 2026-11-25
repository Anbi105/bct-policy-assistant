BCT Nexus: Company Policy Assistant (RAG)

An AI assistant that answers employee questions in plain language, using only the company's policy documents. It retrieves the relevant sections, passes them to an LLM, and replies with a short answer plus the source document. If the answer is not in the documents, it says so instead of guessing.

Example: "What is the payroll cut-off date every month?" → "The payroll cut-off date is the 21st of every month. [Source: Email 5 Day Work Week Effective 15 July 2026]"

Team: <Nexus> | Author: <Anbi Tibina K K>

Table of contents
Features
Architecture
Knowledge base
Tech stack
Project structure
Getting started
Usage
Evaluation
Logging
Design decisions
Limitations
Troubleshooting
Features
Retrieval-Augmented Generation (RAG): answers are grounded in the documents, not in the model's memory.
Hybrid search: BM25 keyword search combined with semantic embeddings, merged using Reciprocal Rank Fusion.
Source citations: every answer ends with the document it came from, and the UI shows the exact chunks retrieved.
"I don't know" behaviour: questions outside the documents get a clear "couldn't find this" reply.
OCR for screenshots: text inside screenshots and flowcharts in the Word manuals is extracted once during ingestion.
Streaming web UI: answers appear word by word in the browser, using Python's built-in web server.
Built-in evaluation: a 14-question test set with an automatic scorer.
Full query logging: question, retrieved chunks with scores, final answer and timings for every request.
Swappable LLM provider: Groq (default), Gemini, or a local Ollama model.
Architecture
Question time
Ingestion (run once)
docs/: 3 .docx + 1 .eml
Convert to text + OCRimages
Split into section chunks
Embed chunks
chunks.json +embeddings.npy
User question
Hybrid search: BM25 +embeddings, merged withRRF
Top 4 chunks
LLM with strict prompt
Answer + source
logs/queries.jsonl

The slow work (reading files, OCR, embedding) happens once in ingest.py. At question time, search takes milliseconds and the hosted LLM streams the answer.

Knowledge base
Document	Format
5 Day Work Week, effective 15 July 2026	.eml (converted to text)
Regularization User Manual	.docx
Timecard User Manual	.docx
Claims User Manual	.docx

After ingestion the documents become 73 chunks (Timecard 29, Regularization 24, Claims 13, Email 7). The converted text of every document is saved in data/text/ so the conversion can be inspected.

Tech stack
Area	Choice
Language	Python 3.10 to 3.12
Document parsing	python-docx, Python email module
OCR	Tesseract via pytesseract and Pillow
Embeddings	fastembed with BAAI/bge-small-en-v1.5
Keyword search	BM25 (implemented in rag.py)
Vector store	NumPy array (the corpus is small, so no database is needed)
LLM	Groq llama-3.3-70b-versatile (default), temperature 0
Server / UI	http.server with a single-page HTML UI
Project structure
BCT_Nexus/
├── docs/                  # Knowledge base documents (.docx, .eml)
├── data/
│   ├── text/              # Plain-text version of every document
│   ├── chunks.json        # Generated chunks
│   └── ocr_cache.json     # Cached OCR output
├── logs/
│   ├── queries.jsonl      # One log line per question
│   └── eval_results.json  # Latest evaluation results
├── ui/index.html          # Web interface
├── config.py              # All settings (reads .env)
├── ingest.py              # Documents -> text -> chunks -> embeddings
├── ocr.py                 # Screenshot text extraction
├── convert_email.py       # Standalone .eml -> .txt conversion
├── rag.py                 # Retrieval, prompt, LLM call, logging
├── server.py              # Web server
├── eval.py                # Evaluation runner
├── evals.json             # Test questions with expected answers
├── requirements.txt
├── .env.example           # Template for your settings
└── README.md
Getting started
Prerequisites
Python 3.10 to 3.12
Tesseract OCR (only needed if you re-run OCR; the OCR results are already cached in data/ocr_cache.json)
A free Groq API key
Installation
bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

Create a virtual environment and install dependencies.

Windows (PowerShell)

powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env

macOS / Linux

bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
Configuration

Open .env and set your key (no quotes, no spaces):

env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_key_here

.env is listed in .gitignore and must never be committed.

Variable	Default	Purpose
LLM_PROVIDER	groq	groq, gemini or ollama
LLM_MODEL	provider default	Override the model name
TOP_K	4	Number of chunks sent to the LLM
USE_EMBEDDINGS	1	Set to 0 for keyword-only search
USE_OCR	1	Set to 0 to skip screenshot OCR
PORT	8000	Web server port
Build the knowledge base
bash
python ingest.py

This converts the documents to text, splits them into chunks and creates embeddings. The first run downloads a small embedding model (about 130 MB). Re-run it whenever a document changes.

Usage
Start the assistant
bash
python server.py

Open http://127.0.0.1:8000. Type a question or click an example. Under each answer, Sources used shows the retrieved chunks and the timings.

Run the evaluation
bash
python eval.py                    # retrieval + LLM answers
python eval.py --retrieval-only   # search only, no API key needed
Evaluation

evals.json contains 14 questions whose correct answers were taken directly from the documents: 12 answerable questions and 2 that are deliberately not in the documents (the correct behaviour is to say "I couldn't find this").

For each question, eval.py checks two things separately:

Retrieval hit: did the retrieved chunks contain the facts needed?
Answer correct: does the final answer contain the expected facts (or, for out-of-scope questions, say it couldn't find the answer)?

Separating the two shows whether a wrong answer is a search problem or a generation problem.

Result: 14 / 14 answers correct, 12 / 12 retrieval hits (see logs/eval_results.json).

#	Question	Expected	Result
1	How many hours of office clock-in must an associate complete each day?	9 hours	Pass
2	What is the payroll cut-off date every month?	21st	Pass
3	Who is eligible for the five-day work week?	Grade G5 and above	Pass
4	What happens to days not regularized and with no leave applied?	Loss of Pay (LOP)	Pass
5	Who must approve exceptions to the five-day work week?	Reporting Manager, Business Head, HRBP	Pass
6	How should I mark attendance when working from a client site?	ON-SITE	Pass
7	Within how many days should a regularization request be performed?	15 days	Pass
8	What is the minimum biometric attendance needed to apply for regularization?	More than 7 hours	Pass
9	When should I use the Missed to Swipe option?	Either in-time or out-time missed	Pass
10	For India associates, what is the maximum normal hours in a timecard per day?	9 hours	Pass
11	Who do I contact if the task I need is not in the timecard dropdown?	RMG	Pass
12	What happens to a claim after the manager approves it?	Goes to Finance	Pass
13	How many casual leaves do I get in a year?	Not in documents	Pass (says not found)
14	What is the maternity leave policy?	Not in documents	Pass (says not found)
Logging

Every question, from the UI or the evals, appends one JSON line to logs/queries.jsonl:

json
{
  "timestamp": "2026-11-22T13:55:18",
  "tag": "app",
  "question": "What is the payroll cut-off date?",
  "retrieved": [
    {"id": "...", "doc_title": "...", "section": "...", "bm25": 12.36, "vector": 0.79, "rrf": 0.0328, "text": "..."}
  ],
  "answer": "...",
  "llm": "groq:llama-3.3-70b-versatile",
  "retrieval_ms": 4.1,
  "first_token_s": 0.52,
  "total_s": 1.1
}

This records the question, which parts of the documents were retrieved (with scores), and the final answer. The tag field separates UI questions (app) from evaluation runs (eval).

Design decisions

Chunking by section. The manuals have headings such as "B. Pre-requisites" and the email has numbered guidelines such as "5. Salary Impact". ingest.py detects these headings and creates one chunk per section (roughly 25 to 180 words). Longer sections are split with one line of overlap. Every chunk starts with [Document | Section], so both the search and the LLM know where the text came from. Small, topic-focused chunks match questions more precisely than whole pages.

Hybrid retrieval. BM25 is strong on exact terms such as "21st", "G5" and "LOP". Embeddings handle different wording of the same idea. The two rankings are merged with Reciprocal Rank Fusion, where each chunk scores the sum of 1 / (60 + rank) across both rankings. The top 4 chunks go to the LLM, so related facts from different documents can reach the model together.

Grounded prompt. The system prompt tells the model to use only the supplied excerpts, to reply with a fixed "couldn't find" sentence when the answer is missing, to quote numbers exactly, to state each limit separately when sources differ, and to cite the source. Temperature is 0 for repeatable answers.

No vector database. With about 73 chunks, a NumPy array is enough and keeps the system easy to understand.

Ingestion separated from querying. OCR and embedding run once, so answering a question stays fast.

Limitations
Heading detection uses simple rules. A document with different formatting may need a rule adjustment.
OCR on flowcharts is imperfect: words can come out in a jumbled order or be missed. OCR text is stored as separate chunks labelled Screenshots (OCR) to keep it away from the main text.
The "C. Workflow" section of the Regularization manual is empty in the supplied file.
Leave plan and holiday policy documents were not part of the supplied knowledge base, so those questions cannot be answered.
The assistant has no conversation memory: each question is answered independently.