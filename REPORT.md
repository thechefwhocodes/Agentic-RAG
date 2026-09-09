# Report

## What I built

A local agentic RAG system that answers questions about Apple, Microsoft, and Alphabet by combining their `financials.db` SQLite database with their 10-K filings. A single agent runs a ReAct loop — call a tool, look at what came back, decide what to do next, repeat until there's enough to answer. It exposes `POST /api/chat` and a small one-page UI at `/` so you can actually ask it things instead of just reading raw JSON.

## How the system is structured

- `src/agent.py` — the ReAct loop itself. Two tools: `run_sql` and `search_filings`. There's a step cap (`MAX_AGENT_STEPS`) that works as a circuit breaker — if it's hit, the agent is forced to answer with whatever evidence it already has instead of just failing outright.
- `src/tools.py` — the tool implementations. SQL is locked to read-only twice over: a string check that only allows `SELECT`/`WITH`, and the connection itself opened with `mode=ro`. Tool errors come back as results instead of exceptions, so the model actually sees what went wrong and can fix its own query on the next turn.
- `src/ingest.py` — turns the PDFs into chunks and loads them into Chroma. Run once, offline (`python -m src.ingest`).
- `src/retrieval.py` — the semantic search over that Chroma collection.
- `src/models.py` — typed evidence objects (`SqlEvidence`, `FilingsEvidence`) and the API's request/response shapes, instead of loose dicts everywhere.
- `src/llm.py` — a thin client around the OpenAI-compatible Fireworks API. Every call tracks its own cost and latency, and there's a `response_model` param for pulling back schema-constrained JSON when I need it.
- `src/eval.py` — the eval harness, covered below.
- `static/index.html` — the UI: ask a question, get the answer, expand an evidence panel to see the exact SQL or filing chunks behind it.

## How I retrieve from SQL and PDFs

**SQL**: I put the full DDL straight into the system prompt, so the model always has the real schema in front of it instead of some hand-written description that can quietly go stale. I tested this directly and confirmed the model can write genuine multi-table joins when a question calls for one — asking for revenue and assets side by side across all three companies, for example. On other questions it just breaks the work into a few simple queries across multiple steps instead. Both get to the right answer; the second way is just slower.

**PDFs**: text comes out page by page via pymupdf. All three companies format their 10-K section headers differently — Apple puts the number and title on one line, Alphabet puts the title on the next line, and Microsoft repeats a running header on every single page that had to be stripped out before anything else would work. I ran a dry pass against all six filings before writing the real pipeline specifically to catch this, and got clean section detection across all 22 sections of every filing. Chunks are roughly 1500 characters, never cross a section boundary, and get a short prefix — company, ticker, fiscal year, section, page — before being embedded with `nomic-embed-text-v1.5` through Fireworks. That prefix was the single biggest lever on retrieval quality, more than anything I tried with chunk size or overlap.

## How I evaluate the system

`python -m src.eval` runs all 10 dev questions through the same agent that backs the API, scores each one with whatever method the answer key specifies (`fuzzy_numeric`, `exact_match_entity`, or `llm_judge`), and writes out `questions/dev_answers.json` plus a full `questions/eval_trace.json`. The trace has every tool call, every chunk retrieved, cost, and latency, so when something fails I can see why instead of just getting a pass/fail bit.

The `llm_judge` path scores three things, each 0 or 1, through a schema-constrained JSON response: correctness against the gold answer, groundedness against the evidence the agent actually retrieved (not the gold answer — this is what catches an answer that sounds right but isn't backed by anything it actually looked up), and completeness. It only passes if all three do.

Right now that's 10/10 on the public dev set, at roughly $0.002–0.004 and 1–10 seconds per query depending on how much work a question needs.

## Trade-offs I made and why

- I went with a single agent instead of multiple agents. There are only two tools here, and most of the harder questions need SQL and filing evidence reasoned about together in the same context — exactly the situation where splitting into separate agents tends to hurt more than help. A multi-agent setup would've cost several times the tokens for no real accuracy gain at this scale.

- I pinned temperature to 0 everywhere. That costs some natural variation in how answers are phrased, but it means running the eval twice gives me the same result twice, which matters more here than it would for an ordinary chat product.

- I also skipped BM25/hybrid retrieval for now. Embeddings alone are genuinely weak on exact numbers and tickers, but every numeric question in the dev set actually gets answered from SQL rather than the filings, so this gap isn't costing me anything yet.

- A few other things I deliberately skipped: a reranker (unnecessary at ~1,700 chunks — plain top-k retrieval is cheap enough here), and retry logic on transient Fireworks errors (hit one mid-build, but low-stakes enough as a local prototype to leave for later). The system prompt also stays deliberately bare — schema, tools, "cite your evidence," nothing else — even though I found real gaps I could have patched, like a fiscal-year mismatch and a stray hedging line in Alphabet's segment table. I didn't, so the eval reflects what the system actually knows rather than what I taught it about these ten questions.

## What I'd improve with more time

1. Stop the agent from repeating itself. I traced one dev question where it burned most of its step budget on near-duplicate filing searches before finally landing on a useful passage. Penalizing or skipping repeated near-identical searches would save both time and money.
2. Harden table extraction. PDF tables come out as loosely ordered text fragments right now. It's held up better than I expected — the model usually still connects the right label to the right number from context — but it's fragile, and it's the first thing I'd fix properly.
3. Add BM25 as a fallback next to the embedding search, mostly as insurance against the exact-number weakness above, once a real case actually needs it.

---

I used Claude Code throughout to help build this quickly — drafting code, running the dry runs and live tests behind the claims above, and putting this report together — but the architecture decisions, what to defer, and what went into this write-up were mine.
