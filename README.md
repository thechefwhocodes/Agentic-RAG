# Fireworks AI Applied MLE Take-Home: Agentic RAG for 10-K Analysis

This take-home is meant to mirror part of the Applied Machine Learning Engineer role: supporting customers in their journey to build GenAI applications on Fireworks.

In this exercise, you should approach the problem like a Fireworks engineer supporting a customer who needs an agentic RAG workflow over structured financial data and 10-K filings.

## What We're Looking For

1. Customer-oriented problem solving: translate the customer's needs into a practical system design.
2. Agent and tool design: decide when to query SQL, search PDFs, or combine both.
3. Evaluation discipline: show how you measured quality and where the system still fails.
4. Practical trade-offs: explain choices around models, latency, cost, reliability, and complexity.
5. Communication: provide clear instructions, clear answers, and a concise technical report.



## Customer Scenario

**From:** Natalie Brooks [natalie.brooks@acmecorp.example.com](mailto:natalie.brooks@acmecorp.example.com)  
**To:** Solutions Team [solutions@fireworks.ai](mailto:solutions@fireworks.ai)  
**Subject:** Help Needed: Local Research Assistant for 10-K Analysis

Hi Fireworks team,

Our research team spends a lot of time reading annual reports, cross-checking management commentary against financial tables, and building simple comparisons across companies. We have a local dataset that combines structured financial data with the original 10-K filings, and we want a local AI assistant that can help analysts answer increasingly complex questions over that material.

Our current prototype can handle simple lookups, but it breaks down when a question requires planning, multiple retrieval steps, or combining narrative disclosures with structured financials. In particular, we need a system that can:

- decide when to query the SQLite database versus the filings
- gather evidence from the right sources
- answer questions that range from direct lookup to multi-step synthesis
- stay grounded in the provided documents and data

We are providing:

- six 10-K filings for Apple, Microsoft, and Alphabet across FY2024 and FY2025
- a local SQLite database with structured financial data
- a 10-question development set

We would like a local proof of concept that a reviewer can run on their machine and interact with directly.

Thanks,  
Natalie Brooks  
Director of Research Systems, Acme Corp

## Project Structure

- `data/`: generated or provided assignment data, including the SQLite DB and 10-K PDFs
- `questions/`: the development-set questions, public dev answer key, and the `dev_answers.json` example template
- `scripts/`: helper scripts that can fetch the SEC source data, render PDFs, and build `financials.db`
- `starter/`: lightweight starter dependencies for setup and experimentation
- `setup.sh`: end-to-end local setup script



## What You Receive

- `questions/dev_questions.json`: 10 development-set questions
- `questions/dev_questions_with_answers.json`: the public dev-set answer key
- `questions/dev_answers_example.json`: template for your `dev_answers.json`
- `setup.sh`: local bootstrap script
- `starter/requirements.txt`: setup and starter dependencies
- `scripts/`: scripts that can fetch or rebuild the data if it is not already present

If `data/financials.db` and the 10-K PDFs are already present, `setup.sh` will reuse them. It only fetches SEC data if it needs to rebuild missing assets.

The dev-set answer key is public so you can evaluate your system locally. We intentionally do not provide an evaluation harness; part of the assignment is deciding how to measure correctness against the provided questions, answers, and data. Fireworks keeps a separate held-out set for the hidden final evaluation.

## Data Overview

The SQLite database includes these tables:

- `companies`: company metadata
- `income_statements`: revenue, gross profit, operating income, net income, EPS, and R&D
- `balance_sheets`: assets, liabilities, equity, cash, debt, and current balance metrics
- `segment_revenue`: revenue by business segment
- `geographic_revenue`: revenue by geography

The filings provide the narrative context needed for questions about risks, strategy, segment definitions, geographic commentary, and management discussion.

## Your Task

Build a local agentic RAG system that can answer increasingly complex questions about the provided companies and filings.

Your system should:

- run locally on a reviewer's machine
- support interactive use (e.g., with a simple UI)
- expose an HTTP API at `http://localhost:8000/api/chat` that accepts `POST` requests with `{"question": "..."}` and returns the answer either as JSON with a top-level `answer` or `content` field (for example, `{"answer": "..."}`) or as an SSE stream with an `answer` event whose data is `{"content": "..."}`.
- route questions to the right source or sources
- return grounded answers that make it easy to inspect evidence
- handle both straightforward retrieval and multi-step reasoning

---

## Running This Submission

1. `./setup.sh` — sets up the venv and the provided data (skips anything already present).
2. `uv pip install -r requirements.txt` — installs this submission's own dependencies (FastAPI, Chroma, etc.) into that venv.
3. `export FIREWORKS_API_KEY=<your-key>`
4. `python -m src.ingest` — builds the PDF search index (one-time, ~1-2 min, small Fireworks embedding cost). The index isn't shipped in this zip, so this step is required before filing-based questions will work.
5. `uvicorn src.api:app --port 8000`
6. Open `http://localhost:8000` in a browser to ask questions interactively, or `POST http://localhost:8000/api/chat` with `{"question": "..."}`.

To reproduce the dev-set evaluation: `python -m src.eval` (writes `questions/dev_answers.json` and `questions/eval_trace.json`).

**Required environment variable:** `FIREWORKS_API_KEY`. See [REPORT.md](REPORT.md) for system design, evaluation, and trade-offs.

---

