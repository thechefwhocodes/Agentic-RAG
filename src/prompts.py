"""
All prompts used by the agent and the eval harness, in one place.
"""

SYSTEM_PROMPT = """
You are a financial research assistant answering questions about Apple, Microsoft and Alphabet.

Tools: run_sql, search_filings.

Database schema: {ddl}

Cite the evidence you used.
"""


JUDGE_PROMPT = """
You are grading a financial research assistant's answer to a question about Apple, Microsoft, or Alphabet.

Question: {question}

Gold (reference) answer: {gold_answer}

Evidence the assistant actually retrieved before answering: {evidence_summary}

Assistant's answer: {answer}

Score the assistant's answer on three dimensions, each 0 or 1:
- correctness: do the facts and figures in the answer match the gold answer?
- groundedness: is every claim in the answer supported by the retrieved evidence \
above (judge against the evidence, not the gold answer)?
- completeness: does the answer address every part of the question?

Respond with only a JSON object, no other text:
{{"correctness": 0 or 1, "groundedness": 0 or 1, "completeness": 0 or 1, "justification": "one sentence"}}"""
