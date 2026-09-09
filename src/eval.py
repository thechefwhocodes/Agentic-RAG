"""Run the dev question set through the agent and score each answer with the
method the answer key specifies. Writes dev_answers.json and a full trace.

Run as: python -m src.eval
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from src.agent import Agent, Answer
from src.llm import LLM
from src.models import Evidence, SqlEvidence
from src.prompts import JUDGE_PROMPT

QUESTIONS_PATH = Path("questions/dev_questions_with_answers.json")
DEV_ANSWERS_PATH = Path("questions/dev_answers.json")
TRACE_PATH = Path("questions/eval_trace.json")

NUMBER_RE = re.compile(
    r"(-?[\d,]+(?:\.\d+)?)\s*(trillion|billion|million|thousand|%)?", re.IGNORECASE
)
UNIT_MULTIPLIER = {"trillion": 1e12, "billion": 1e9, "million": 1e6, "thousand": 1e3}
NUMERIC_TOLERANCE = 0.02  # 2% relative tolerance for fuzzy_numeric

COMPANY_ALIASES = {
    "AAPL": ("apple", "aapl"),
    "MSFT": ("microsoft", "msft"),
    "GOOGL": ("alphabet", "googl", "google"),
}


@dataclass
class EvalResult:
    method: str
    passed: bool
    reason: str
    correctness: int | None = None
    groundedness: int | None = None
    completeness: int | None = None
    judge_cost_usd: float = 0.0


def extract_numbers(text: str) -> list[float]:
    values = []
    for num_str, unit in NUMBER_RE.findall(text):
        try:
            value = float(num_str.replace(",", ""))
        except ValueError:
            continue
        if unit.lower() in UNIT_MULTIPLIER:
            value *= UNIT_MULTIPLIER[unit.lower()]
        values.append(value)
    return values


def score_fuzzy_numeric(answer_text: str, gold_numeric: float) -> EvalResult:
    found = extract_numbers(answer_text)
    for value in found:
        if gold_numeric == 0:
            if value == 0:
                return EvalResult("fuzzy_numeric", True, f"found {value}, matches gold 0")
            continue
        if abs(value - gold_numeric) / abs(gold_numeric) <= NUMERIC_TOLERANCE:
            return EvalResult(
                "fuzzy_numeric", True,
                f"found {value}, within {NUMERIC_TOLERANCE:.0%} of gold {gold_numeric}",
            )
    return EvalResult(
        "fuzzy_numeric", False,
        f"expected ~{gold_numeric}, found no matching number among {found[:5]}",
    )


def score_exact_match_entity(answer_text: str, gold_answer: str) -> EvalResult:
    """The gold answer names exactly one company; check the agent's answer
    names that same company."""
    gold_lower, answer_lower = gold_answer.lower(), answer_text.lower()
    for ticker, aliases in COMPANY_ALIASES.items():
        if any(alias in gold_lower for alias in aliases):
            if any(alias in answer_lower for alias in aliases):
                return EvalResult("exact_match_entity", True, f"answer names {ticker}, matching gold")
            return EvalResult("exact_match_entity", False, f"gold names {ticker}, not found in answer")
    return EvalResult("exact_match_entity", False, "could not identify expected company in gold answer")


def summarize_evidence(evidence: list[Evidence]) -> str:
    lines = []
    for e in evidence:
        if isinstance(e, SqlEvidence):
            if e.error:
                lines.append(f"SQL (error): {e.sql} -> {e.error}")
            else:
                lines.append(f"SQL: {e.sql} -> {e.rows}")
        else:
            for h in e.hits:
                lines.append(
                    f"Filing [{h.ticker} FY{h.fiscal_year} {h.section} p{h.page}]: {h.text[:300]}"
                )
    return "\n".join(lines) if lines else "(no evidence retrieved)"


def score_llm_judge(
    llm: LLM, question: str, gold_answer: str, answer_text: str, evidence: list[Evidence]
) -> EvalResult:
    prompt = JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        evidence_summary=summarize_evidence(evidence),
        answer=answer_text,
    )
    result = llm.chat([{"role": "user", "content": prompt}])
    match = re.search(r"\{.*\}", result.message.content or "", re.DOTALL)
    if not match:
        return EvalResult("llm_judge", False, "judge did not return JSON", judge_cost_usd=result.cost_usd)
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return EvalResult("llm_judge", False, "judge JSON unparseable", judge_cost_usd=result.cost_usd)

    passed = bool(parsed.get("correctness") and parsed.get("groundedness") and parsed.get("completeness"))
    return EvalResult(
        method="llm_judge",
        passed=passed,
        reason=parsed.get("justification", ""),
        correctness=parsed.get("correctness"),
        groundedness=parsed.get("groundedness"),
        completeness=parsed.get("completeness"),
        judge_cost_usd=result.cost_usd,
    )


def score_question(llm: LLM, q: dict, answer: Answer) -> EvalResult:
    method = q["evaluation"]
    if method == "fuzzy_numeric":
        return score_fuzzy_numeric(answer.text, q["gold_answer_numeric"])
    if method == "exact_match_entity":
        return score_exact_match_entity(answer.text, q["gold_answer"])
    if method == "llm_judge":
        return score_llm_judge(llm, q["question"], q["gold_answer"], answer.text, answer.evidence)
    raise ValueError(f"Unknown evaluation method: {method}")


def main() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text())
    llm = LLM()
    agent = Agent(llm=llm)

    results, dev_answers = [], {}
    total_agent_latency, total_agent_cost, total_judge_cost = 0.0, 0.0, 0.0

    for q in questions:
        answer = agent.ask(q["question"])
        score = score_question(llm, q, answer)

        total_agent_latency += answer.latency_s
        total_agent_cost += answer.cost_usd
        total_judge_cost += score.judge_cost_usd

        dev_answers[q["id"]] = answer.text
        results.append({
            "id": q["id"], "tier": q["tier"], "question": q["question"],
            "answer": answer.text, "gold_answer": q["gold_answer"],
            "latency_s": answer.latency_s, "cost_usd": answer.cost_usd,
            "hit_step_cap": answer.hit_step_cap,
            "evidence": [e.model_dump() for e in answer.evidence],
            **asdict(score),
        })

        status = "PASS" if score.passed else "FAIL"
        reason_suffix = "" if score.passed else f"  - {score.reason}"
        print(f"{q['id']:8s} {score.method:20s} {status}{reason_suffix}")

    DEV_ANSWERS_PATH.write_text(json.dumps(dev_answers, indent=2))
    TRACE_PATH.write_text(json.dumps(results, indent=2, default=str))

    n = len(questions)
    n_passed = sum(1 for r in results if r["passed"])
    print("-" * 50)
    print(f"accuracy    : {n_passed}/{n} ({round(100 * n_passed / n, 1)}%)")
    print(f"agent cost  : ${round(total_agent_cost, 5)} total, ${round(total_agent_cost / n, 6)}/query avg")
    print(f"agent lat   : {round(total_agent_latency / n, 2)}s/query avg")
    print(f"judge cost  : ${round(total_judge_cost, 5)} total (evaluation-only overhead)")
    step_cap_hits = [r["id"] for r in results if r["hit_step_cap"]]
    if step_cap_hits:
        print(f"step cap hit: {step_cap_hits}")
    failed = [(r["id"], r["reason"]) for r in results if not r["passed"]]
    if failed:
        print("missed      :")
        for qid, reason in failed:
            print(f"  {qid}: {reason}")
    print(f"wrote {DEV_ANSWERS_PATH} and {TRACE_PATH}")


if __name__ == "__main__":
    main()
