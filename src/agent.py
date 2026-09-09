"""Single-agent ReAct loop: call a tool, observe, decide again, until the
model answers without calling a tool or the step cap is hit.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field

from pydantic import ValidationError

from src.config import DB_PATH, MAX_AGENT_STEPS
from src.llm import LLM
from src.models import Evidence
from src.prompts import SYSTEM_PROMPT
from src.tools import RunSqlTool, SearchFilingsTool, Tool

logger = logging.getLogger(__name__)


# Get the DDL of the database
def get_ddl(conn: sqlite3.Connection) -> str:
    rows = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence' "
        "AND sql IS NOT NULL ORDER BY name"
    ).fetchall()
    return "\n\n".join(r[0] for r in rows)


@dataclass
class Answer:
    text: str
    evidence: list[Evidence] = field(default_factory=list)
    latency_s: float = 0.0
    cost_usd: float = 0.0
    steps: int = 0
    hit_step_cap: bool = False


class Agent:
    """A stateless (per-question) agent over the financials DB and 10-K filings."""

    def __init__(self, db_path=DB_PATH, llm: LLM | None = None):
        self.db_path = db_path
        self.llm = llm or LLM()

        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            ddl = get_ddl(conn)

        self.tools: dict[str, Tool] = {
            "run_sql": RunSqlTool(db_path),
            "search_filings": SearchFilingsTool(self.llm),
        }

        self.system_prompt = SYSTEM_PROMPT.format(ddl=ddl)

    def _tool_definitions(self) -> list[dict]:
        return [tool.to_definition() for tool in self.tools.values()]

    def ask(self, question: str) -> Answer:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        evidence: list[Evidence] = []
        latency_s, cost_usd = 0.0, 0.0

        for step in range(MAX_AGENT_STEPS):
            result = self.llm.chat(
                messages, tools=self._tool_definitions(), tool_choice="auto"
            )
            latency_s += result.latency_s
            cost_usd += result.cost_usd
            message = result.message

            if not message.tool_calls:
                return Answer(
                    text=message.content or "",
                    evidence=evidence,
                    latency_s=latency_s,
                    cost_usd=cost_usd,
                    steps=step + 1,
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": message.tool_calls,
                }
            )

            for tool_call in message.tool_calls:
                try:
                    tool = self.tools[tool_call.function.name]
                    args = tool.parse_args(tool_call.function.arguments)
                    tool_text, record = tool.run(args)
                    evidence.append(record)
                except (ValueError, ValidationError, KeyError) as e:
                    tool_text = json.dumps({"error": str(e)})
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": tool_text}
                )

        # Circuit breaker, force a final answer from evidence gathered so far
        logger.warning(
            "Agent hit MAX_AGENT_STEPS=%d for question: %r", MAX_AGENT_STEPS, question
        )
        messages.append(
            {
                "role": "user",
                "content": "Answer now using only the information already gathered above.",
            }
        )
        result = self.llm.chat(messages)
        latency_s += result.latency_s
        cost_usd += result.cost_usd

        return Answer(
            text=result.message.content or "",
            evidence=evidence,
            latency_s=latency_s,
            cost_usd=cost_usd,
            steps=MAX_AGENT_STEPS,
            hit_step_cap=True,
        )
