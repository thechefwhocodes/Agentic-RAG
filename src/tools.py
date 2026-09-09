"""The tools the agent can call.

Each tool's run() returns (llm_payload, evidence): llm_payload is compact text
fed back into the conversation, evidence is a structured record collected by
the agent for grounding/inspection (never self-reported by the model).
"""

import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, Field

from src.llm import LLM
from src.models import Evidence, FilingsEvidence, SqlEvidence
from src.retrieval import FilingsIndex


class Tool(ABC):
    name: str
    description: str
    parameters: type[BaseModel]

    def to_definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters.model_json_schema(),
            },
        }

    def parse_args(self, raw_json: str) -> BaseModel:
        return self.parameters.model_validate_json(raw_json)

    @abstractmethod
    def run(self, args: BaseModel) -> tuple[str, Evidence]:
        """Execute the tool call. Return (text for the LLM, evidence record)."""


class RunSqlArgs(BaseModel):
    sql: str = Field(description="A single read-only SQLite SELECT (or WITH) query.")


class RunSqlTool(Tool):
    name = "run_sql"
    description = (
        "Run a read-only SQL query against the financial database and return "
        "the rows. Use this for any question about revenue, income, balance "
        "sheet items, segment revenue, or geographic revenue."
    )
    parameters = RunSqlArgs

    def __init__(self, db_path: Path):
        # A fresh connection per query, not one held for the tool's lifetime:
        # trivially safe across FastAPI's worker threads with no special
        # sqlite3 flags to reason about, and opening a local read-only file
        # is sub-millisecond.
        self.db_path = db_path

    @staticmethod
    def _check_sql(sql: str) -> None:
        """Defense in depth alongside the read-only connection: reject
        anything but a single SELECT/WITH statement before it ever runs."""
        stripped = sql.strip().rstrip(";").strip()
        if ";" in stripped:
            raise ValueError("Only a single SQL statement is allowed.")
        if not stripped.upper().startswith(("SELECT", "WITH")):
            raise ValueError("Only SELECT queries are supported.")

    def run(self, args: RunSqlArgs) -> tuple[str, SqlEvidence]:
        sql = args.sql
        try:
            self._check_sql(sql)
            with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as conn:
                cursor = conn.execute(sql)
                columns = [d[0] for d in cursor.description]
                rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        except (ValueError, sqlite3.Error) as e:
            # Returned as a tool result, not raised, so the model sees the
            # error and can correct its own query on the next step.
            error = str(e)
            return json.dumps({"error": error}), SqlEvidence(sql=sql, error=error)

        return json.dumps(rows, default=str), SqlEvidence(sql=sql, rows=rows)


class SearchFilingsArgs(BaseModel):
    query: str = Field(description="What to search for in the 10-K filing text.")
    ticker: str | None = Field(
        default=None, description="Optional: restrict to one company (AAPL, MSFT, or GOOGL)."
    )
    fiscal_year: int | None = Field(
        default=None, description="Optional: restrict to one fiscal year."
    )


class SearchFilingsTool(Tool):
    name = "search_filings"
    description = (
        "Search the 10-K filing text for narrative content: risk factors, "
        "strategy, segment definitions, geographic commentary, and "
        "management discussion. Not for numbers already in the database."
    )
    parameters = SearchFilingsArgs

    def __init__(self, llm: LLM, k: int = 8):
        self.llm = llm
        self.k = k
        self.index = FilingsIndex()  # one Chroma client, reused for every call

    def run(self, args: SearchFilingsArgs) -> tuple[str, FilingsEvidence]:
        hits = self.index.search(
            args.query, self.llm, k=self.k, ticker=args.ticker, fiscal_year=args.fiscal_year
        )
        payload = [
            {"source": f"{h.ticker} FY{h.fiscal_year} {h.section} p{h.page}", "text": h.text}
            for h in hits
        ]
        return json.dumps(payload), FilingsEvidence(query=args.query, hits=hits)
