"""Shared typed data models for retrieval hits, tool evidence, and the API."""

from typing import Literal, Union

from pydantic import BaseModel


class SearchHit(BaseModel):
    """One retrieved filing chunk."""

    text: str
    ticker: str
    company: str
    fiscal_year: int
    form_type: str
    section: str
    page: int
    distance: float


class SqlEvidence(BaseModel):
    sql: str
    rows: list[dict] | None = None
    error: str | None = None
    type: Literal["sql"] = "sql"


class FilingsEvidence(BaseModel):
    query: str
    hits: list[SearchHit]
    type: Literal["filings"] = "filings"


Evidence = Union[SqlEvidence, FilingsEvidence]


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Evidence]
    latency_s: float
    cost_usd: float
